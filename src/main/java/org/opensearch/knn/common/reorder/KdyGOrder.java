/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common.reorder;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHNSW;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHnswGraph;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIdMapIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

public class KdyGOrder {
    public static void main(String... args) throws IOException {
        final String faissGraphDir = "/Users/kdooyong/workspace/Gorder/kdy";
        final String faissGraphFileName = "_e_165_target_field.faiss";
        final int window = 16;

        try (final Directory directory = new MMapDirectory(Path.of(faissGraphDir));) {
            try (final IndexInput indexInput = directory.openInput(faissGraphFileName, IOContext.DEFAULT);) {
                final FaissIndex faissIndex = FaissIndex.load(indexInput);
                final int numVectors = faissIndex.getTotalNumberOfVectors();
                System.out.println("Total #vectors: " + numVectors);
                if (faissIndex instanceof FaissIdMapIndex idMapIndex) {
                    final FaissHNSW faissHNSW = idMapIndex.getFaissHnsw();
                    final int[] permutation = getPermutation(faissHNSW, indexInput, window);
                    System.out.println(Arrays.toString(permutation));
                } else {
                    throw new IllegalStateException("faissIndex is not FaissIdMapIndex! Actual type: " + faissIndex.getClass());
                }
            }
        }
    }

    private static int[] getPermutation(FaissHNSW faissHNSW, IndexInput indexInput, int window) throws IOException {
        // Prepare
        final FaissHnswGraph faissHnswGraph = new FaissHnswGraph(faissHNSW, indexInput);
        final int numVectors = Math.toIntExact(faissHNSW.getTotalNumberOfVectors());

        // Construct incoming vertexes
        final IncomingVertexes incomingVertexes = IncomingVertexes.collectIncomingVertices(faissHnswGraph);

        // Isolated vectors
        List<Integer> isolatedVertexes = new ArrayList<>();

        // Heap
        final UnitHeap unitHeap = new UnitHeap(numVectors);

        // Bit set
        final BitSet popvExist = new BitSet(numVectors);

        // New permutation
        int orderIndex = 0;
        int[] newPermutation = new int[numVectors];

        // ???
        for (int i = 0; i < numVectors; ++i) {
            unitHeap.linkedList[i].key = incomingVertexes.getDegree(i);
            unitHeap.update[i] = -unitHeap.linkedList[i].key;
        }

        // Reset the unit heap
        //  - Array
        //  - linked list
        //  - head table
        unitHeap.reconstruct();

        // Find the first vertex with the largest indegree
        int veryFirstVertex = -1;
        int maxInDegreeFound = -1;
        for (int i = 0; i < numVectors; i++) {
            final int inDegree = incomingVertexes.getDegree(i);
            if (inDegree > maxInDegreeFound) {
                maxInDegreeFound = inDegree;
                veryFirstVertex = i;
            } else if (inDegree + HnswGraphHelper.getOutDegree(faissHnswGraph, i) == 0) {
                unitHeap.update[i] = Integer.MAX_VALUE / 2;
                isolatedVertexes.add(i);
                unitHeap.deleteElement(i);
            }
        }

        // Push the first vertex and remove it from the queue
        newPermutation[orderIndex++] = veryFirstVertex;
        unitHeap.update[veryFirstVertex] = Integer.MAX_VALUE / 2;
        unitHeap.deleteElement(veryFirstVertex);

        // For incoming vertexes to `veryFirstVertex`
        // e.g.
        // A -> V_e
        //   -> X
        // Then increase counter for A and X
        for (int i = incomingVertexes.startOffset(veryFirstVertex), limit1 = incomingVertexes.endOffset(veryFirstVertex); i < limit1; ++i) {
            final int u = incomingVertexes.incomingVertices[i];
            if (unitHeap.update[u] == 0) {
                unitHeap.increaseKey(u);
            } else {
                unitHeap.update[u] += 1;
            }

            // Increase sibling count
            HnswGraphHelper.forAllOutgoingNodes(
                faissHnswGraph, u, 1, (w) -> {
                    if (unitHeap.update[w] == 0) {
                        unitHeap.increaseKey(w);
                    } else {
                        unitHeap.update[w] += 1;
                    }
                }
            );
        }  // End for

        // For the vertexes pointed by vertex of `veryFirstVertex`
        HnswGraphHelper.forAllOutgoingNodes(
            faissHnswGraph, veryFirstVertex, 0, (w) -> {
                if (unitHeap.update[w] == 0) {
                    unitHeap.increaseKey(w);
                } else {
                    unitHeap.update[w] += 1;
                }
            }
        );

        int count = 0;
        final AtomicReference<Boolean> hasV = new AtomicReference<>(false);
        while (count < numVectors - 1 - isolatedVertexes.size()) {
            // Extract max
            final int maxVertex = unitHeap.extractMax();
            // How many elements we pulled from Q so far?
            ++count;

            // Append max vertex and invalidate `update`
            newPermutation[orderIndex++] = maxVertex;
            unitHeap.update[maxVertex] = Integer.MAX_VALUE / 2;

            // Is there a vertex we should exclude from the window?
            // -1 -> no, we don't
            int popv;
            if (count - window >= 0) {
                popv = newPermutation[count - window];
            } else {
                popv = -1;
            }

            if (popv >= 0) {
                // For vertexes pointed by `popv`
                HnswGraphHelper.forAllOutgoingNodes(
                    faissHnswGraph, popv, 0, (w) -> {
                        unitHeap.update[w] -= 1;
                    }
                );

                // For incoming vertexes to `popv`
                for (int i = incomingVertexes.startOffset(popv), limit1 = incomingVertexes.endOffset(popv); i < limit1; ++i) {
                    final int u = incomingVertexes.incomingVertices[i];
                    unitHeap.update[u]--;
                    if (HnswGraphHelper.getOutDegree(faissHnswGraph, u) > 1) {
                        hasV.set(false);
                        HnswGraphHelper.forAllOutgoingNodes(
                            faissHnswGraph, maxVertex, 0, (w) -> {
                                if (w == maxVertex) {
                                    hasV.set(true);
                                    // Stop the loop
                                    return false;
                                }
                                return true;
                            }
                        );

                        if (hasV.get() == false) {
                            // If `popv` (e.g. v_b) and v (e.g. v_max) are NOT sibling, then do below:
                            HnswGraphHelper.forAllOutgoingNodes(
                                faissHnswGraph, maxVertex, 0, (w) -> {
                                    unitHeap.update[w] -= 1;
                                }
                            );
                        } else {
                            popvExist.set(u);
                        }  // End if
                    }  // End if
                }  // End for
            }  // End if

            // For the outgoing vertexes of `v`
            HnswGraphHelper.forAllOutgoingNodes(
                faissHnswGraph, maxVertex, 0, (w) -> {
                    if (unitHeap.update[w] == 0) {
                        unitHeap.increaseKey(w);
                    } else {
                        unitHeap.update[w] += 1;
                    }
                }
            );

            // For the incoming vertexes to `v`
            for (int i = incomingVertexes.startOffset(maxVertex), limit1 = incomingVertexes.endOffset(maxVertex); i < limit1; ++i) {
                final int u = incomingVertexes.incomingVertices[i];
                if (unitHeap.update[u] == 0) {
                    unitHeap.increaseKey(u);
                } else {
                    unitHeap.update[u] += 1;
                }

                // If `popv` and `v` are not siblings, then increment the counter for all its siblings
                if (popvExist.get(u) == false) {
                    HnswGraphHelper.forAllOutgoingNodes(
                        faissHnswGraph, u, 1, (w) -> {
                            if (unitHeap.update[w] == 0) {
                                unitHeap.increaseKey(w);
                            } else {
                                unitHeap.update[w] += 1;
                            }
                        }
                    );
                } else {
                    popvExist.set(u, false);
                }
            }  // End for
        }  // End while

        // Insert isolated vertexes
        for (final int isolatedIndex : isolatedVertexes) {
            newPermutation[orderIndex++] = isolatedIndex;
        }

        // Create a new permutation inplace : old vertex -> new vertex
        for (int i = newPermutation[0], j = 0; j < numVectors; ++j) {
            final int next = newPermutation[i];
            newPermutation[i] = j;
            i = next;
        }

        return newPermutation;
    }
}
