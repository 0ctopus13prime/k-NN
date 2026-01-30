/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
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

public class GOrderVectorReorder {
    public static void main(String... args) throws IOException {
        final String faissGraphDir = "/Users/kdooyong/workspace/Gorder/kdy";
        final String faissGraphFileName = "_e_165_target_field.faiss";
        final int window = 16;

        try (final Directory directory = new MMapDirectory(Path.of(faissGraphDir));) {
            try (final IndexInput indexInput = directory.openInput(faissGraphFileName, IOContext.DEFAULT)) {
                final FaissIndex faissIndex = FaissIndex.load(indexInput);
                final int numVectors = faissIndex.getTotalNumberOfVectors();
                System.out.println("Total #vectors: " + numVectors);
                if (faissIndex instanceof FaissIdMapIndex idMapIndex) {
                    // Get the permutation
                    final FaissHNSW faissHNSW = idMapIndex.getFaissHnsw();
                    // Permutation -> newOrd2Old
                    final int[] permutation = getPermutation(faissHNSW, indexInput, window);
                    final ReorderOrdMap reorderOrdMap = new ReorderOrdMap(permutation);
                    System.out.println(Arrays.toString(permutation));

                    // Transform the index
                    if (true) {
                        try (final IndexOutput indexOutput = directory.createOutput(faissGraphFileName + ".reorder", IOContext.DEFAULT)) {
                            FaissIndexReorderTransformer.transform(faissIndex, indexInput, indexOutput, reorderOrdMap);
                        }
                    }
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
        unitHeap.deleteElement(veryFirstVertex);

        final AtomicReference<Boolean> skipIterationForCommonNeighbor = new AtomicReference<>(false);
        final int untilOrderIndex = (numVectors - isolatedVertexes.size());
        while (true) {
            final int ve = newPermutation[orderIndex - 1];
            final int vb = (orderIndex > (window + 1)) ? newPermutation[orderIndex - window - 1] : -1;

            // For incoming vertexes to `ve`
            // e.g.
            // A -> V_e -> P
            // -> X
            // Then increase counter for A, X and P.
            for (int i = incomingVertexes.startOffset(ve), limit1 = incomingVertexes.endOffset(ve); i < limit1; ++i) {
                // Increase incoming vertex count
                final int u = incomingVertexes.incomingVertices[i];
                if (unitHeap.update[u] == 0) {
                    unitHeap.increaseKey(u);
                } else {
                    unitHeap.update[u] += 1;
                }

                // Increase sibling count
                if (HnswGraphHelper.getOutDegree(faissHnswGraph, u) > 1) {
                    skipIterationForCommonNeighbor.set(false);
                    if (vb != -1) {
                        // We have a leaving node from the window
                        HnswGraphHelper.forAllOutgoingNodes(faissHnswGraph, u, 0, (w) -> {
                            if (w == vb) {
                                skipIterationForCommonNeighbor.set(true);
                                // Stop the loop
                                return false;
                            }
                            return true;
                        });
                    }

                    if (skipIterationForCommonNeighbor.get() == false) {
                        // If `popv` (e.g. v_b) and v (e.g. v_max) are NOT sibling, then do below:
                        HnswGraphHelper.forAllOutgoingNodes(faissHnswGraph, u, 0, (w) -> { unitHeap.update[w] += 1; });
                    } else {
                        popvExist.set(u);
                    }  // End if
                }
            }  // End for

            // For the vertexes pointed by vertex of `veryFirstVertex`
            HnswGraphHelper.forAllOutgoingNodes(faissHnswGraph, ve, 0, (w) -> {
                if (unitHeap.update[w] == 0) {
                    unitHeap.increaseKey(w);
                } else {
                    unitHeap.update[w] += 1;
                }
            });

            if (vb != -1) {
                // For incoming vertexes to `vb`
                // e.g.
                // A -> V_b
                // -> X
                // Then increase counter for A and X
                for (int i = incomingVertexes.startOffset(vb), limit1 = incomingVertexes.endOffset(vb); i < limit1; ++i) {
                    final int u = incomingVertexes.incomingVertices[i];
                    unitHeap.update[u] -= 1;

                    // Increase sibling count
                    if (HnswGraphHelper.getOutDegree(faissHnswGraph, u) > 1) {
                        if (popvExist.get(u) == false) {
                            HnswGraphHelper.forAllOutgoingNodes(faissHnswGraph, u, 1, (w) -> { unitHeap.update[w] -= 1; });
                        } else {
                            popvExist.set(u, false);
                        }
                    }
                }  // End for

                // For the vertexes pointed by vertex of `veryFirstVertex`
                HnswGraphHelper.forAllOutgoingNodes(faissHnswGraph, vb, 0, (w) -> { unitHeap.update[w] -= 1; });
            }  // End if

            if (orderIndex < untilOrderIndex) {
                final int vmax = unitHeap.extractMax();
                newPermutation[orderIndex++] = vmax;
            } else {
                break;
            }
        }  // End while

        // Insert isolated vertexes
        for (final int isolatedIndex : isolatedVertexes) {
            newPermutation[orderIndex++] = isolatedIndex;
        }

        return newPermutation;
    }
}
