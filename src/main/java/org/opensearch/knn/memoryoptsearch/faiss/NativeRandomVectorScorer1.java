/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.Getter;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.jni.FaissService;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

public class NativeRandomVectorScorer1 implements RandomVectorScorer {
    private final int maxOrd;
    @Getter
    private final Arena arena;
    private final MemorySegment query;
    private final int dimension;
    private int numNeighbors;
    private int nextNeighborIndex;
    private int[] neighborList;
    private MemorySegment scores;
    private final ByteVectorValues bytes;
    private final byte[][] tmpBuffers;

    public NativeRandomVectorScorer1(
        final int maxOrd, final int dimension, final float[] target, final ByteVectorValues bytes
    ) {
        this.maxOrd = maxOrd;
        this.dimension = dimension;
        this.arena = Arena.ofShared();
        this.query = arena.allocate(Float.BYTES * target.length, ValueLayout.JAVA_FLOAT.byteAlignment());
        for (int i = 0; i < target.length; i++) {
            this.query.setAtIndex(ValueLayout.JAVA_FLOAT, i, target[i]);
        }
        this.bytes = bytes;
        this.tmpBuffers = new byte[4][];
    }

    public void close() {
        arena.close();
    }

    @Override
    public float score(int internalVectorId) throws IOException {
        if (nextNeighborIndex < numNeighbors) {
            final int expectedId = neighborList[nextNeighborIndex];
            if (expectedId == internalVectorId) {
                return scores.getAtIndex(ValueLayout.JAVA_FLOAT, nextNeighborIndex++);
            }
        }

        tmpBuffers[0] = bytes.vectorValue(internalVectorId);
        if (scores == null || scores.byteSize() < 16) {
            scores = arena.allocate(16 * Float.BYTES, 64);
        }
        FaissService.bulkScoring1(query.address(), tmpBuffers, 1, scores.address(), 0, dimension);
        return scores.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
    }

    public void bulkScoring(final int[] neighborList, final int numNeighbors) {
        try {
            this.neighborList = neighborList;
            this.nextNeighborIndex = 0;
            this.numNeighbors = numNeighbors;
            if (scores == null || scores.byteSize() < (Float.BYTES * numNeighbors)) {
                scores = arena.allocate(2 * numNeighbors * Float.BYTES, 64);
            }

            int i = 0;
            while (i < numNeighbors) {
                final int numVectorsToPass = Math.min(numNeighbors - i, 4);
                for (int j = 0; j < numVectorsToPass; ++j) {
                    final int vectorId = neighborList[i + j];
                    tmpBuffers[j] = bytes.vectorValue(vectorId);
                }
                FaissService.bulkScoring1(query.address(), tmpBuffers, numVectorsToPass, scores.address(), i, dimension);
                i += 4;
            }
        } catch (Throwable e) {
            throw new RuntimeException("!!!!!!!!!!!!!!!!!!!!!!!!!!!", e);
        }
    }

    @Override
    public int maxOrd() {
        return maxOrd;
    }

    static final MethodHandle bulkScoreMethodHandle;

    static {
        final Linker linker = Linker.nativeLinker();
        final SymbolLookup lookup = SymbolLookup.loaderLookup();

        bulkScoreMethodHandle = linker.downcallHandle(lookup.find("Java_org_opensearch_knn_jni_FaissService_bulkScoring1").orElseThrow(),
                                                      FunctionDescriptor.ofVoid(
                                                          ValueLayout.ADDRESS,  // query
                                                          ValueLayout.ADDRESS,  // 4 byte[]
                                                          ValueLayout.JAVA_INT, // Num vectors to calculate
                                                          ValueLayout.ADDRESS,  // scores
                                                          ValueLayout.JAVA_INT, // index in scores
                                                          ValueLayout.JAVA_INT  // dimension
                                                      )
        );
    }
}
