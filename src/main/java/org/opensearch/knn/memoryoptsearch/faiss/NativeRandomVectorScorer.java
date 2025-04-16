/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.Getter;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

public class NativeRandomVectorScorer implements RandomVectorScorer {
    private final int maxOrd;
    @Getter
    private final Arena arena;
    private final MemorySegment query;
    private final long flatVectorsManagerAddress;
    private final int dimension;
    private int numNeighbors;
    private int nextNeighborIndex;
    private MemorySegment neighborList;
    private MemorySegment scores;

    public NativeRandomVectorScorer(
        final int maxOrd,
        final long flatVectorsManagerAddress,
        final int dimension,
        final float[] target
    ) {
        this.maxOrd = maxOrd;
        this.flatVectorsManagerAddress = flatVectorsManagerAddress;
        this.dimension = dimension;
        this.arena = Arena.ofShared();
        this.query = arena.allocate(Float.BYTES * target.length, ValueLayout.JAVA_FLOAT.byteAlignment());
        for (int i = 0; i < target.length; i++) {
            this.query.setAtIndex(ValueLayout.JAVA_FLOAT, i, target[i]);
        }
    }

    public void close() {
        arena.close();
    }

    @Override
    public float score(int internalVectorId) throws IOException {
        if (nextNeighborIndex < numNeighbors) {
            final int expectedId = neighborList.getAtIndex(ValueLayout.JAVA_INT, nextNeighborIndex);
            if (expectedId == internalVectorId) {
                return scores.getAtIndex(ValueLayout.JAVA_FLOAT, nextNeighborIndex++);
            }
        }

        try {
            final float singleScore = (float) singleScoreMethodHandle.invoke(
                flatVectorsManagerAddress, query, internalVectorId, dimension);
            KdyPrint.println("__________ single score=" + singleScore + ", vid=" + internalVectorId);
            return singleScore;
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    public void bulkScoring(final MemorySegment neighborList, final int numNeighbors) {
        this.neighborList = neighborList;
        this.nextNeighborIndex = 0;
        this.numNeighbors = numNeighbors;
        if (scores == null || scores.byteSize() < (Float.BYTES * numNeighbors)) {
            scores = arena.allocate(2 * numNeighbors * Float.BYTES, 64);
            KdyPrint.println("___________________ size(scores)==" + scores.byteSize());
        }
        try {
            bulkScoreMethodHandle.invoke(
                flatVectorsManagerAddress, query, neighborList, scores, numNeighbors, dimension);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public int maxOrd() {
        return maxOrd;
    }

    static final MethodHandle singleScoreMethodHandle;

    static final MethodHandle bulkScoreMethodHandle;

    static {
        final Linker linker = Linker.nativeLinker();
        final SymbolLookup lookup = SymbolLookup.loaderLookup();

        bulkScoreMethodHandle = linker.downcallHandle(
            lookup.find("Java_org_opensearch_knn_jni_FaissService_bulkScoring").orElseThrow(),
            FunctionDescriptor.ofVoid(
                ValueLayout.JAVA_LONG,  // manager address
                ValueLayout.ADDRESS,  // query
                ValueLayout.ADDRESS,  // neighbor list
                ValueLayout.ADDRESS,  // scores
                ValueLayout.JAVA_INT,  // size
                ValueLayout.JAVA_INT  // dimension
            )
        );

        singleScoreMethodHandle = linker.downcallHandle(
            lookup.find("Java_org_opensearch_knn_jni_FaissService_singleScoring").orElseThrow(),
            FunctionDescriptor.of(
                ValueLayout.JAVA_FLOAT,  // return float score
                ValueLayout.JAVA_LONG,  // manager address
                ValueLayout.ADDRESS,  // query
                ValueLayout.JAVA_INT, // vector id
                ValueLayout.JAVA_INT  // dimension
            )
        );
    }
}
