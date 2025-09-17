/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.Getter;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.jni.FaissService;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public class NativeRandomVectorScorer2 implements RandomVectorScorer {
    private final static ThreadLocal<Arena> ARENA = ThreadLocal.withInitial(Arena::ofShared);

    private final int maxOrd;
    @Getter
    private final Arena arena;
    private final MemorySegment query;
    private final int dimension;
    private int numNeighbors;
    private int nextNeighborIndex;
    private MemorySegment neighborList;
    private MemorySegment scores;
    private MMapByteVectorValues mmapByteVectorValues;

    public NativeRandomVectorScorer2(final int maxOrd, final int dimension, final float[] target, final MMapByteVectorValues vectorValues) {
        this.maxOrd = maxOrd;
        this.dimension = dimension;
        // this.arena = Arena.ofShared();
        this.arena = ARENA.get();
        this.query = arena.allocate(Float.BYTES * target.length, ValueLayout.JAVA_FLOAT.byteAlignment());
        for (int i = 0; i < target.length; i++) {
            this.query.setAtIndex(ValueLayout.JAVA_FLOAT, i, target[i]);
        }
        this.mmapByteVectorValues = vectorValues;
    }

    public void close() {
        // arena.close();
    }

    @Override
    public float score(int internalVectorId) throws IOException {
        if (nextNeighborIndex < numNeighbors) {
            final int expectedId = neighborList.getAtIndex(ValueLayout.JAVA_INT, nextNeighborIndex);
            if (expectedId == internalVectorId) {
                return scores.getAtIndex(ValueLayout.JAVA_FLOAT, nextNeighborIndex++);
            }
        }

        if (scores == null || scores.byteSize() < 16) {
            scores = arena.allocate(16 * Float.BYTES, 64);
        }
        neighborList.setAtIndex(ValueLayout.OfInt.JAVA_INT, 0, internalVectorId);
        FaissService.bulkScoring2(query.address(), neighborList.address(), 1, scores.address(), 0, dimension);
        return scores.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
    }

    public void bulkScoring(final MemorySegment neighborList, final int numNeighbors) {
        this.neighborList = neighborList;
        this.nextNeighborIndex = 0;
        this.numNeighbors = numNeighbors;
        if (scores == null || scores.byteSize() < (Float.BYTES * numNeighbors)) {
            scores = arena.allocate(2 * numNeighbors * Float.BYTES, 64);
        }
        FaissService.bulkScoring2(
            query.address(),
            neighborList.address(),
            numNeighbors,
            mmapByteVectorValues.getMemorySegments()[0].address() + mmapByteVectorValues.getBaseOffset(),
            scores.address(),
            (int) mmapByteVectorValues.getOneVectorByteSize()
        );
    }

    @Override
    public int maxOrd() {
        return maxOrd;
    }
}
