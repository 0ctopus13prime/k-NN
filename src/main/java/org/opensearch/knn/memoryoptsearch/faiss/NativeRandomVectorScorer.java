/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.jni.SimdVectorComputeService;

import java.io.IOException;

public class NativeRandomVectorScorer implements RandomVectorScorer {
    private long[] addressAndSize;
    private int maxOrd;
    private int nativeFunctionTypeOrd;

    public NativeRandomVectorScorer(
        final float[] query,
        final KnnVectorValues knnVectorValues,
        final SimdVectorComputeService.SimilarityFunctionType similarityFunctionType
    ) {
        this(knnVectorValues, similarityFunctionType);
        SimdVectorComputeService.saveSearchContext(query, addressAndSize, nativeFunctionTypeOrd);
    }

    public NativeRandomVectorScorer(
        final KnnVectorValues knnVectorValues, final SimdVectorComputeService.SimilarityFunctionType similarityFunctionType
    ) {
        final MMapVectorValues mmapVectorValues = (MMapVectorValues) knnVectorValues;
        this.addressAndSize = mmapVectorValues.getAddressAndSize();
        this.nativeFunctionTypeOrd = similarityFunctionType.ordinal();
        this.maxOrd = knnVectorValues.size();
    }

    @Override
    public void bulkScore(final int[] internalVectorIds, final float[] scores, final int numVectors) {
        SimdVectorComputeService.bulkDistanceCalculation(internalVectorIds, scores, numVectors);
    }

    @Override
    public float score(final int internalVectorId) throws IOException {
        return SimdVectorComputeService.scoreSingleVector(internalVectorId);
    }

    @Override
    public int maxOrd() {
        return maxOrd;
    }
}
