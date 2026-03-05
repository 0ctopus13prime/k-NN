/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.jni.SimdVectorComputeService;

import java.io.IOException;

/**
 * A {@link RandomVectorScorer} implementation that offloads vector similarity computation
 * to native SIMD-optimized code for maximum performance.
 * <p>
 * This class initializes a native search context based on the given query vector and
 * memory-mapped vector chunks, and delegates all similarity scoring operations to the
 * {@link SimdVectorComputeService}. The underlying native library is expected to
 * leverage SIMD instructions (e.g., AVX, AVX512, or NEON) to accelerate computations.
 */
public class BBQNativeRandomVectorScorer implements RandomVectorScorer {
    private final int maxOrd;

    public BBQNativeRandomVectorScorer(
        final byte[] quantized,
        final OptimizedScalarQuantizer.QuantizationResult correctionFactors,
        final long[] addressAndSize,
        final int maxOrd,
        final int dimension,
        final float centroidDp
    ) {
        this.maxOrd = maxOrd;
        SimdVectorComputeService.saveBBQSearchContext(
            quantized,
            correctionFactors.lowerInterval(),
            correctionFactors.upperInterval(),
            correctionFactors.additionalCorrection(),
            correctionFactors.quantizedComponentSum(),
            addressAndSize,
            SimdVectorComputeService.SimilarityFunctionType.BBQ_IP.ordinal(),
            dimension,
            centroidDp
        );
    }

    /**
     * Computes similarity scores for multiple vectors in bulk using native SIMD code.
     *
     * @param internalVectorIds the array of internal vector IDs to score
     * @param scores            the output array to store computed similarity scores
     * @param numVectors        the number of vectors to process
     */
    @Override
    public void bulkScore(final int[] internalVectorIds, final float[] scores, final int numVectors) {
        SimdVectorComputeService.scoreSimilarityInBulk(internalVectorIds, scores, numVectors);
    }

    /**
     * Computes the similarity score for a single vector using native SIMD code.
     *
     * @param internalVectorId the internal vector ID to score
     * @return the computed similarity score
     * @throws IOException if the native scoring operation fails
     */
    @Override
    public float score(final int internalVectorId) throws IOException {
        return SimdVectorComputeService.scoreSimilarity(internalVectorId);
    }

    /**
     * Returns the maximum vector id for scoring.
     *
     * @return the maximum vector id
     */
    @Override
    public int maxOrd() {
        return maxOrd;
    }

    /**
     * Maps an internal vector ordinal to its corresponding document ID.
     *
     * @param ord the internal vector id
     * @return the document ID associated with the given vector id
     */
    @Override
    public int ordToDoc(int ord) {
        // TODO : Dense case only
        return ord;
    }

    /**
     * Returns a filtered {@link Bits} view representing accepted documents.
     *
     * @param acceptDocs the bit set of accepted documents
     * @return a {@link Bits} object describing acceptable vector ids for scoring
     */
    @Override
    public Bits getAcceptOrds(Bits acceptDocs) {
        return null;
    }
}
