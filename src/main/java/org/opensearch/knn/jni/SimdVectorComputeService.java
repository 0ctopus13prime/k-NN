/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

/**
 * A service that computes vector similarity using native SIMD acceleration.
 * This service relies on a shared native library that implements optimized SIMD instructions to achieve faster performance during
 * similarity computations. The library must be properly loaded and available on the system before invoking any methods
 * that depend on native code.
 */
public class SimdVectorComputeService {
    static {
        KNNLibraryLoader.loadSimdLibrary();
    }

    /**
     * Similarity calculation type to passed down to native code.
     */
    public enum SimilarityFunctionType {
        // FP16 Maximum Inner Product. The result will be the same as we acquired from VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT.
        FP16_MAXIMUM_INNER_PRODUCT,
        // FP16 Maximum Inner Product. The result will be the same as we acquired from VectorSimilarityFunction.EUCLIDEAN.
        FP16_L2,
        SQ_IP,
        SQ_L2
    }

    public static native void saveSQSearchContext(
        byte[] quantizedQuery,
        float queryLowerInterval,
        float queryUpperInterval,
        float queryAdditionalCorrection,
        int queryQuantizedComponentSum,
        long[] addressAndSize,
        int nativeFunctionTypeOrd,
        int dimension,
        float centroidDp
    );

    /**
     * With vector ids, performing bulk SIMD similarity calculations and put the results into `scores`.
     *
     * @param internalVectorIds Vectors to load for similarity calculations.
     * @param scores            Results will be put into this array.
     * @param numVectors        The number of valid vector ids in `internalVectorIds`. Therefore, this will put exactly `numVectors` result
     *                          values into `scores`.
     */
    public native static float scoreSimilarityInBulk(int[] internalVectorIds, float[] scores, int numVectors);

    /**
     * Before vector search starts, it persists required information into a storage. Those persisted information will be used during search.
     * This must be called prior to each search.
     *
     * @param query                 Query vector
     * @param addressAndSize        An array describing vector chunks, where each pair of elements represents a chunk.
     *                              addressAndSize[i] is the starting memory address of the j-th chunk,
     *                              and addressAndSize[i + 1] is the size (in bytes) of that chunk where i = 2 * j.
     *                              Ex: addressAndSize[6] is the starting memory address of 3rd chunk, addressAndSize[7] is the size of
     *                              that chunk.
     * @param nativeFunctionTypeOrd Similarity function type index.
     */
    public native static void saveSearchContext(float[] query, long[] addressAndSize, int nativeFunctionTypeOrd);

    /**
     * Perform similarity search on a single vector.
     *
     * @param internalVectorId Vector id
     * @return Similarity score.
     */
    public native static float scoreSimilarity(int internalVectorId);

    /**
     * Similar to {@link #saveSQSearchContext}, but without a memory-mapped region (no addressAndSize).
     * It stores query correction factors, dimension, centroidDp and the function type in the thread-local
     * native context. The vector region (mmapPages/mmapPageSizes) is populated per scoring call by
     * {@link #scoreSimilarityInBulk2} / {@link #scoreSimilarity2} from a Java heap buffer.
     *
     * @param quantizedQuery               Quantized query vector (nibble transposed).
     * @param queryLowerInterval           Query lower interval correction factor.
     * @param queryUpperInterval           Query upper interval correction factor.
     * @param queryAdditionalCorrection    Query additional correction factor.
     * @param queryQuantizedComponentSum   Query quantized component sum.
     * @param nativeFunctionTypeOrd        Similarity function type index.
     * @param dimension                    Vector dimension.
     * @param centroidDp                   Centroid dot-product correction.
     */
    public static native void saveSQSearchContext2(
        byte[] quantizedQuery,
        float queryLowerInterval,
        float queryUpperInterval,
        float queryAdditionalCorrection,
        int queryQuantizedComponentSum,
        int nativeFunctionTypeOrd,
        int dimension,
        float centroidDp
    );

    /**
     * Performing bulk SIMD similarity calculations over a contiguous Java heap buffer and put the results into `scores`.
     * The buffer must hold exactly `numVectors` elements laid out back-to-back, each element being
     * [quantized vector | lowerInterval (float) | upperInterval (float) | additionalCorrection (float) | quantizedComponentSum (int)].
     * Internal vector ids are implicitly 0, 1, ..., numVectors - 1 (i.e. the slot index within the buffer).
     *
     * @param buffer     Contiguous buffer holding `numVectors` elements (vector + correction factors each).
     * @param scores     Results will be put into this array.
     * @param numVectors The number of vectors stored in `buffer`. This will put exactly `numVectors` result
     *                   values into `scores`.
     */
    public native static float scoreSimilarityInBulk2(byte[] buffer, float[] scores, int numVectors);

    /**
     * Single vector variant of {@link #scoreSimilarityInBulk2}. The buffer must hold exactly one element
     * (vector + correction factors).
     *
     * @param buffer Buffer holding one element (vector + correction factors).
     * @return Similarity score.
     */
    public native static float scoreSimilarity2(byte[] buffer);
}
