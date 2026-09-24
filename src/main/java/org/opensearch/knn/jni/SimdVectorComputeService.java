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
     * @param query                  Query vector
     * @param addressAndSize         An array describing vector chunks, where each pair of elements represents a chunk.
     *                               addressAndSize[i] is the starting memory address of the j-th chunk,
     *                               and addressAndSize[i + 1] is the size (in bytes) of that chunk where i = 2 * j.
     *                               Ex: addressAndSize[6] is the starting memory address of 3rd chunk, addressAndSize[7] is the size of
     *                               that chunk.
     * @param nativeFunctionTypeOrd  Similarity function type index.
     */
    public native static void saveSearchContext(float[] query, long[] addressAndSize, int nativeFunctionTypeOrd);

    /**
     * Perform similarity search on a single vector.
     *
     * @param internalVectorId Vector id
     * @return Similarity score.
     */
    public native static float scoreSimilarity(int internalVectorId);

    // ===== ClusterANN bulk operations (pure functions, no state, thread-safe) =====

    /**
     * Batch quantized dot product for block-columnar ADC scoring. Supports 1-bit, 2-bit, and 4-bit document
     * quantization against a 4-bit query; the native side batches several documents per SIMD pass and shares the
     * query loads across them.
     *
     * <ul>
     *   <li>1-bit / 2-bit: documents are transposed bit planes ({@code packAsBinary} / {@code transposeDibit}), the
     *       query is transposed with {@code OptimizedScalarQuantizer.transposeHalfByte} into 4 planes. Same math as
     *       {@code VectorUtil.int4BitDotProduct} / {@code int4DibitDotProduct}. Query length
     *       {@code >= 4 * (bytesPerCode / docBits)}.</li>
     *   <li>4-bit: documents are split-half nibbles ({@code ScalarBitEncoding.FOUR_BIT}, the Lucene104
     *       {@code PACKED_NIBBLE} layout), the query is unpacked 4-bit codes zero padded to {@code 2 * bytesPerCode}.
     *       Same math as {@code QuantizedVectorReader.int4NibbleDotProduct}. Uses VNNI / udot where available.</li>
     * </ul>
     *
     * <p>For {@code k in [0, numOffsets)}: {@code results[offsets[k]] = dot(query, record offsets[k])}.
     * Entries of {@code results} not addressed by {@code offsets} are left untouched. Offsets are validated natively;
     * an out-of-range offset throws instead of reading out of bounds.
     *
     * @param query           quantized query in the layout described above
     * @param docCodesBatch   contiguous packed codes, one record of {@code bytesPerCode} bytes per vector
     * @param offsets         record indexes (into {@code docCodesBatch}) to score
     * @param numOffsets      number of valid entries in {@code offsets}
     * @param results         output raw dot products, indexed by record index
     * @param bytesPerCode    packed bytes per vector ({@code ScalarBitEncoding.docPackedBytes(dim)})
     * @param docBits         document quantization bits (1, 2, or 4)
     */
    public static native void bulkQuantizedDotProduct(
        byte[] query,
        byte[] docCodesBatch,
        int[] offsets,
        int numOffsets,
        float[] results,
        int bytesPerCode,
        int docBits
    );

    /**
     * Name of the compiled-in kernel behind {@link #bulkQuantizedDotProduct}: {@code avx512}, {@code avx2},
     * {@code neon} or {@code scalar}. Diagnostic only.
     */
    public static native String bulkQuantizedDotProductKernel();

    /**
     * Compute distances from one vector to multiple centroids.
     * Returns index of nearest centroid.
     *
     * @param vector       query vector
     * @param centroids    flat array of centroids (numCentroids × dimension)
     * @param distances    output distances
     * @param dimension    vector dimension
     * @param numCentroids number of centroids
     * @param metricOrd    0=L2, 1=DOT_PRODUCT
     * @return index of nearest centroid
     */
    public static native int bulkCentroidDistance(
        float[] vector,
        float[] centroids,
        float[] distances,
        int dimension,
        int numCentroids,
        int metricOrd
    );

    /**
     * Compute SOAR distances from a vector to candidate centroids.
     * SOAR distance = L2² + lambda × (projection² / ||residual||²)
     *
     * @param vector          the vector
     * @param primaryCentroid the primary assigned centroid
     * @param candidates      flat array of candidate centroids (numCandidates × dimension)
     * @param distances       output SOAR distances
     * @param dimension       vector dimension
     * @param numCandidates   number of candidates
     * @param soarLambda      SOAR lambda parameter
     */
    public static native void bulkSOARDistance(
        float[] vector,
        float[] primaryCentroid,
        float[] candidates,
        float[] distances,
        int dimension,
        int numCandidates,
        float soarLambda
    );
}
