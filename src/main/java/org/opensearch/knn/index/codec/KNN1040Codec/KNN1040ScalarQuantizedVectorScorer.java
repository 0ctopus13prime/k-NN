/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorScorer;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer;
import org.opensearch.knn.jni.SimdVectorComputeService;
import org.opensearch.knn.memoryoptsearch.MemorySegmentAddressExtractorUtil;
import org.opensearch.knn.memoryoptsearch.faiss.WrappedFloatVectorValues;

import java.io.IOException;

import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
import static org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;

/**
 * A specialized {@link Lucene104ScalarQuantizedVectorScorer} that leverages
 * FAISS-style SIMD-accelerated scoring for scalar-quantized vectors with fallback.
 * Will hit fallback only if we cannot use native bulk simd scorer.
 *
 * <p>This scorer attempts to use a native SIMD-backed bulk scoring path when:
 * <ul>
 *   <li>The underlying vector values are {@link QuantizedByteVectorValues}</li>
 *   <li>The backing storage can expose a raw memory address</li>
 *   <li>The scalar encoding matches the expected FAISS-compatible format</li>
 * </ul>
 *
 * <p>If these conditions are not met, it falls back to the default Lucene scoring implementation.
 *
 * <p>The SIMD path uses a precomputed search context and performs scoring in native code
 * (e.g., AVX-512), significantly improving throughput for large-scale vector search.
 *
 * <p>All scorers returned by {@link #getRandomVectorScorer} are wrapped with
 * {@link PrefetchableRandomVectorScorer} to prefetch vector data ahead of bulk scoring
 * operations, improving cache locality and reducing I/O latency during graph traversal.
 */
@Log4j2
public class KNN1040ScalarQuantizedVectorScorer extends Lucene104ScalarQuantizedVectorScorer {
    /**
     * POC experiment switch. One of 'MMap' (baseline, bulk SIMD over mmap'd region), 'Fallback' (Lucene's
     * pure Java scoring) or 'JavaArray' (copy vectors into a heap byte[] then bulk SIMD via JNI).
     * Set once during {@link org.opensearch.knn.index.KNNIndexShard#warmup()} by reading
     * /tmp/bulk-simd-experiment. Defaults to the baseline behavior.
     */
    public static volatile String EXP_TYPE = "MMap";

    /**
     * Creates a new scorer that wraps a non-quantized delegate scorer.
     *
     * @param delegate fallback scorer used when SIMD acceleration is not applicable
     */
    public KNN1040ScalarQuantizedVectorScorer(final FlatVectorsScorer delegate) {
        super(delegate);
    }

    /**
     * Returns a {@link RandomVectorScorer} for the given query vector.
     *
     * <p><b>Important:</b> This method only supports {@link QuantizedByteVectorValues}. It will fail
     * with an exception if called with raw (non-quantized) vector values such as
     * {@code OffHeapFloatVectorValues}. Callers must ensure that this scorer is not used as the
     * scorer for raw vector formats (e.g., {@link org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat}).
     *
     * <p>This method attempts to construct a SIMD-accelerated scorer when the input vectors
     * are quantized and backed by memory that can be accessed directly (e.g., via a memory segment).
     * Otherwise, it falls back to the parent's quantized scoring implementation.
     *
     * @param similarityFunction the similarity function (e.g., inner product or L2)
     * @param vectorValues       the quantized vector storage (must be {@link QuantizedByteVectorValues})
     * @param target             the query vector (float32)
     * @return a scorer capable of computing similarity scores
     * @throws IOException if an error occurs while accessing vector data
     */
    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues,
        float[] target
    ) throws IOException {
        // For the sparse case, KnnVectorValues having `QuantizedByteVectorValues` might be wrapped to support
        // vector ordinal to doc id mapping. For the dense case, it's not needed as vector ordinal is always the same
        // as doc id.
        if (vectorValues instanceof WrappedFloatVectorValues) {
            vectorValues = WrappedFloatVectorValues.getBottomFloatVectorValues(vectorValues);
        }

        final QuantizedByteVectorValues quantizedByteVectorValues;
        if (vectorValues instanceof QuantizedByteVectorValues) {
            quantizedByteVectorValues = (QuantizedByteVectorValues) vectorValues;
        } else {
            // Extract QuantizedByteVectorValues from `vectorValues`.
            // This should not be null, otherwise it can't get entroid + correction factors.
            quantizedByteVectorValues = KNN1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(vectorValues);
        }

        return new PrefetchableRandomVectorScorer(getScorer(similarityFunction, quantizedByteVectorValues, target));
    }

    private RandomVectorScorer.AbstractRandomVectorScorer getScorer(
        final VectorSimilarityFunction similarityFunction,
        final QuantizedByteVectorValues quantizedByteVectorValues,
        final float[] target
    ) throws IOException {
        if ("MMap".equals(EXP_TYPE)) {
            // Baseline: MMap + bulk SIMD (existing path)
            final IndexInput indexInput = quantizedByteVectorValues.getSlice();
            final long[] addressAndSize = MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(indexInput, 0, indexInput.length());
            if (addressAndSize != null) {
                // Try bulk SIMD
                return bulkSimdRandomVectorScorer(quantizedByteVectorValues, target, addressAndSize, similarityFunction);
            }

            // Fallback
            log.warn("Bulk SIMD for SQ is not supported, falling back to Lucene's random vector scorer");
            return (RandomVectorScorer.AbstractRandomVectorScorer) super.getRandomVectorScorer(
                similarityFunction,
                quantizedByteVectorValues,
                target
            );
        }

        if ("JavaArray".equals(EXP_TYPE)) {
            // Experiment: copy vectors into a Java heap byte[] then bulk SIMD via JNI.
            // We deliberately ignore any extractable MMap address so the experiment reflects the "no MMap" world.
            return javaArrayBulkSimdRandomVectorScorer(quantizedByteVectorValues, target, similarityFunction);
        }

        // Experiment: 'Fallback', Lucene's pure Java scoring
        return (RandomVectorScorer.AbstractRandomVectorScorer) super.getRandomVectorScorer(
            similarityFunction,
            quantizedByteVectorValues,
            target
        );
    }

    /**
     * Builds a SIMD-accelerated scorer using quantized vectors and a precomputed query.
     *
     * <p>This method:
     * <ol>
     *   <li>Validates the scalar encoding format</li>
     *   <li>Quantizes the query vector into the same representation as stored vectors</li>
     *   <li>Applies transformations (e.g., nibble transposition) required for SIMD efficiency</li>
     *   <li>Initializes a native SIMD search context</li>
     * </ol>
     *
     * <p>The resulting scorer uses bulk SIMD instructions to compute similarity scores.
     *
     * @param quantizedByteVectorValues the quantized vector storage
     * @param target                    the query vector (float32)
     * @param addressAndSize            raw memory address and size of vector data
     * @param similarityFunction        similarity function to use
     * @return a SIMD-accelerated scorer
     * @throws IOException if quantization or initialization fails
     */
    private BulkSimdRandomVectorScorer bulkSimdRandomVectorScorer(
        final QuantizedByteVectorValues quantizedByteVectorValues,
        final float[] target,
        final long[] addressAndSize,
        final VectorSimilarityFunction similarityFunction
    ) throws IOException {
        // Quantize + transpose the query vector
        final QuantizedQuery quantizedQuery = quantizeQuery(quantizedByteVectorValues, target);

        // Return Bulk SIMD scorer
        return new BulkSimdRandomVectorScorer(
            quantizedQuery.targetQuantized,
            quantizedQuery.targetCorrectiveTerms,
            addressAndSize,
            quantizedByteVectorValues,
            similarityFunction,
            target.length,
            quantizedByteVectorValues.getCentroidDP()
        );
    }

    /**
     * Builds a SIMD-accelerated scorer that copies vectors into a Java heap buffer per scoring call
     * instead of passing a raw MMap pointer. Used for the 'JavaArray' POC experiment.
     *
     * <p>Query preparation is identical to {@link #bulkSimdRandomVectorScorer}; only the vector
     * source differs.
     *
     * @param quantizedByteVectorValues the quantized vector storage
     * @param target                    the query vector (float32)
     * @param similarityFunction        similarity function to use
     * @return a SIMD-accelerated scorer backed by heap-copied vectors
     * @throws IOException if quantization or initialization fails
     */
    private JavaArrayBulkSimdRandomVectorScorer javaArrayBulkSimdRandomVectorScorer(
        final QuantizedByteVectorValues quantizedByteVectorValues,
        final float[] target,
        final VectorSimilarityFunction similarityFunction
    ) throws IOException {
        // Quantize + transpose the query vector
        final QuantizedQuery quantizedQuery = quantizeQuery(quantizedByteVectorValues, target);

        // Return Java array backed bulk SIMD scorer
        return new JavaArrayBulkSimdRandomVectorScorer(
            quantizedQuery.targetQuantized,
            quantizedQuery.targetCorrectiveTerms,
            quantizedByteVectorValues,
            similarityFunction,
            target.length,
            quantizedByteVectorValues.getCentroidDP()
        );
    }

    /**
     * Holder for a quantized query vector and its corrective terms.
     */
    private static final class QuantizedQuery {
        private final byte[] targetQuantized;
        private final OptimizedScalarQuantizer.QuantizationResult targetCorrectiveTerms;

        private QuantizedQuery(final byte[] targetQuantized, final OptimizedScalarQuantizer.QuantizationResult targetCorrectiveTerms) {
            this.targetQuantized = targetQuantized;
            this.targetCorrectiveTerms = targetCorrectiveTerms;
        }
    }

    /**
     * Validates the scalar encoding then quantizes and transposes the query vector into the same
     * representation as stored vectors. Shared by the MMap and JavaArray bulk SIMD scorers.
     *
     * @param quantizedByteVectorValues the quantized vector storage
     * @param target                    the query vector (float32)
     * @return the quantized query and its corrective terms
     * @throws IOException if quantization fails
     */
    private static QuantizedQuery quantizeQuery(final QuantizedByteVectorValues quantizedByteVectorValues, final float[] target)
        throws IOException {
        // Check encoding type
        final ScalarEncoding scalarEncoding = quantizedByteVectorValues.getScalarEncoding();

        // We only support 32x quantization with 4 bit query quantization for search.
        if (scalarEncoding != SINGLE_BIT_QUERY_NIBBLE) {
            throw new IllegalStateException(String.format("SQ only supports %s encoding.", SINGLE_BIT_QUERY_NIBBLE));
        }

        // Validate dimensionality
        FlatVectorsScorer.checkDimensions(target.length, quantizedByteVectorValues.dimension());

        // Transpose query vector if it needs to
        final OptimizedScalarQuantizer quantizer = quantizedByteVectorValues.getQuantizer();
        final byte[] scratch = new byte[scalarEncoding.getDiscreteDimensions(quantizedByteVectorValues.dimension())];
        final byte[] targetQuantized;
        if (scalarEncoding.isAsymmetric() == false) {
            targetQuantized = scratch;
        } else {
            // Asymmetric encoding requires packed representation
            targetQuantized = new byte[scalarEncoding.getQueryPackedLength(scratch.length)];
        }

        // We make a copy as the quantization process mutates the input
        final float[] targetCopy = ArrayUtil.copyOfSubArray(target, 0, target.length);

        // For cosine similarity, the query vector is expected to already be normalized.
        // Normalization is performed upfront in KNNQueryBuilder via VectorTransformerFactory
        // for Lucene cosine with SQ 1-bit and flat methods and for Faiss.

        // Perform scalar quantization
        final OptimizedScalarQuantizer.QuantizationResult targetCorrectiveTerms = quantizer.scalarQuantize(
            targetCopy,
            scratch,
            scalarEncoding.getQueryBits(),
            quantizedByteVectorValues.getCentroid()
        );

        // Transpose half-bytes (nibbles) for SIMD-friendly layout
        OptimizedScalarQuantizer.transposeHalfByte(scratch, targetQuantized);

        return new QuantizedQuery(targetQuantized, targetCorrectiveTerms);
    }

    /**
     * A {@link RandomVectorScorer} implementation backed by native SIMD computation.
     *
     * <p>This scorer delegates all similarity computations to a native service
     * ({@link SimdVectorComputeService}), which uses preloaded query state and raw
     * vector memory to compute scores efficiently using SIMD instructions.
     *
     * <p>The query is preprocessed (quantized + transformed) once during construction,
     * and reused across all scoring calls.
     */
    private static class BulkSimdRandomVectorScorer extends RandomVectorScorer.AbstractRandomVectorScorer {
        /**
         * Constructs a SIMD-backed scorer and initializes the native search context.
         *
         * <p>This constructor pushes all necessary query state into native memory,
         * including quantized query values and correction terms required for accurate scoring.
         *
         * @param targetQuantized       quantized query vector
         * @param targetCorrectiveTerms correction terms from quantization
         * @param addressAndSize        raw memory location of vector data
         * @param knnVectorValues       vector storage abstraction
         * @param similarityFunction    similarity function (IP or L2)
         * @param dimension             vector dimensionality
         * @param centroidDp            centroid dot-product correction
         */
        public BulkSimdRandomVectorScorer(
            final byte[] targetQuantized,
            final OptimizedScalarQuantizer.QuantizationResult targetCorrectiveTerms,
            final long[] addressAndSize,
            final QuantizedByteVectorValues knnVectorValues,
            final VectorSimilarityFunction similarityFunction,
            final int dimension,
            final float centroidDp
        ) {
            super(knnVectorValues);

            // Initialize native SIMD search context
            SimdVectorComputeService.saveSQSearchContext(
                targetQuantized,
                targetCorrectiveTerms.lowerInterval(),
                targetCorrectiveTerms.upperInterval(),
                targetCorrectiveTerms.additionalCorrection(),
                targetCorrectiveTerms.quantizedComponentSum(),
                addressAndSize,
                similarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT
                    || similarityFunction == VectorSimilarityFunction.COSINE
                        ? SimdVectorComputeService.SimilarityFunctionType.SQ_IP.ordinal()
                        : SimdVectorComputeService.SimilarityFunctionType.SQ_L2.ordinal(),
                dimension,
                centroidDp
            );
        }

        /**
         * Computes similarity scores for multiple vectors in bulk using native SIMD code.
         *
         * <p>This method is optimized for throughput and should be preferred when scoring
         * large batches of vectors.
         *
         * @param internalVectorIds vector ordinals to score
         * @param scores            output buffer for similarity scores
         * @param numVectors        number of vectors to process
         * @return implementation-defined value (typically unused aggregate)
         */
        @Override
        public float bulkScore(final int[] internalVectorIds, final float[] scores, final int numVectors) {
            return SimdVectorComputeService.scoreSimilarityInBulk(internalVectorIds, scores, numVectors);
        }

        /**
         * Computes the similarity score for a single vector using native SIMD code.
         *
         * @param internalVectorId the internal vector ID to score
         * @return the computed similarity score
         * @throws IOException if the native scoring operation fails
         */
        @Override
        public float score(final int internalVectorId) {
            return SimdVectorComputeService.scoreSimilarity(internalVectorId);
        }
    }

    /**
     * A {@link RandomVectorScorer} implementation backed by native SIMD computation over vectors
     * copied into a Java heap buffer. Used for the 'JavaArray' POC experiment.
     *
     * <p>Unlike {@link BulkSimdRandomVectorScorer}, which hands the native side a raw MMap pointer,
     * this scorer copies each requested vector's element bytes
     * ([quantized vector | lowerInterval | upperInterval | additionalCorrection | quantizedComponentSum])
     * from the underlying {@link IndexInput} slice into a contiguous heap scratch buffer, then passes
     * that buffer to JNI. Since batch vector ids arriving from HNSW traversal are scattered, elements
     * are copied one by one into batch slots; the native side then addresses them with implicit ids
     * 0..numVectors-1 (the slot indices).
     *
     * <p>The on-disk element layout already matches what the native kernel expects, so the copy is a
     * raw byte copy with no repacking. The scratch buffer starts at 32 elements and grows by doubling.
     * It is only borrowed by native code for the duration of a single JNI call, so there is no
     * cross-call pinning.
     */
    private static class JavaArrayBulkSimdRandomVectorScorer extends RandomVectorScorer.AbstractRandomVectorScorer {
        /** Default number of elements the scratch buffer can hold. HNSW batch sizes are typically <= 32. */
        private static final int DEFAULT_SCRATCH_NUM_VECTORS = 32;

        /** Byte size of the correction factors trailing each quantized vector: 3 floats + 1 int. */
        private static final int CORRECTION_FACTORS_BYTES = 3 * Float.BYTES + Integer.BYTES;

        /** Read source for element bytes. Cloned so this scorer owns its own seek position. */
        private final IndexInput slice;

        /** Per-element byte size: (dim + 7) / 8 quantized vector bytes + correction factors. */
        private final int oneVecSize;

        /** Batch buffer holding copied elements, grown by doubling. */
        private byte[] scratch;

        /**
         * Constructs a heap-copy backed SIMD scorer and initializes the native search context.
         *
         * @param targetQuantized       quantized query vector
         * @param targetCorrectiveTerms correction terms from quantization
         * @param knnVectorValues       vector storage abstraction
         * @param similarityFunction    similarity function (IP or L2)
         * @param dimension             vector dimensionality
         * @param centroidDp            centroid dot-product correction
         */
        public JavaArrayBulkSimdRandomVectorScorer(
            final byte[] targetQuantized,
            final OptimizedScalarQuantizer.QuantizationResult targetCorrectiveTerms,
            final QuantizedByteVectorValues knnVectorValues,
            final VectorSimilarityFunction similarityFunction,
            final int dimension,
            final float centroidDp
        ) {
            super(knnVectorValues);

            // Clone to own an independent seek position on the slice.
            this.slice = knnVectorValues.getSlice().clone();
            this.oneVecSize = (dimension + 7) / 8 + CORRECTION_FACTORS_BYTES;
            this.scratch = new byte[DEFAULT_SCRATCH_NUM_VECTORS * oneVecSize];

            // Initialize native SIMD search context. Unlike the MMap path, no addressAndSize is passed;
            // the vector region is supplied per scoring call from the scratch buffer.
            SimdVectorComputeService.saveSQSearchContext2(
                targetQuantized,
                targetCorrectiveTerms.lowerInterval(),
                targetCorrectiveTerms.upperInterval(),
                targetCorrectiveTerms.additionalCorrection(),
                targetCorrectiveTerms.quantizedComponentSum(),
                similarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT
                    || similarityFunction == VectorSimilarityFunction.COSINE
                        ? SimdVectorComputeService.SimilarityFunctionType.SQ_IP.ordinal()
                        : SimdVectorComputeService.SimilarityFunctionType.SQ_L2.ordinal(),
                dimension,
                centroidDp
            );
        }

        /**
         * Grows the scratch buffer by doubling until it can hold `numVectors` elements.
         *
         * @param numVectors the number of elements the buffer must hold
         */
        private void ensureCapacity(final int numVectors) {
            final int needed = numVectors * oneVecSize;
            if (scratch.length < needed) {
                int newLength = scratch.length;
                while (newLength < needed) {
                    newLength <<= 1;
                }
                scratch = new byte[newLength];
            }
        }

        /**
         * Computes similarity scores for multiple vectors in bulk using native SIMD code.
         *
         * <p>Copies each vector's element bytes into consecutive scratch buffer slots, then scores the
         * whole batch in one JNI call. Ids passed down to native are implicitly 0..numVectors-1.
         *
         * @param internalVectorIds vector ordinals to score
         * @param scores            output buffer for similarity scores
         * @param numVectors        number of vectors to process
         * @return implementation-defined value (typically unused aggregate)
         */
        @Override
        public float bulkScore(final int[] internalVectorIds, final float[] scores, final int numVectors) throws IOException {
            ensureCapacity(numVectors);
            for (int k = 0; k < numVectors; k++) {
                slice.seek((long) internalVectorIds[k] * oneVecSize);
                slice.readBytes(scratch, k * oneVecSize, oneVecSize);
            }
            return SimdVectorComputeService.scoreSimilarityInBulk2(scratch, scores, numVectors);
        }

        /**
         * Computes the similarity score for a single vector using native SIMD code.
         *
         * @param internalVectorId the internal vector ID to score
         * @return the computed similarity score
         * @throws IOException if reading the vector bytes fails
         */
        @Override
        public float score(final int internalVectorId) throws IOException {
            slice.seek((long) internalVectorId * oneVecSize);
            slice.readBytes(scratch, 0, oneVecSize);
            return SimdVectorComputeService.scoreSimilarity2(scratch);
        }
    }
}
