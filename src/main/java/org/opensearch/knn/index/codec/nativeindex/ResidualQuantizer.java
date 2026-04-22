/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.lucene104.QuantizedByteVectorValues;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;
import java.util.function.Supplier;

import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

/**
 * Utility class for computing and writing 4-bit quantized error residuals to a {@code .ver} file.
 *
 * <h2>Background</h2>
 * With 1-bit scalar quantization (SQ), the approximated inner product is:
 * <pre>
 *   &lt;q, x&gt; ≈ &lt;q', Q(x')&gt; + &lt;q, c&gt; + &lt;c, x&gt; - &lt;c, c&gt;
 * </pre>
 * where {@code q' = q - c}, {@code x' = x - c}, and {@code Q(x')} is the 1-bit quantized centered vector.
 * The residual {@code r = x' - Q(x')} captures the quantization error. By quantizing {@code r} to 4 bits
 * and storing it in a separate file, the 2nd-phase rescoring can compute {@code <q', Q_4(r)>} as a
 * correction term — using 8x less IO than loading full-precision vectors.
 *
 * <h2>Single-pass approach with per-vector bounds</h2>
 * For 1-bit SQ with per-vector interval {@code [lower, upper]}, the quantization step is
 * {@code delta = upper - lower}. Each residual component is bounded by {@code [-delta/2, delta/2]}
 * because the 1-bit quantizer snaps each component to either {@code lower} or {@code upper},
 * and the maximum error from that snap is {@code delta/2}.
 *
 * This means the 4-bit quantization range is derived analytically per-vector — no global scanning
 * pass is needed. The build iterates all vectors exactly once.
 *
 * <h2>File format ({@code .ver})</h2>
 * <pre>
 * [Header — 20 bytes]
 * Offset  Size    Field
 * ──────  ──────  ──────────────────────────
 * 0       4B      magic (0x56455231 = "VER1")
 * 4       4B      dimension
 * 8       4B      numVectors
 * 12      1B      bitsPerDimension (4)
 * 13      4B      bytesPerBlock (packed residual + 16B metadata)
 * 17      3B      reserved (padding)
 *
 * [Per-vector block — repeated numVectors times]
 * Offset  Size              Field
 * ──────  ──────            ──────────────────────────
 * 0       (dim+1)/2 bytes   packed 4-bit residual (low nibble first)
 * P       4B                lowerInterval = -delta/2 (float, LE)
 * P+4     4B                upperInterval = delta/2 (float, LE)
 * P+8     4B                additionalCorrection = 0.0 (reserved)
 * P+12    4B                componentSum (int, LE)
 * </pre>
 * The per-vector metadata layout mirrors the {@code .veb} format (packed quantized data + correction
 * factors), enabling potential reuse of the ADC scoring infrastructure at the SIMD P1 stage.
 *
 * A CodecUtil footer (16 bytes) follows the data; the caller is responsible for writing it.
 *
 * @see MemOptimizedScalarQuantizedIndexBuildStrategy — invokes this class as Phase 4 of index build
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class ResidualQuantizer {

    /** Codec name for the .ver file header, used by {@link CodecUtil#writeIndexHeader}. */
    public static final String CODEC_NAME = "ErrorResidualVectors";

    /** Codec version for the .ver file format. */
    public static final int VERSION_CURRENT = 0;

    /** Number of bits used per dimension for residual quantization. */
    public static final int BITS_PER_DIMENSION = 4;

    /** Per-vector metadata size: lowerInterval(4) + upperInterval(4) + additionalCorrection(4) + componentSum(4). */
    public static final int PER_VECTOR_META_BYTES = 16;

    /**
     * Unpack a single bit from a bit-packed binary code.
     *
     * The 1-bit SQ binary codes are packed MSB-first by Lucene's
     * {@code OptimizedScalarQuantizer.packAsBinary()}: dimension 0 is stored in bit 7 (MSB)
     * of byte 0, dimension 7 in bit 0 (LSB) of byte 0, dimension 8 in bit 7 of byte 1, etc.
     *
     * @param binaryCode     the bit-packed 1-bit quantized code from {@code .veb}
     * @param dimensionIndex which dimension to unpack (0-based)
     * @return 0 or 1
     */
    public static int unpackBit(byte[] binaryCode, int dimensionIndex) {
        return (binaryCode[dimensionIndex / 8] >>> (7 - (dimensionIndex % 8))) & 1;
    }

    /**
     * Compute the residual for a single vector component.
     *
     * The residual captures the quantization error for dimension {@code d}:
     * <pre>
     *   Q(x')_d = lowerInterval + bit_d * delta              (dequantized 1-bit value)
     *   r_d     = centeredVecComponent - Q(x')_d              (residual)
     * </pre>
     * For 1-bit SQ, {@code nSteps = 2^1 - 1 = 1}, so {@code delta = upperInterval - lowerInterval}.
     *
     * <p>Note: the input vector is expected to be already centered (centroid subtracted).
     * Lucene's {@code OptimizedScalarQuantizer.scalarQuantize()} mutates the vector in-place
     * by subtracting the centroid, so vectors read back from the supplier are already centered.
     *
     * @param centeredVecComponent the centered value x'_d (already has centroid subtracted)
     * @param binaryCode           the bit-packed 1-bit quantized code (MSB-first)
     * @param dimensionIndex       which dimension (d) — used for bit unpacking
     * @param lowerInterval        per-vector quantization lower bound
     * @param delta                upperInterval - lowerInterval
     * @return the residual r_d = x'_d - Q(x')_d
     */
    public static float computeResidual(
        float centeredVecComponent,
        byte[] binaryCode,
        int dimensionIndex,
        float lowerInterval,
        float delta
    ) {
        int bit = unpackBit(binaryCode, dimensionIndex);
        float qVal = lowerInterval + bit * delta;
        return centeredVecComponent - qVal;
    }

    /**
     * Pack two 4-bit values into one byte.
     *
     * Layout: low nibble holds the first dimension, high nibble holds the second.
     * <pre>
     *   byte = (high &lt;&lt; 4) | (low &amp; 0x0F)
     * </pre>
     * This matches the unpacking convention used at search time:
     * <pre>
     *   low  = byte &amp; 0x0F         (even dimension)
     *   high = (byte &gt;&gt;&gt; 4) &amp; 0x0F (odd dimension)
     * </pre>
     *
     * @param low  4-bit value for the even dimension [0, 15]
     * @param high 4-bit value for the odd dimension [0, 15]
     * @return packed byte
     */
    public static byte packNibbles(int low, int high) {
        return (byte) ((high << 4) | (low & 0x0F));
    }

    /**
     * Quantize a residual vector uniformly into {@code [0, 2^bits - 1]} using per-vector delta.
     *
     * Maps each residual component from {@code [-delta/2, delta/2]} to {@code [0, nSteps]}
     * where {@code nSteps = 2^BITS_PER_DIMENSION - 1 = 15}:
     * <pre>
     *   normalized = (r_d + delta/2) / delta     →  [0.0, 1.0]
     *   q_d = clamp(round(normalized * nSteps), 0, nSteps)
     * </pre>
     *
     * When {@code delta == 0} (all components quantized to the same level), all outputs are
     * set to the midpoint (nSteps/2) to avoid division by zero.
     *
     * @param residual  per-dimension residual values (length = dimension)
     * @param scratch   output buffer for quantized values, one byte per dimension (length >= dimension)
     * @param dimension vector dimensionality
     * @param delta     per-vector quantization step (upperInterval - lowerInterval)
     * @return componentSum — sum of all quantized values (used in ADC scoring formula)
     */
    public static int quantizeResidualUniform(float[] residual, byte[] scratch, int dimension, float delta) {
        final int nSteps = (1 << BITS_PER_DIMENSION) - 1; // 15
        int componentSum = 0;

        if (delta == 0.0f) {
            // All residuals are zero — assign midpoint to avoid division by zero
            int midpoint = nSteps / 2;
            for (int d = 0; d < dimension; d++) {
                scratch[d] = (byte) midpoint;
                componentSum += midpoint;
            }
            return componentSum;
        }

        for (int d = 0; d < dimension; d++) {
            // Map [-delta/2, delta/2] → [0, 1] → [0, nSteps]
            float normalized = (residual[d] + delta / 2.0f) / delta;
            int q = Math.max(0, Math.min(nSteps, Math.round(normalized * nSteps)));
            scratch[d] = (byte) q;
            componentSum += q;
        }
        return componentSum;
    }

    /**
     * Single-pass: compute residuals, quantize to 4-bit, and write the {@code .ver} file
     * with buffered output.
     *
     * <h3>Algorithm</h3>
     * For each vector:
     * <ol>
     *   <li>Read full-precision vector and 1-bit binary code with correction factors</li>
     *   <li>Compute per-dimension residual: {@code r_d = (x_d - c_d) - (lower + bit_d * delta)}</li>
     *   <li>Derive per-vector quantization bounds: {@code [-delta/2, delta/2]}</li>
     *   <li>Quantize residual to 4-bit using {@link #quantizeResidualUniform}</li>
     *   <li>Pack nibbles and per-vector metadata into the batch buffer</li>
     *   <li>Flush buffer when full (~64KB) or on the last vector</li>
     * </ol>
     *
     * <h3>Buffering strategy</h3>
     * Follows the same pattern as {@code passQuantizedVectorsAndCorrectionFactors}: accumulates
     * complete vector blocks (packed residual + 16B metadata) in a ~64KB byte buffer and
     * flushes in bulk to the {@link IndexOutput}. Batch size is
     * {@code max(1, 65536 / oneBlockSize)} vectors per flush.
     *
     * <h3>Scratch arrays</h3>
     * Two scratch arrays ({@code float[] residual} and {@code byte[] residualScratch}) are
     * allocated once and reused across all vectors to avoid per-vector heap allocation.
     *
     * @param output          Lucene IndexOutput to write the .ver file to
     * @param state           segment write state (for codec header with segment ID and suffix)
     * @param vectorSupplier  supplies a fresh iterator over full-precision float vectors
     * @param quantizedValues provides 1-bit binary codes and per-vector correction factors
     * @param centroid        centroid vector (mean of all vectors)
     * @param dimension       vector dimensionality
     * @param numVectors      total number of vectors in this segment
     */
    public static void writeResidualFile(
        IndexOutput output,
        SegmentWriteState state,
        Supplier<KNNVectorValues<?>> vectorSupplier,
        QuantizedByteVectorValues quantizedValues,
        float[] centroid,
        int dimension,
        int numVectors
    ) throws IOException {
        // Packed residual bytes: 4-bit packing = 2 dimensions per byte, rounded up for odd dims
        final int packedResidualBytes = (dimension + 1) / 2;
        // Each vector block: [packed residual] + [lower(4B) + upper(4B) + correction(4B) + sum(4B)]
        final int oneBlockSize = packedResidualBytes + PER_VECTOR_META_BYTES;

        writeHeader(output, state, dimension, numVectors, oneBlockSize);

        final KNNVectorValues<?> vectors = vectorSupplier.get();
        initializeVectorValues(vectors);

        // Size the batch buffer to hold ~64KB worth of complete vector blocks.
        // Math.max(1, ...) ensures at least one vector per batch even for very high dimensions.
        final int batchSize = Math.max(1, (64 * 1024) / oneBlockSize);
        final byte[] buffer = new byte[batchSize * oneBlockSize];

        // Scratch arrays reused per vector to avoid allocation in the hot loop
        final float[] residual = new float[dimension];
        final byte[] residualScratch = new byte[dimension];

        for (int ord = 0; ord < numVectors; ord++) {
            final int batchIdx = ord % batchSize;
            final int bufOffset = batchIdx * oneBlockSize;

            final float[] fullVec = (float[]) vectors.getVector();
            final byte[] binaryCode = quantizedValues.vectorValue(ord);
            final OptimizedScalarQuantizer.QuantizationResult terms = quantizedValues.getCorrectiveTerms(ord);
            final float lower = terms.lowerInterval();
            final float upper = terms.upperInterval();
            final float delta = upper - lower;

            // Compute per-dimension residual: r_d = (x_d - c_d) - Q(x')_d
            for (int d = 0; d < dimension; d++) {
                // fullVec[d] is already centered (centroid subtracted by scalarQuantize in-place)
                residual[d] = computeResidual(fullVec[d], binaryCode, d, lower, delta);
            }

            // Quantize residual to 4-bit with per-vector bounds [-delta/2, delta/2]
            final int componentSum = quantizeResidualUniform(residual, residualScratch, dimension, delta);

            // Pack 4-bit nibbles: two dimensions per byte, low nibble = even dim, high nibble = odd dim
            for (int d = 0; d < dimension; d += 2) {
                int q0 = residualScratch[d] & 0x0F;
                int q1 = (d + 1 < dimension) ? (residualScratch[d + 1] & 0x0F) : 0;
                buffer[bufOffset + d / 2] = packNibbles(q0, q1);
            }

            // Write per-vector metadata after the packed residual (little-endian)
            final int metaOffset = bufOffset + packedResidualBytes;
            final float halfDelta = delta / 2.0f;
            writeFloatLE(buffer, metaOffset, -halfDelta);            // lowerInterval
            writeFloatLE(buffer, metaOffset + 4, halfDelta);         // upperInterval
            writeFloatLE(buffer, metaOffset + 8, 0.0f);             // additionalCorrection (reserved)
            writeIntLE(buffer, metaOffset + 12, componentSum);       // componentSum

            // Flush the batch buffer when full, or on the last vector
            if (batchIdx == batchSize - 1 || ord == numVectors - 1) {
                final int count = batchIdx + 1;
                output.writeBytes(buffer, 0, count * oneBlockSize);
            }

            vectors.nextDoc();
        }
    }

    /**
     * Write the .ver file header: a standard Lucene codec header followed by format-specific fields.
     *
     * The codec header is required for Lucene's compound file format (.cfs) to recognize
     * the file. After the codec header, we write our format-specific fields using IndexOutput's
     * native big-endian methods (consistent with Lucene conventions for on-disk metadata).
     *
     * @param output        the IndexOutput to write to
     * @param state         segment write state (for segment ID and suffix)
     * @param dimension     vector dimensionality
     * @param numVectors    total number of vectors
     * @param bytesPerBlock total bytes per vector block (packed residual + metadata)
     */
    static void writeHeader(IndexOutput output, SegmentWriteState state, int dimension, int numVectors, int bytesPerBlock)
        throws IOException {
        CodecUtil.writeIndexHeader(output, CODEC_NAME, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);
        output.writeInt(dimension);
        output.writeInt(numVectors);
        output.writeInt(bytesPerBlock);
    }

    /**
     * Write a float value into a byte array at the given offset in little-endian order.
     */
    static void writeFloatLE(byte[] buffer, int offset, float value) {
        int bits = Float.floatToRawIntBits(value);
        buffer[offset] = (byte) (bits);
        buffer[offset + 1] = (byte) (bits >>> 8);
        buffer[offset + 2] = (byte) (bits >>> 16);
        buffer[offset + 3] = (byte) (bits >>> 24);
    }

    /**
     * Write an int value into a byte array at the given offset in little-endian order.
     */
    static void writeIntLE(byte[] buffer, int offset, int value) {
        buffer[offset] = (byte) (value);
        buffer[offset + 1] = (byte) (value >>> 8);
        buffer[offset + 2] = (byte) (value >>> 16);
        buffer[offset + 3] = (byte) (value >>> 24);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Search-time score refinement methods
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Precompute {@code q' = queryVector - centroid}. Called once per query per segment.
     *
     * <p>This avoids recomputing the subtraction for every candidate vector during refinement.
     *
     * @param queryVector the original query vector
     * @param centroid    the centroid vector (mean of all vectors in the field)
     * @return float array of length {@code queryVector.length} with {@code q'_d = query_d - centroid_d}
     */
    public static float[] computeQPrime(float[] queryVector, float[] centroid) {
        float[] qPrime = new float[queryVector.length];
        for (int d = 0; d < queryVector.length; d++) {
            qPrime[d] = queryVector[d] - centroid[d];
        }
        return qPrime;
    }

    /**
     * Compute the corrected score for a single candidate using per-vector quantization bounds.
     *
     * <p>Dequantizes each 4-bit nibble from the packed residual using the per-vector
     * {@code [lower, upper]} interval, computes the dot product {@code <q', Q_4(r)>},
     * and adds it to the phase-1 score:
     * <pre>
     *   residualStep = (upper - lower) / 15
     *   for each dimension d:
     *     r_d = lower + nibble_d * residualStep
     *     correction += q'_d * r_d
     *   rawDot = unscaleMaxInnerProductScore(phase1Score)
     *   correctedScore = scaleMaxInnerProductScore(rawDot + correction)
     * </pre>
     *
     * <p>The phase-1 score is MIP-scaled (non-linear). We must unscale it to a raw dot product
     * before adding the correction, then re-scale the result. Adding the correction directly
     * to the scaled score would give wrong results because MIP scaling is non-linear:
     * {@code dp >= 0 → score = dp + 1}, {@code dp < 0 → score = 1/(1-dp)}.
     *
     * @param qPrime         precomputed {@code q' = query - centroid} (length = dimension)
     * @param packedResidual 4-bit packed residual bytes from the .ver block (low nibble first)
     * @param lower          per-vector lowerInterval ({@code -delta/2})
     * @param upper          per-vector upperInterval ({@code delta/2})
     * @param phase1Score    MIP-scaled score from 1st-phase approximate search
     * @param dimension      vector dimensionality
     * @return corrected MIP-scaled score
     */
    public static float computeCorrectedScore(
        float[] qPrime,
        byte[] packedResidual,
        float lower,
        float upper,
        float phase1Score,
        int dimension
    ) {
        float residualStep = (upper - lower) / 15.0f;
        float correction = 0.0f;

        for (int d = 0; d < dimension; d++) {
            // Unpack 4-bit nibble: even dim → low nibble, odd dim → high nibble
            int nibble = (d % 2 == 0)
                ? (packedResidual[d / 2] & 0x0F)
                : ((packedResidual[d / 2] >>> 4) & 0x0F);

            // Dequantize and accumulate dot product with q'
            float r_d = lower + nibble * residualStep;
            correction += qPrime[d] * r_d;
        }

        // Unscale MIP score → raw dot product, add correction, re-scale
        float rawDot = unscaleMaxInnerProductScore(phase1Score);
        return scaleMaxInnerProductScore(rawDot + correction);
    }

    /**
     * Reverse of {@code VectorUtil.scaleMaxInnerProductScore}.
     * Recovers the raw dot product from a MIP-scaled score.
     *
     * @param score the MIP-scaled score (always > 0)
     * @return the raw dot product value
     */
    static float unscaleMaxInnerProductScore(float score) {
        if (score >= 1.0f) {
            return score - 1.0f;
        }
        return -(1.0f / score - 1.0f);
    }

    /**
     * Scales a raw dot product to a positive MIP score, matching
     * {@code VectorUtil.scaleMaxInnerProductScore}.
     *
     * @param dotProduct the raw dot product
     * @return the MIP-scaled score (always > 0)
     */
    static float scaleMaxInnerProductScore(float dotProduct) {
        if (dotProduct >= 0) {
            return dotProduct + 1.0f;
        }
        return 1.0f / (1.0f - dotProduct);
    }
}
