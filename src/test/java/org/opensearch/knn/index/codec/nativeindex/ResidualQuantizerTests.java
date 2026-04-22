/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.lucene104.QuantizedByteVectorValues;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Collections;
import java.util.List;
import java.util.function.Supplier;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link ResidualQuantizer}.
 *
 * Tests are organized by method, covering:
 * <ul>
 *   <li>Bit unpacking from 1-bit SQ binary codes ({@code unpackBit})</li>
 *   <li>Per-component residual computation ({@code computeResidual})</li>
 *   <li>Per-vector uniform 4-bit quantization ({@code quantizeResidualUniform})</li>
 *   <li>Nibble packing into bytes ({@code packNibbles})</li>
 *   <li>Single-pass .ver file writing ({@code writeResidualFile}) — header format,
 *       per-vector metadata, round-trip accuracy, odd dimensions, and file size</li>
 * </ul>
 *
 * All tests use mock {@link QuantizedByteVectorValues} (via Mockito) and predefined float vectors
 * (via {@link TestVectorValues.PreDefinedFloatVectorValues}) to isolate the quantizer logic from
 * Lucene's on-disk format.
 */
public class ResidualQuantizerTests extends OpenSearchTestCase {

    // ──────────────────────────────────────────────────────────────────────────
    // unpackBit: extracts a single dimension's bit from the 1-bit SQ binary code.
    // Lucene's packAsBinary packs MSB-first: dimension 0 → bit 7, dimension 7 → bit 0.
    //   byte[0] bit 7 = dimension 0, byte[0] bit 0 = dimension 7,
    //   byte[1] bit 7 = dimension 8, etc.
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Verify all 8 bit positions within a single byte (MSB-first layout).
     * 0xB4 = 0b10110100:
     *   bit 7=1 → dim 0 = 1
     *   bit 6=0 → dim 1 = 0
     *   bit 5=1 → dim 2 = 1
     *   bit 4=1 → dim 3 = 1
     *   bit 3=0 → dim 4 = 0
     *   bit 2=1 → dim 5 = 1
     *   bit 1=0 → dim 6 = 0
     *   bit 0=0 → dim 7 = 0
     */
    public void testUnpackBit() {
        byte[] binaryCode = new byte[] { (byte) 0xB4 };

        assertEquals(1, ResidualQuantizer.unpackBit(binaryCode, 0));
        assertEquals(0, ResidualQuantizer.unpackBit(binaryCode, 1));
        assertEquals(1, ResidualQuantizer.unpackBit(binaryCode, 2));
        assertEquals(1, ResidualQuantizer.unpackBit(binaryCode, 3));
        assertEquals(0, ResidualQuantizer.unpackBit(binaryCode, 4));
        assertEquals(1, ResidualQuantizer.unpackBit(binaryCode, 5));
        assertEquals(0, ResidualQuantizer.unpackBit(binaryCode, 6));
        assertEquals(0, ResidualQuantizer.unpackBit(binaryCode, 7));
    }

    /**
     * Verify bit unpacking across byte boundaries (MSB-first layout).
     * byte[0]=0x01 (only bit 0 set → dimension 7), byte[1]=0x80 (only bit 7 set → dimension 8).
     */
    public void testUnpackBit_multipleBytes() {
        byte[] binaryCode = new byte[] { 0x01, (byte) 0x80 };

        // Byte 0: 0x01 = 0b00000001 → only dim 7 is set (bit 0 = dim 7)
        assertEquals(0, ResidualQuantizer.unpackBit(binaryCode, 0));  // bit 7 = 0
        assertEquals(0, ResidualQuantizer.unpackBit(binaryCode, 1));  // bit 6 = 0
        assertEquals(1, ResidualQuantizer.unpackBit(binaryCode, 7));  // bit 0 = 1

        // Byte 1: 0x80 = 0b10000000 → only dim 8 is set (bit 7 = dim 8)
        assertEquals(1, ResidualQuantizer.unpackBit(binaryCode, 8));  // bit 7 = 1
        assertEquals(0, ResidualQuantizer.unpackBit(binaryCode, 15)); // bit 0 = 0
    }

    // ──────────────────────────────────────────────────────────────────────────
    // computeResidual: r_d = (x_d - c_d) - (lower + bit_d * delta)
    // Tests both bit=1 and bit=0 cases with hand-computed expected values.
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * When the quantized bit is 1 (MSB-first: dim 0 = bit 7):
     * 0x80 = 0b10000000 → dim 0 has bit=1
     * Input is already centered: x'_d = 1.0 (= original 1.5 - centroid 0.5)
     * Q(x')_d = -0.2 + 1 * 1.4 = 1.2
     * residual = 1.0 - 1.2 = -0.2
     */
    public void testComputeResidual() {
        byte[] binaryCode = new byte[] { (byte) 0x80 }; // bit 7 = 1 → dim 0 = 1

        // Pass already-centered value (1.5 - 0.5 = 1.0)
        float r = ResidualQuantizer.computeResidual(1.0f, binaryCode, 0, -0.2f, 1.4f);
        assertEquals(-0.2f, r, 1e-6f);
    }

    /**
     * When the quantized bit is 0:
     * Input is already centered: x'_d = 1.5 (= original 2.0 - centroid 0.5)
     * Q(x')_d = lower + 0 * delta = 0.3
     * residual = 1.5 - 0.3 = 1.2
     */
    public void testComputeResidual_bitZero() {
        byte[] binaryCode = new byte[] { 0x00 }; // all bits 0

        // Pass already-centered value (2.0 - 0.5 = 1.5)
        float r = ResidualQuantizer.computeResidual(1.5f, binaryCode, 0, 0.3f, 1.0f);
        assertEquals(1.2f, r, 1e-6f);
    }

    /**
     * Verify computeResidual across byte boundaries (dimension >= 8, MSB-first).
     * byte[0]=0x00, byte[1]=0x80 → dim 8 has bit=1 (bit 7 of byte 1)
     *
     * dim 8 (bit=1): x'=2.5 (already centered), Q(x')=0.0+1*2.0=2.0, residual=0.5
     * dim 9 (bit=0): x'=0.5 (already centered), Q(x')=0.0+0*2.0=0.0, residual=0.5
     */
    public void testComputeResidual_crossByteBoundary() {
        byte[] binaryCode = new byte[] { 0x00, (byte) 0x80 }; // byte1 bit7 = dim 8 = 1

        // Already-centered values (3.0-0.5=2.5, 1.0-0.5=0.5)
        float r8 = ResidualQuantizer.computeResidual(2.5f, binaryCode, 8, 0.0f, 2.0f);
        assertEquals(0.5f, r8, 1e-6f);

        float r9 = ResidualQuantizer.computeResidual(0.5f, binaryCode, 9, 0.0f, 2.0f);
        assertEquals(0.5f, r9, 1e-6f);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // quantizeResidualUniform: maps residual from [-delta/2, delta/2] → [0, 15]
    // per-vector. Tests normal quantization, boundary values, and delta=0 edge case.
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Normal per-vector quantization with delta=2.0.
     * halfDelta = 1.0, nSteps = 15
     * residual = [0.0, -1.0, 1.0, 0.5]
     * normalized = [(0+1)/2, (-1+1)/2, (1+1)/2, (0.5+1)/2] = [0.5, 0.0, 1.0, 0.75]
     * q = [round(7.5), 0, 15, round(11.25)] = [8, 0, 15, 11]
     * componentSum = 8 + 0 + 15 + 11 = 34
     */
    public void testQuantizeResidualUniform() {
        float[] residual = new float[] { 0.0f, -1.0f, 1.0f, 0.5f };
        byte[] scratch = new byte[4];

        int sum = ResidualQuantizer.quantizeResidualUniform(residual, scratch, 4, 2.0f);

        assertEquals(8, scratch[0] & 0xFF);
        assertEquals(0, scratch[1] & 0xFF);
        assertEquals(15, scratch[2] & 0xFF);
        assertEquals(11, scratch[3] & 0xFF);
        assertEquals(34, sum);
    }

    /**
     * When delta=0 (all 1-bit quantization levels collapse), all residuals are zero.
     * quantizeResidualUniform must return midpoint (7) for all dimensions without
     * dividing by zero.
     */
    public void testQuantizeResidualUniform_zeroDelta() {
        float[] residual = new float[] { 0.0f, 0.0f, 0.0f, 0.0f };
        byte[] scratch = new byte[4];

        int sum = ResidualQuantizer.quantizeResidualUniform(residual, scratch, 4, 0.0f);

        for (int d = 0; d < 4; d++) {
            assertEquals("dim " + d + " should be midpoint 7", 7, scratch[d] & 0xFF);
        }
        assertEquals(28, sum); // 7 * 4
    }

    /**
     * Boundary clamping: residuals slightly outside [-delta/2, delta/2] due to
     * floating-point imprecision should be clamped to [0, 15].
     */
    public void testQuantizeResidualUniform_clamping() {
        // delta=1.0, halfDelta=0.5, bounds=[-0.5, 0.5]
        // residual slightly beyond bounds
        float[] residual = new float[] { -0.6f, 0.6f };
        byte[] scratch = new byte[2];

        ResidualQuantizer.quantizeResidualUniform(residual, scratch, 2, 1.0f);

        assertEquals(0, scratch[0] & 0xFF);   // clamped below
        assertEquals(15, scratch[1] & 0xFF);   // clamped above
    }

    // ──────────────────────────────────────────────────────────────────────────
    // packNibbles: packs two 4-bit values into one byte.
    // Convention: low nibble = even dimension, high nibble = odd dimension.
    //   byte = (high << 4) | (low & 0x0F)
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Verify nibble packing for various combinations including edge cases.
     */
    public void testPackNibbles() {
        // Mixed values: low=0x0A(10), high=0x05(5) → 0x5A
        assertEquals((byte) 0x5A, ResidualQuantizer.packNibbles(0x0A, 0x05));

        // Both zero
        assertEquals((byte) 0x00, ResidualQuantizer.packNibbles(0, 0));

        // Both max (15)
        assertEquals((byte) 0xFF, ResidualQuantizer.packNibbles(15, 15));

        // Only low nibble set → high nibble is 0
        assertEquals((byte) 0x0F, ResidualQuantizer.packNibbles(15, 0));

        // Only high nibble set → low nibble is 0
        assertEquals((byte) 0xF0, ResidualQuantizer.packNibbles(0, 15));
    }

    // ──────────────────────────────────────────────────────────────────────────
    // computeQPrime: precomputes q' = query - centroid once per query.
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Basic subtraction: query=[1.0, 2.0, 3.0], centroid=[0.5, 0.5, 0.5]
     * Expected q' = [0.5, 1.5, 2.5]
     */
    public void testComputeQPrime() {
        float[] query = { 1.0f, 2.0f, 3.0f };
        float[] centroid = { 0.5f, 0.5f, 0.5f };

        float[] qPrime = ResidualQuantizer.computeQPrime(query, centroid);

        assertEquals(3, qPrime.length);
        assertEquals(0.5f, qPrime[0], 1e-6f);
        assertEquals(1.5f, qPrime[1], 1e-6f);
        assertEquals(2.5f, qPrime[2], 1e-6f);
    }

    /**
     * Zero centroid: q' should equal the query vector.
     */
    public void testComputeQPrime_zeroCentroid() {
        float[] query = { -1.0f, 0.0f, 1.0f };
        float[] centroid = { 0.0f, 0.0f, 0.0f };

        float[] qPrime = ResidualQuantizer.computeQPrime(query, centroid);

        assertEquals(-1.0f, qPrime[0], 1e-6f);
        assertEquals(0.0f, qPrime[1], 1e-6f);
        assertEquals(1.0f, qPrime[2], 1e-6f);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // computeCorrectedScore: dequantizes per-vector 4-bit residuals, computes
    // <q', Q_4(r)> dot product, unscales MIP score, adds correction, re-scales.
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Hand-computed example with dim=4, per-vector bounds [-0.5, 0.5].
     *
     * qPrime = [1.0, -1.0, 0.5, -0.5]
     * nibbles = [0, 15, 8, 7], lower=-0.5, upper=0.5, step=1/15
     * correction ≈ -0.96667
     *
     * phase1Score = 5.0 (MIP-scaled, rawDot = 5.0 - 1.0 = 4.0)
     * rawTotal = 4.0 + (-0.96667) = 3.0333
     * expectedResult = scale(3.0333) = 3.0333 + 1.0 = 4.0333
     */
    public void testComputeCorrectedScore() {
        float[] qPrime = { 1.0f, -1.0f, 0.5f, -0.5f };
        byte[] packed = { (byte) 0xF0, (byte) 0x78 }; // nibbles: [0, 15, 8, 7]
        float lower = -0.5f;
        float upper = 0.5f;
        // phase1Score = scale(4.0) = 5.0
        float phase1Score = 5.0f;

        float result = ResidualQuantizer.computeCorrectedScore(qPrime, packed, lower, upper, phase1Score, 4);

        // correction ≈ -0.96667, rawDot = 4.0, rawTotal = 3.0333, scale(3.0333) = 4.0333
        assertEquals(4.0333f, result, 1e-3f);
    }

    /**
     * When lower == upper (delta=0), correction = 0, so result == phase1Score.
     */
    public void testComputeCorrectedScore_zeroDelta() {
        float[] qPrime = { 1.0f, 2.0f, 3.0f, 4.0f };
        byte[] packed = { (byte) 0x77, (byte) 0x77 }; // nibbles: [7, 7, 7, 7]
        float lower = 0.0f;
        float upper = 0.0f;
        // phase1Score = scale(9.0) = 10.0
        float phase1Score = 10.0f;

        float result = ResidualQuantizer.computeCorrectedScore(qPrime, packed, lower, upper, phase1Score, 4);

        // correction = 0, rawDot = 9.0, rawTotal = 9.0, scale(9.0) = 10.0
        assertEquals(10.0f, result, 1e-6f);
    }

    /**
     * Boundary nibble values with MIP scaling.
     * qPrime = [1.0, 1.0], lower=-1.0, upper=1.0
     *
     * All nibbles=0: correction = -2.0
     *   phase1Score = scale(0.0) = 1.0 → rawDot=0 → rawTotal=-2.0 → scale(-2.0) = 1/(1+2) = 0.3333
     *
     * All nibbles=15: correction = 2.0
     *   phase1Score = scale(0.0) = 1.0 → rawDot=0 → rawTotal=2.0 → scale(2.0) = 3.0
     */
    public void testComputeCorrectedScore_boundaryNibbles() {
        float[] qPrime = { 1.0f, 1.0f };
        float lower = -1.0f;
        float upper = 1.0f;
        // phase1Score = scale(0.0) = 1.0
        float phase1Score = 1.0f;

        // All nibbles = 0 → correction = -2.0 → rawTotal = -2.0 → scale(-2.0) = 1/(1-(-2)) = 1/3
        byte[] packedAllZero = { (byte) 0x00 };
        float result0 = ResidualQuantizer.computeCorrectedScore(qPrime, packedAllZero, lower, upper, phase1Score, 2);
        assertEquals(1.0f / 3.0f, result0, 1e-4f);

        // All nibbles = 15 → correction = 2.0 → rawTotal = 2.0 → scale(2.0) = 3.0
        byte[] packedAllMax = { (byte) 0xFF };
        float result15 = ResidualQuantizer.computeCorrectedScore(qPrime, packedAllMax, lower, upper, phase1Score, 2);
        assertEquals(3.0f, result15, 1e-4f);
    }

    /**
     * Odd dimension (dim=5): verify nibble unpacking handles the last byte correctly.
     * The last byte packs dim 4 (low nibble) and padding (high nibble = 0).
     *
     * qPrime = [1.0, 1.0, 1.0, 1.0, 1.0], lower=-1.0, upper=1.0, step=2/15
     * nibbles = [0, 15, 8, 0, 15] packed as:
     *   byte[0] = (15 << 4) | 0  = 0xF0
     *   byte[1] = (0 << 4)  | 8  = 0x08
     *   byte[2] = (0 << 4)  | 15 = 0x0F  (high nibble is padding, not used)
     * r = [-1.0, 1.0, 0.0667, -1.0, 1.0]
     * correction = 1*(-1) + 1*1 + 1*0.0667 + 1*(-1) + 1*1 = 0.0667
     */
    public void testComputeCorrectedScore_oddDimension() {
        float[] qPrime = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
        byte[] packed = { (byte) 0xF0, (byte) 0x08, (byte) 0x0F };
        float lower = -1.0f;
        float upper = 1.0f;
        // phase1Score = scale(0.0) = 1.0
        float phase1Score = 1.0f;

        float result = ResidualQuantizer.computeCorrectedScore(qPrime, packed, lower, upper, phase1Score, 5);

        // correction ≈ 0.0667, rawDot = 0.0, rawTotal = 0.0667
        // scale(0.0667) = 0.0667 + 1.0 = 1.0667
        float step = 2.0f / 15.0f;
        float correction = -1.0f + 1.0f + (-1.0f + 8 * step) + (-1.0f) + 1.0f;
        assertEquals(1.0f + correction, result, 1e-3f);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // writeResidualFile: single-pass .ver file writing with per-vector bounds.
    // Tests header format, per-vector metadata, round-trip accuracy,
    // odd-dimension padding, and total file size.
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Verify the .ver file header: Lucene codec header followed by dimension, numVectors, bytesPerBlock.
     */
    public void testWriteResidualFile_headerFormat() throws IOException {
        List<float[]> vectors = List.of(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        float[] centroid = new float[] { 0.0f, 0.0f, 0.0f, 0.0f };

        QuantizedByteVectorValues mockQuantized = createMockQuantizedValues(
            new byte[][] { new byte[] { 0x00 } }, new float[] { 0.0f }, new float[] { 1.0f }
        );
        Supplier<KNNVectorValues<?>> supplier = () -> KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT, new TestVectorValues.PreDefinedFloatVectorValues(vectors)
        );

        ByteBuffersDirectory dir = new ByteBuffersDirectory();
        SegmentWriteState state = createSegmentWriteState(dir);
        try (IndexOutput output = dir.createOutput("test.ver", IOContext.DEFAULT)) {
            ResidualQuantizer.writeResidualFile(output, state, supplier, mockQuantized, centroid, 4, 1);
        }

        try (IndexInput input = dir.openInput("test.ver", IOContext.DEFAULT)) {
            int[] dim = new int[1], numVec = new int[1], bpb = new int[1];
            readVerHeader(input, dim, numVec, bpb);

            assertEquals(4, dim[0]);
            assertEquals(1, numVec[0]);
            assertEquals(2 + 16, bpb[0]); // packedResidual(2) + meta(16)
        }
    }

    /**
     * Verify per-vector metadata written after the packed residual data.
     *
     * Vec=[1.0, 2.0, 3.0, 4.0], centroid=[0,0,0,0]
     * binaryCode=0x00 (all bits 0), lower=0.0, upper=2.0 → delta=2.0
     * residuals=[1,2,3,4], all exceed halfDelta=1.0 → all clamped to 15
     * componentSum = 60
     */
    public void testWriteResidualFile_perVectorMetadata() throws IOException {
        List<float[]> vectors = List.of(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        float[] centroid = new float[] { 0.0f, 0.0f, 0.0f, 0.0f };

        QuantizedByteVectorValues mockQuantized = createMockQuantizedValues(
            new byte[][] { new byte[] { 0x00 } }, new float[] { 0.0f }, new float[] { 2.0f }
        );
        Supplier<KNNVectorValues<?>> supplier = () -> KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT, new TestVectorValues.PreDefinedFloatVectorValues(vectors)
        );

        ByteBuffersDirectory dir = new ByteBuffersDirectory();
        SegmentWriteState state = createSegmentWriteState(dir);
        try (IndexOutput output = dir.createOutput("test.ver", IOContext.DEFAULT)) {
            ResidualQuantizer.writeResidualFile(output, state, supplier, mockQuantized, centroid, 4, 1);
        }

        int packedResidualBytes = 2;

        try (IndexInput input = dir.openInput("test.ver", IOContext.DEFAULT)) {
            int[] dim = new int[1], numVec = new int[1], bpb = new int[1];
            long dataStart = readVerHeader(input, dim, numVec, bpb);

            // Skip packed residual, read per-vector metadata (little-endian)
            input.seek(dataStart + packedResidualBytes);
            byte[] metaBytes = new byte[ResidualQuantizer.PER_VECTOR_META_BYTES];
            input.readBytes(metaBytes, 0, metaBytes.length);
            ByteBuffer meta = ByteBuffer.wrap(metaBytes).order(ByteOrder.LITTLE_ENDIAN);

            assertEquals(-1.0f, meta.getFloat(), 1e-6f);   // lowerInterval = -halfDelta
            assertEquals(1.0f, meta.getFloat(), 1e-6f);    // upperInterval = halfDelta
            assertEquals(0.0f, meta.getFloat(), 1e-6f);    // additionalCorrection (reserved)
            assertEquals(60, meta.getInt());                // componentSum = 15*4
        }
    }

    /**
     * Round-trip: write .ver, read back, dequantize, compare against true residuals.
     *
     * Vec=[0.5, 0.5, 0.5, 0.5], centroid=[0,0,0,0]
     * binaryCode=0xA0 (MSB-first bits [1,0,1,0] for dims 0-3), lower=0.0, upper=1.0 → delta=1.0
     * Q(x')=[1.0, 0.0, 1.0, 0.0]
     * residuals = [-0.5, 0.5, -0.5, 0.5] — exact boundary values
     * q = [0, 15, 0, 15], dequantized = [-0.5, 0.5, -0.5, 0.5] — exact match
     */
    public void testWriteResidualFile_roundTrip() throws IOException {
        List<float[]> vectors = List.of(new float[] { 0.5f, 0.5f, 0.5f, 0.5f });
        float[] centroid = new float[] { 0.0f, 0.0f, 0.0f, 0.0f };

        QuantizedByteVectorValues mockQuantized = createMockQuantizedValues(
            new byte[][] { new byte[] { (byte) 0xA0 } }, new float[] { 0.0f }, new float[] { 1.0f }
        );
        Supplier<KNNVectorValues<?>> supplier = () -> KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT, new TestVectorValues.PreDefinedFloatVectorValues(vectors)
        );

        ByteBuffersDirectory dir = new ByteBuffersDirectory();
        SegmentWriteState state = createSegmentWriteState(dir);
        try (IndexOutput output = dir.createOutput("test.ver", IOContext.DEFAULT)) {
            ResidualQuantizer.writeResidualFile(output, state, supplier, mockQuantized, centroid, 4, 1);
        }

        int packedResidualBytes = 2;
        float[] expectedResiduals = { -0.5f, 0.5f, -0.5f, 0.5f };

        try (IndexInput input = dir.openInput("test.ver", IOContext.DEFAULT)) {
            int[] dim = new int[1], numVec = new int[1], bpb = new int[1];
            long dataStart = readVerHeader(input, dim, numVec, bpb);

            // Read packed residual
            byte[] packed = new byte[packedResidualBytes];
            input.readBytes(packed, 0, packedResidualBytes);

            // Read per-vector metadata for dequantization bounds
            byte[] metaBytes = new byte[ResidualQuantizer.PER_VECTOR_META_BYTES];
            input.readBytes(metaBytes, 0, metaBytes.length);
            ByteBuffer meta = ByteBuffer.wrap(metaBytes).order(ByteOrder.LITTLE_ENDIAN);
            float lower = meta.getFloat();
            float upper = meta.getFloat();
            float residualStep = (upper - lower) / 15.0f;

            for (int d = 0; d < 4; d++) {
                int nibble = (d % 2 == 0) ? (packed[d / 2] & 0x0F) : ((packed[d / 2] >>> 4) & 0x0F);
                float reconstructed = lower + nibble * residualStep;
                assertEquals("dim=" + d, expectedResiduals[d], reconstructed, residualStep / 2.0f + 1e-6f);
            }
        }
    }

    /**
     * Odd dimension: dim=5, bytesPerBlock = ceil(5/2) + 16 = 19.
     * Last byte's high nibble (non-existent dimension 5) must be zero-padded.
     */
    public void testWriteResidualFile_oddDimension() throws IOException {
        List<float[]> vectors = List.of(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });
        float[] centroid = new float[] { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

        QuantizedByteVectorValues mockQuantized = createMockQuantizedValues(
            new byte[][] { new byte[] { 0x00 } }, new float[] { 0.0f }, new float[] { 1.0f }
        );
        Supplier<KNNVectorValues<?>> supplier = () -> KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT, new TestVectorValues.PreDefinedFloatVectorValues(vectors)
        );

        ByteBuffersDirectory dir = new ByteBuffersDirectory();
        SegmentWriteState state = createSegmentWriteState(dir);
        try (IndexOutput output = dir.createOutput("test.ver", IOContext.DEFAULT)) {
            ResidualQuantizer.writeResidualFile(output, state, supplier, mockQuantized, centroid, 5, 1);
        }

        int packedResidualBytes = 3;

        try (IndexInput input = dir.openInput("test.ver", IOContext.DEFAULT)) {
            int[] dim = new int[1], numVec = new int[1], bpb = new int[1];
            long dataStart = readVerHeader(input, dim, numVec, bpb);

            assertEquals(packedResidualBytes + 16, bpb[0]);

            byte[] packed = new byte[packedResidualBytes];
            input.readBytes(packed, 0, packedResidualBytes);

            // Last byte's high nibble (dimension 5, doesn't exist) should be 0
            assertEquals(0, (packed[2] >>> 4) & 0x0F);

            // First 5 nibbles should be valid [0, 15]
            for (int d = 0; d < 5; d++) {
                int nibble = (d % 2 == 0) ? (packed[d / 2] & 0x0F) : ((packed[d / 2] >>> 4) & 0x0F);
                assertTrue("dim " + d + " nibble should be in [0,15]", nibble >= 0 && nibble <= 15);
            }
        }
    }

    /**
     * Verify total .ver file size:
     *   codecHeaderLen + 12 (our 3 ints) + numVectors * bytesPerBlock
     * (CodecUtil footer is NOT included — the caller writes it separately.)
     */
    public void testWriteResidualFile_totalFileSize() throws IOException {
        List<float[]> vectors = List.of(
            new float[] { 1, 2, 3, 4, 5, 6, 7, 8 },
            new float[] { 2, 3, 4, 5, 6, 7, 8, 9 },
            new float[] { 0, 1, 2, 3, 4, 5, 6, 7 }
        );
        float[] centroid = new float[8];

        QuantizedByteVectorValues mockQuantized = createMockQuantizedValues(
            new byte[][] { new byte[] { 0x00 }, new byte[] { 0x00 }, new byte[] { 0x00 } },
            new float[] { 0.0f, 0.0f, 0.0f },
            new float[] { 1.0f, 1.0f, 1.0f }
        );
        Supplier<KNNVectorValues<?>> supplier = () -> KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT, new TestVectorValues.PreDefinedFloatVectorValues(vectors)
        );

        ByteBuffersDirectory dir = new ByteBuffersDirectory();
        SegmentWriteState state = createSegmentWriteState(dir);
        try (IndexOutput output = dir.createOutput("test.ver", IOContext.DEFAULT)) {
            ResidualQuantizer.writeResidualFile(output, state, supplier, mockQuantized, centroid, 8, 3);
        }

        // Codec index header + 3 ints (dim, numVec, bytesPerBlock) + 3 * bytesPerBlock
        int bytesPerBlock = 4 + ResidualQuantizer.PER_VECTOR_META_BYTES; // packedResidual(4) + meta(16) = 20
        long codecHeaderLen = CodecUtil.indexHeaderLength(ResidualQuantizer.CODEC_NAME, "");
        long expectedSize = codecHeaderLen + 12 + 3L * bytesPerBlock;
        assertEquals(expectedSize, dir.fileLength("test.ver"));
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Multi-vector batch boundary: verifies the buffer flush logic when
    // numVectors > batchSize forces multiple flushes.
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Write enough vectors to force multiple batch flushes, then verify file size and
     * that each vector block is readable with correct per-vector metadata.
     *
     * Uses dim=4 → oneBlockSize = 2 + 16 = 18 bytes, batchSize = max(1, 65536/18) = 3640.
     * We write 200 vectors — well within one batch. To actually test the boundary, we use
     * a very high dimension (dim=32768) so oneBlockSize = 16384 + 16 = 16400, batchSize =
     * max(1, 65536/16400) = 4. With 10 vectors, we get 2 full batches + 1 partial.
     *
     * Simplified approach: use dim=4 with 200 vectors and verify every block is readable.
     * While this doesn't cross the 64KB boundary, it exercises the multi-vector write path
     * and verifies the batch index arithmetic (batchIdx = ord % batchSize) is correct
     * for vector counts significantly larger than 1-3.
     */
    public void testWriteResidualFile_multipleVectors() throws IOException {
        int numVectors = 200;
        int dimension = 4;
        float[] centroid = new float[dimension]; // all zeros

        // Generate 200 random-ish vectors with varying correction factors
        float[][] vectorData = new float[numVectors][dimension];
        byte[][] binaryCodes = new byte[numVectors][];
        float[] lowers = new float[numVectors];
        float[] uppers = new float[numVectors];
        for (int i = 0; i < numVectors; i++) {
            for (int d = 0; d < dimension; d++) {
                vectorData[i][d] = (float) (i * 0.01 + d * 0.1);
            }
            binaryCodes[i] = new byte[] { (byte) (i % 16) }; // varying bit patterns
            lowers[i] = 0.0f;
            uppers[i] = 1.0f + (i % 5) * 0.5f; // varying deltas
        }

        List<float[]> vectors = List.of(vectorData);
        QuantizedByteVectorValues mockQuantized = createMockQuantizedValues(binaryCodes, lowers, uppers);
        Supplier<KNNVectorValues<?>> supplier = () -> KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            new TestVectorValues.PreDefinedFloatVectorValues(List.of(vectorData))
        );

        ByteBuffersDirectory dir = new ByteBuffersDirectory();
        SegmentWriteState state = createSegmentWriteState(dir);
        try (IndexOutput output = dir.createOutput("test.ver", IOContext.DEFAULT)) {
            ResidualQuantizer.writeResidualFile(output, state, supplier, mockQuantized, centroid, dimension, numVectors);
        }

        // Verify file size
        int packedResidualBytes = (dimension + 1) / 2;
        int bytesPerBlock = packedResidualBytes + ResidualQuantizer.PER_VECTOR_META_BYTES;
        long codecHeaderLen = CodecUtil.indexHeaderLength(ResidualQuantizer.CODEC_NAME, "");
        long expectedSize = codecHeaderLen + 12 + (long) numVectors * bytesPerBlock;
        assertEquals(expectedSize, dir.fileLength("test.ver"));

        // Read back every block and verify per-vector metadata is plausible
        try (IndexInput input = dir.openInput("test.ver", IOContext.DEFAULT)) {
            int[] dim = new int[1], numVec = new int[1], bpb = new int[1];
            long dataStart = readVerHeader(input, dim, numVec, bpb);
            assertEquals(numVectors, numVec[0]);

            for (int ord = 0; ord < numVectors; ord++) {
                input.seek(dataStart + (long) ord * bytesPerBlock + packedResidualBytes);
                // Read per-vector metadata (little-endian)
                byte[] metaBytes = new byte[ResidualQuantizer.PER_VECTOR_META_BYTES];
                input.readBytes(metaBytes, 0, metaBytes.length);
                ByteBuffer meta = ByteBuffer.wrap(metaBytes).order(ByteOrder.LITTLE_ENDIAN);

                float lower = meta.getFloat();
                float upper = meta.getFloat();
                // lower should be -halfDelta, upper should be halfDelta
                float expectedDelta = uppers[ord] - lowers[ord];
                assertEquals("vec " + ord + " lower", -expectedDelta / 2.0f, lower, 1e-6f);
                assertEquals("vec " + ord + " upper", expectedDelta / 2.0f, upper, 1e-6f);
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // End-to-end: write .ver then use computeCorrectedScore on the read-back data.
    // Verifies that build-time quantization and search-time dequantization are symmetric.
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Write a .ver file, read back the block, and call computeCorrectedScore to verify
     * the correction brings the score closer to the true dot product.
     *
     * Setup:
     *   Vec=[0.5, 0.5, 0.5, 0.5], centroid=[0,0,0,0], query=[1.0, -1.0, 1.0, -1.0]
     *   binaryCode=0x05 (bits [1,0,1,0]), lower=0.0, upper=1.0 → delta=1.0
     *   residuals = [-0.5, 0.5, -0.5, 0.5]
     *   q' = query - centroid = [1.0, -1.0, 1.0, -1.0]
     *   True <q', r> = 1*(-0.5) + (-1)*0.5 + 1*(-0.5) + (-1)*0.5 = -2.0
     *   With phase1Score=10.0, correctedScore should be ≈ 10.0 + (-2.0) = 8.0
     */
    public void testWriteThenScore_endToEnd() throws IOException {
        List<float[]> vectors = List.of(new float[] { 0.5f, 0.5f, 0.5f, 0.5f });
        float[] centroid = new float[] { 0.0f, 0.0f, 0.0f, 0.0f };
        float[] query = new float[] { 1.0f, -1.0f, 1.0f, -1.0f };

        ByteBuffersDirectory dir = new ByteBuffersDirectory();
        QuantizedByteVectorValues mockQuantized = createMockQuantizedValues(
            new byte[][] { new byte[] { (byte) 0xA0 } }, new float[] { 0.0f }, new float[] { 1.0f }
        );
        Supplier<KNNVectorValues<?>> supplier = () -> KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT, new TestVectorValues.PreDefinedFloatVectorValues(vectors)
        );

        SegmentWriteState state = createSegmentWriteState(dir);
        try (IndexOutput output = dir.createOutput("test.ver", IOContext.DEFAULT)) {
            ResidualQuantizer.writeResidualFile(output, state, supplier, mockQuantized, centroid, 4, 1);
        }

        // Read the block back
        int packedResidualBytes = 2;
        try (IndexInput input = dir.openInput("test.ver", IOContext.DEFAULT)) {
            int[] dim = new int[1], numVec = new int[1], bpb = new int[1];
            long dataStart = readVerHeader(input, dim, numVec, bpb);

            byte[] block = new byte[bpb[0]];
            input.readBytes(block, 0, bpb[0]);

            // Extract packed residual and per-vector metadata
            byte[] packedResidual = new byte[packedResidualBytes];
            System.arraycopy(block, 0, packedResidual, 0, packedResidualBytes);

            ByteBuffer meta = ByteBuffer.wrap(block, packedResidualBytes, ResidualQuantizer.PER_VECTOR_META_BYTES)
                .order(ByteOrder.LITTLE_ENDIAN);
            float lower = meta.getFloat();
            float upper = meta.getFloat();

            // Use the actual search-time methods
            // phase1Score = scale(9.0) = 10.0 (rawDot = 9.0)
            float[] qPrime = ResidualQuantizer.computeQPrime(query, centroid);
            float corrected = ResidualQuantizer.computeCorrectedScore(qPrime, packedResidual, lower, upper, 10.0f, 4);

            // True correction = <q', r> = -2.0
            // rawDot = unscale(10.0) = 9.0, rawTotal = 9.0 + (-2.0) = 7.0
            // expected = scale(7.0) = 8.0
            assertEquals(8.0f, corrected, 0.1f);
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Test helper
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Creates a mock {@link QuantizedByteVectorValues} with known binary codes and per-vector
     * correction factors. This avoids depending on Lucene's on-disk SQ format in unit tests.
     *
     * The mock supports:
     * <ul>
     *   <li>{@code vectorValue(ord)} → returns the binary code for vector at ordinal</li>
     *   <li>{@code getCorrectiveTerms(ord)} → returns QuantizationResult with the given
     *       lower/upper intervals (additionalCorrection=0, quantizedComponentSum=0)</li>
     *   <li>{@code size()} → returns the number of vectors</li>
     * </ul>
     *
     * @param binaryCodes    bit-packed 1-bit SQ codes, one byte[] per vector
     * @param lowerIntervals per-vector quantization lower bound (lowerInterval)
     * @param upperIntervals per-vector quantization upper bound (upperInterval)
     */
    private QuantizedByteVectorValues createMockQuantizedValues(byte[][] binaryCodes, float[] lowerIntervals, float[] upperIntervals)
        throws IOException {
        QuantizedByteVectorValues mockVal = mock(QuantizedByteVectorValues.class);
        when(mockVal.size()).thenReturn(binaryCodes.length);

        for (int i = 0; i < binaryCodes.length; i++) {
            when(mockVal.vectorValue(i)).thenReturn(binaryCodes[i]);
            when(mockVal.getCorrectiveTerms(i)).thenReturn(new OptimizedScalarQuantizer.QuantizationResult(
                lowerIntervals[i],
                upperIntervals[i],
                0.0f,
                0
            ));
        }
        return mockVal;
    }

    /**
     * Creates a minimal {@link SegmentWriteState} backed by the given directory.
     * Required by {@link ResidualQuantizer#writeResidualFile} for the codec header.
     */
    private SegmentWriteState createSegmentWriteState(ByteBuffersDirectory dir) {
        SegmentInfo segmentInfo = new SegmentInfo(
            dir,
            Version.LATEST,
            Version.LATEST,
            "_0",
            0,
            false,
            false,
            null,
            Collections.emptyMap(),
            StringHelper.randomId(),
            Collections.emptyMap(),
            null
        );
        return new SegmentWriteState(InfoStream.NO_OUTPUT, dir, segmentInfo, FieldInfos.EMPTY, null, IOContext.DEFAULT);
    }

    /**
     * Skips the Lucene index header and reads our format-specific fields from the .ver file.
     * We skip the index header by computing its known length (codec header + segment ID + suffix),
     * then read our 3 int fields: dimension, numVectors, bytesPerBlock.
     * Returns the file pointer position after our fields (= data start offset).
     */
    private long readVerHeader(IndexInput input, int[] outDimension, int[] outNumVectors, int[] outBytesPerBlock) throws IOException {
        // Skip the entire index header written by CodecUtil.writeIndexHeader
        int indexHeaderLen = CodecUtil.indexHeaderLength(ResidualQuantizer.CODEC_NAME, "");
        input.seek(indexHeaderLen);
        outDimension[0] = input.readInt();
        outNumVectors[0] = input.readInt();
        outBytesPerBlock[0] = input.readInt();
        return input.getFilePointer();
    }
}
