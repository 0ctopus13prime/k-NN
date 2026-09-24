/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.clusterann.codec.*;
import org.opensearch.knn.jni.SimdVectorComputeService;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import java.util.Random;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

/**
 * Tests for multi-bit quantization packing and dot product methods.
 * Verifies correctness of 1-bit, 2-bit, and 4-bit paths end-to-end.
 */
public class ClusterANNQuantizationTests extends KNNTestCase {

    private static final int DIM = 128;
    private static final long SEED = 99L;

    // ========== Packing Size ==========

    public void testPackedBytesPerVector_1bit() {
        assertEquals(16, QuantizedVectorWriter.packedBytesPerVector(128, 1));
        assertEquals(4, QuantizedVectorWriter.packedBytesPerVector(32, 1));
        assertEquals(1, QuantizedVectorWriter.packedBytesPerVector(8, 1));
        assertEquals(1, QuantizedVectorWriter.packedBytesPerVector(1, 1));
    }

    public void testPackedBytesPerVector_2bit() {
        assertEquals(32, QuantizedVectorWriter.packedBytesPerVector(128, 2));
        assertEquals(8, QuantizedVectorWriter.packedBytesPerVector(32, 2));
        assertEquals(2, QuantizedVectorWriter.packedBytesPerVector(8, 2));
    }

    public void testPackedBytesPerVector_4bit() {
        assertEquals(64, QuantizedVectorWriter.packedBytesPerVector(128, 4));
        assertEquals(16, QuantizedVectorWriter.packedBytesPerVector(32, 4));
        assertEquals(4, QuantizedVectorWriter.packedBytesPerVector(8, 4));
    }

    // ========== 1-bit Pack + Dot Product ==========

    public void testPackAsBinary_allOnes() {
        byte[] raw = new byte[8];
        for (int i = 0; i < 8; i++)
            raw[i] = 1;
        byte[] packed = new byte[1];
        OptimizedScalarQuantizer.packAsBinary(raw, packed);
        assertEquals((byte) 0xFF, packed[0]); // all bits set MSB-first
    }

    public void testPackAsBinary_alternating() {
        byte[] raw = new byte[] { 1, 0, 1, 0, 1, 0, 1, 0 };
        byte[] packed = new byte[1];
        OptimizedScalarQuantizer.packAsBinary(raw, packed);
        assertEquals((byte) 0xAA, packed[0]); // 10101010
    }

    public void testInt4BitDotProduct_allOnes() {
        // Query: all 15 (4-bit max), Doc: all 1s
        int dim = 8;
        int stripeSize = 1; // (8+7)/8 = 1
        byte[] queryTransposed = new byte[4]; // 4 stripes
        queryTransposed[0] = (byte) 0xFF; // bit0 all set
        queryTransposed[1] = (byte) 0xFF; // bit1 all set
        queryTransposed[2] = (byte) 0xFF; // bit2 all set
        queryTransposed[3] = (byte) 0xFF; // bit3 all set

        byte[] docPacked = new byte[] { (byte) 0xFF }; // all 1-bits

        long dot = VectorUtil.int4BitDotProduct(queryTransposed, docPacked);
        // Each of 8 positions: query=15, doc=1, contribution = 15
        // Total = 8 * 15 = 120
        // Formula: sum of bitCount(q_stripe & doc) * weight
        // = 8*1 + 8*2 + 8*4 + 8*8 = 8 + 16 + 32 + 64 = 120
        assertEquals(120L, dot);
    }

    public void testInt4BitDotProduct_zeros() {
        byte[] queryTransposed = new byte[4];
        byte[] docPacked = new byte[] { (byte) 0xFF };
        long dot = VectorUtil.int4BitDotProduct(queryTransposed, docPacked);
        assertEquals(0L, dot);
    }

    // ========== 2-bit Pack + Dot Product ==========

    public void testTransposeDibit_allThrees() {
        // All values = 3 (binary 11) → lower stripe all 1s, upper stripe all 1s
        byte[] raw = new byte[8];
        for (int i = 0; i < 8; i++)
            raw[i] = 3;
        byte[] packed = new byte[2]; // 2 stripes of 1 byte each
        OptimizedScalarQuantizer.transposeDibit(raw, packed);
        assertEquals((byte) 0xFF, packed[0]); // lower bits all 1
        assertEquals((byte) 0xFF, packed[1]); // upper bits all 1
    }

    public void testTransposeDibit_alternating() {
        // Values: 1, 2, 1, 2, 1, 2, 1, 2
        // 1 = binary 01 (lower=1, upper=0)
        // 2 = binary 10 (lower=0, upper=1)
        byte[] raw = new byte[] { 1, 2, 1, 2, 1, 2, 1, 2 };
        byte[] packed = new byte[2];
        OptimizedScalarQuantizer.transposeDibit(raw, packed);
        assertEquals((byte) 0xAA, packed[0]); // lower: 10101010
        assertEquals((byte) 0x55, packed[1]); // upper: 01010101
    }

    public void testInt4DibitDotProduct_maxValues() {
        // Query: all 15 (4-bit), Doc: all 3 (2-bit)
        int stripeSize = 1;
        byte[] queryTransposed = new byte[4];
        for (int i = 0; i < 4; i++)
            queryTransposed[i] = (byte) 0xFF;

        byte[] dibitPacked = new byte[2];
        dibitPacked[0] = (byte) 0xFF; // lower all 1
        dibitPacked[1] = (byte) 0xFF; // upper all 1

        long dot = VectorUtil.int4DibitDotProduct(queryTransposed, dibitPacked);
        // Each position: query=15, doc=3, contribution = 15*3 = 45
        // 8 positions: 8 * 45 = 360
        // Formula: sum of all cross-products
        // Per byte: (8+8*2) + (8*2+8*4) + (8*4+8*8) + (8*8+8*16) = 24 + 48 + 96 + 192 = 360
        assertEquals(360L, dot);
    }

    // ========== 4-bit Pack + Dot Product ==========

    public void testTransposeHalfByte_value15() {
        // All values = 15 (binary 1111) → all 4 stripes should be all 1s
        byte[] raw = new byte[8];
        for (int i = 0; i < 8; i++)
            raw[i] = 15;
        byte[] packed = new byte[4]; // 4 stripes of 1 byte
        OptimizedScalarQuantizer.transposeHalfByte(raw, packed);
        for (int s = 0; s < 4; s++) {
            assertEquals("Stripe " + s + " should be all 1s", (byte) 0xFF, packed[s]);
        }
    }

    public void testTransposeHalfByte_value5() {
        // Value 5 = binary 0101 → stripe0=1, stripe1=0, stripe2=1, stripe3=0
        byte[] raw = new byte[8];
        for (int i = 0; i < 8; i++)
            raw[i] = 5;
        byte[] packed = new byte[4];
        OptimizedScalarQuantizer.transposeHalfByte(raw, packed);
        assertEquals((byte) 0xFF, packed[0]); // bit0 stripe
        assertEquals((byte) 0x00, packed[1]); // bit1 stripe
        assertEquals((byte) 0xFF, packed[2]); // bit2 stripe
        assertEquals((byte) 0x00, packed[3]); // bit3 stripe
    }

    public void testPackNibbles_splitHalf() {
        // 8 dims, codes 1..8 -> byte i = code[i] << 4 | code[i + 4], identical to Lucene104 packNibbles.
        byte[] raw = { 1, 2, 3, 4, 5, 6, 7, 8 };
        byte[] packed = new byte[4];
        ScalarBitEncoding.FOUR_BIT.packDoc(raw, packed, 8);
        assertEquals((byte) 0x15, packed[0]);
        assertEquals((byte) 0x26, packed[1]);
        assertEquals((byte) 0x37, packed[2]);
        assertEquals((byte) 0x48, packed[3]);
    }

    public void testPackNibbles_oddDimensionPadsLowNibble() {
        // 5 dims -> half = 3 -> 3 bytes; the last byte's low nibble (dim 5) does not exist and must be 0.
        byte[] raw = { 15, 14, 13, 12, 11 };
        byte[] packed = new byte[3];
        ScalarBitEncoding.FOUR_BIT.packDoc(raw, packed, 5);
        assertEquals(3, ScalarBitEncoding.FOUR_BIT.docPackedBytes(5));
        assertEquals((byte) 0xFC, packed[0]); // 15 | 12
        assertEquals((byte) 0xEB, packed[1]); // 14 | 11
        assertEquals((byte) 0xD0, packed[2]); // 13 | pad
    }

    public void testInt4NibbleDotProduct_identity() {
        // 8 dims, query and doc both all 15 -> dot = 8 * 15 * 15 = 1800
        byte[] queryUnpacked = new byte[8];
        java.util.Arrays.fill(queryUnpacked, (byte) 15);
        byte[] docPacked = new byte[4];
        java.util.Arrays.fill(docPacked, (byte) 0xFF);

        long dot = QuantizedVectorReader.int4NibbleDotProduct(queryUnpacked, docPacked);
        assertEquals(1800L, dot);
    }

    public void testInt4NibbleDotProduct_zeros() {
        byte[] queryUnpacked = new byte[8];
        byte[] docPacked = new byte[4];
        long dot = QuantizedVectorReader.int4NibbleDotProduct(queryUnpacked, docPacked);
        assertEquals(0L, dot);
    }

    // ========== Roundtrip: Pack → Dot Product ==========

    public void testRoundtrip_1bit() {
        Random rng = new Random(SEED);
        byte[] rawDoc = new byte[DIM];
        byte[] rawQuery = new byte[DIM];
        for (int i = 0; i < DIM; i++) {
            rawDoc[i] = (byte) (rng.nextInt(2));   // 0 or 1
            rawQuery[i] = (byte) (rng.nextInt(16)); // 0-15
        }

        // Pack
        byte[] packedDoc = new byte[QuantizedVectorWriter.packedBytesPerVector(DIM, 1)];
        OptimizedScalarQuantizer.packAsBinary(rawDoc, packedDoc);

        int stripeSize = (DIM + 7) / 8;
        byte[] queryTransposed = new byte[stripeSize * 4];
        OptimizedScalarQuantizer.transposeHalfByte(rawQuery, queryTransposed);

        // Dot product via bit manipulation
        long bitDot = VectorUtil.int4BitDotProduct(queryTransposed, packedDoc);

        // Brute force reference
        long expected = 0;
        for (int i = 0; i < DIM; i++) {
            expected += (long) rawQuery[i] * rawDoc[i];
        }

        assertEquals("1-bit roundtrip dot product mismatch", expected, bitDot);
    }

    public void testRoundtrip_2bit() {
        Random rng = new Random(SEED);
        byte[] rawDoc = new byte[DIM];
        byte[] rawQuery = new byte[DIM];
        for (int i = 0; i < DIM; i++) {
            rawDoc[i] = (byte) (rng.nextInt(4));   // 0-3
            rawQuery[i] = (byte) (rng.nextInt(16)); // 0-15
        }

        // Pack
        byte[] packedDoc = new byte[QuantizedVectorWriter.packedBytesPerVector(DIM, 2)];
        OptimizedScalarQuantizer.transposeDibit(rawDoc, packedDoc);

        int stripeSize = (DIM + 7) / 8;
        byte[] queryTransposed = new byte[stripeSize * 4];
        OptimizedScalarQuantizer.transposeHalfByte(rawQuery, queryTransposed);

        // Dot product via bit manipulation
        long bitDot = VectorUtil.int4DibitDotProduct(queryTransposed, packedDoc);

        // Brute force reference
        long expected = 0;
        for (int i = 0; i < DIM; i++) {
            expected += (long) rawQuery[i] * rawDoc[i];
        }

        assertEquals("2-bit roundtrip dot product mismatch", expected, bitDot);
    }

    public void testRoundtrip_4bit() {
        Random rng = new Random(SEED);
        byte[] rawDoc = new byte[DIM];
        byte[] rawQuery = new byte[DIM];
        for (int i = 0; i < DIM; i++) {
            rawDoc[i] = (byte) (rng.nextInt(16));  // 0-15
            rawQuery[i] = (byte) (rng.nextInt(16)); // 0-15
        }

        // Pack doc as split-half nibbles; the query stays unpacked (one code per dimension)
        byte[] packedDoc = new byte[ScalarBitEncoding.FOUR_BIT.docPackedBytes(DIM)];
        ScalarBitEncoding.FOUR_BIT.packDoc(rawDoc, packedDoc, DIM);

        long bitDot = QuantizedVectorReader.int4NibbleDotProduct(rawQuery, packedDoc);

        // Brute force reference
        long expected = 0;
        for (int i = 0; i < DIM; i++) {
            expected += (long) rawQuery[i] * rawDoc[i];
        }

        assertEquals("4-bit roundtrip dot product mismatch", expected, bitDot);
    }

    // ========== Native bulk dot product (jni/src/simd/similarity_function/clusterann_batch_dot_product.cpp) ==========

    private static boolean nativeBulkDotAvailable() {
        try {
            SimdVectorComputeService.bulkQuantizedDotProductKernel();
            return true;
        } catch (Throwable t) {
            return false;
        }
    }

    /** Pack per-dimension codes the way QuantizedVectorWriter does: bit planes for 1/2 bits, nibbles for 4. */
    private static byte[] packDoc(byte[] raw, int docBits) {
        int dim = raw.length;
        byte[] packed = new byte[ScalarBitEncoding.fromDocBits(docBits).docPackedBytes(dim)];
        switch (docBits) {
            case 1 -> OptimizedScalarQuantizer.packAsBinary(raw, packed);
            case 2 -> OptimizedScalarQuantizer.transposeDibit(raw, packed);
            default -> ScalarBitEncoding.FOUR_BIT.packDoc(raw, packed, dim);
        }
        return packed;
    }

    /** Prepare the query the way QuantizedVectorReader does: 4 transposed planes for 1/2 bits, unpacked codes for 4. */
    private static byte[] packQuery(byte[] rawQuery, int docBits) {
        int dim = rawQuery.length;
        if (docBits == 4) {
            byte[] q = new byte[2 * ScalarBitEncoding.FOUR_BIT.docPackedBytes(dim)];
            System.arraycopy(rawQuery, 0, q, 0, dim);
            return q;
        }
        byte[] transposed = new byte[((dim + 7) / 8) * 4];
        OptimizedScalarQuantizer.transposeHalfByte(rawQuery, transposed);
        return transposed;
    }

    /** Java reference used by QuantizedVectorReader for each docBits. */
    private static long javaDot(byte[] query, byte[] packedDoc, int docBits) {
        return switch (docBits) {
            case 1 -> VectorUtil.int4BitDotProduct(query, packedDoc);
            case 2 -> VectorUtil.int4DibitDotProduct(query, packedDoc);
            default -> QuantizedVectorReader.int4NibbleDotProduct(query, packedDoc);
        };
    }

    public void testNativeBulkDotProduct_matchesJava() {
        assumeTrue("native SIMD library not available (jni/build/release)", nativeBulkDotAvailable());
        Random rng = new Random(SEED);
        int[] dims = { 8, 100, 129, 768, 1024 };
        for (int docBits : new int[] { 1, 2, 4 }) {
            for (int dim : dims) {
                int bytesPerCode = ScalarBitEncoding.fromDocBits(docBits).docPackedBytes(dim);
                for (int n : new int[] { 1, 3, 4, 7, 8, 9, 17, 32, 33 }) {
                    byte[] rawQuery = new byte[dim];
                    for (int i = 0; i < dim; i++) rawQuery[i] = (byte) rng.nextInt(16);
                    byte[] queryTransposed = packQuery(rawQuery, docBits);

                    byte[] block = new byte[n * bytesPerCode];
                    byte[][] docs = new byte[n][];
                    for (int v = 0; v < n; v++) {
                        byte[] raw = new byte[dim];
                        for (int i = 0; i < dim; i++) raw[i] = (byte) rng.nextInt(1 << docBits);
                        docs[v] = packDoc(raw, docBits);
                        System.arraycopy(docs[v], 0, block, v * bytesPerCode, bytesPerCode);
                    }

                    // Score a shuffled subset; untouched slots must keep their sentinel.
                    int[] offsets = new int[n];
                    int count = 0;
                    for (int v = 0; v < n; v++) if (v == 0 || rng.nextInt(4) != 0) offsets[count++] = v;
                    for (int i = count - 1; i > 0; i--) {
                        int j = rng.nextInt(i + 1);
                        int t = offsets[i]; offsets[i] = offsets[j]; offsets[j] = t;
                    }
                    float[] results = new float[n];
                    java.util.Arrays.fill(results, -1f);

                    SimdVectorComputeService.bulkQuantizedDotProduct(queryTransposed, block, offsets, count, results, bytesPerCode, docBits);

                    boolean[] scored = new boolean[n];
                    for (int k = 0; k < count; k++) scored[offsets[k]] = true;
                    for (int v = 0; v < n; v++) {
                        String ctx = "docBits=" + docBits + " dim=" + dim + " n=" + n + " v=" + v;
                        if (!scored[v]) {
                            assertEquals("untouched slot overwritten " + ctx, -1f, results[v], 0f);
                            continue;
                        }
                        long expected = javaDot(queryTransposed, docs[v], docBits);
                        assertEquals("native vs java mismatch " + ctx, expected, (long) results[v]);
                    }
                }
            }
        }
    }

    public void testNativeBulkDotProduct_allMaxCodes() {
        assumeTrue("native SIMD library not available (jni/build/release)", nativeBulkDotAvailable());
        // Overflow guard: query all 15, docs all (2^bits - 1), large dimension.
        int dim = 4096;
        byte[] rawQuery = new byte[dim];
        java.util.Arrays.fill(rawQuery, (byte) 15);
        for (int docBits : new int[] { 1, 2, 4 }) {
            byte[] queryTransposed = packQuery(rawQuery, docBits);
            byte[] raw = new byte[dim];
            java.util.Arrays.fill(raw, (byte) ((1 << docBits) - 1));
            byte[] doc = packDoc(raw, docBits);
            int n = 8;
            byte[] block = new byte[n * doc.length];
            for (int v = 0; v < n; v++) System.arraycopy(doc, 0, block, v * doc.length, doc.length);
            int[] offsets = new int[n];
            for (int v = 0; v < n; v++) offsets[v] = v;
            float[] results = new float[n];
            SimdVectorComputeService.bulkQuantizedDotProduct(queryTransposed, block, offsets, n, results, doc.length, docBits);
            long expected = (long) dim * 15 * ((1 << docBits) - 1);
            for (int v = 0; v < n; v++) assertEquals("docBits=" + docBits, expected, (long) results[v]);
        }
    }

    public void testNativeBulkDotProduct_rejectsBadOffsets() {
        assumeTrue("native SIMD library not available (jni/build/release)", nativeBulkDotAvailable());
        int dim = 64, docBits = 4, bytesPerCode = ScalarBitEncoding.FOUR_BIT.docPackedBytes(dim);
        byte[] query = new byte[2 * bytesPerCode];
        byte[] block = new byte[2 * bytesPerCode];
        float[] results = new float[2];
        expectThrows(Exception.class, () ->
            SimdVectorComputeService.bulkQuantizedDotProduct(query, block, new int[] { 0, 2 }, 2, results, bytesPerCode, docBits));
        expectThrows(Exception.class, () ->
            SimdVectorComputeService.bulkQuantizedDotProduct(query, block, new int[] { -1 }, 1, results, bytesPerCode, docBits));
        expectThrows(Exception.class, () ->
            SimdVectorComputeService.bulkQuantizedDotProduct(query, block, new int[] { 0 }, 1, results, bytesPerCode, 3));
    }

    // ========== Validation ==========

    public void testInvalidDocBits_throws() {
        // Validation happens in writer constructor
        expectThrows(IllegalArgumentException.class, () -> new ClusterANN1040KnnVectorsWriter(null, null, 3));
        expectThrows(IllegalArgumentException.class, () -> new ClusterANN1040KnnVectorsWriter(null, null, 0));
        expectThrows(IllegalArgumentException.class, () -> new ClusterANN1040KnnVectorsWriter(null, null, 8));
    }

    public void testValidDocBits_noThrow() {
        // Format accepts any value, writer validates
        new ClusterANN1040KnnVectorsFormat(1);
        new ClusterANN1040KnnVectorsFormat(2);
        new ClusterANN1040KnnVectorsFormat(4);
    }

    // ========== Non-aligned dimensions ==========

    public void testPackAsBinary_nonAligned() {
        // 13 dimensions (not multiple of 8)
        byte[] raw = new byte[13];
        for (int i = 0; i < 13; i++)
            raw[i] = 1;
        byte[] packed = new byte[2]; // (13+7)/8 = 2
        OptimizedScalarQuantizer.packAsBinary(raw, packed);
        assertEquals((byte) 0xFF, packed[0]); // first 8 bits
        assertEquals((byte) 0xF8, packed[1]); // 5 bits set: 11111000
    }

    public void testTransposeDibit_nonAligned() {
        // 5 dimensions
        byte[] raw = new byte[] { 3, 3, 3, 3, 3 };
        byte[] packed = new byte[2]; // (5+7)/8 * 2 = 2
        OptimizedScalarQuantizer.transposeDibit(raw, packed);
        // 5 bits set MSB-first: 11111000 = 0xF8
        assertEquals((byte) 0xF8, packed[0]);
        assertEquals((byte) 0xF8, packed[1]);
    }
}
