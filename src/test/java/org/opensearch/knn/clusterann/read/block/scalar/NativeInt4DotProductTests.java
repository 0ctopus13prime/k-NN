/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar;

import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * {@link NativeInt4DotProduct} against {@link Int4DotProduct}, the scalar Java kernels the block scorer falls back to.
 * Same query bytes, same code block, same integer result, for every encoding, many dimensions and block sizes,
 * shuffled partial position sets, and the all-max overflow case. Skipped when the SIMD library is not built
 * ({@code jni/build/release}).
 */
@EnabledIf("nativeAvailable")
class NativeInt4DotProductTests {

    static boolean nativeAvailable() {
        return NativeInt4DotProduct.AVAILABLE;
    }

    @ParameterizedTest(name = "docBits {0}")
    @ValueSource(ints = { 1, 2, 4 })
    void testBulk_matchesJavaKernels(int docBits) {
        ScalarEncoding encoding = ScalarEncoding.fromNumBits(docBits);
        Random random = new Random(docBits * 17L);
        for (int dim : new int[] { 8, 100, 129, 768, 1024 }) {
            int packedBytes = encoding.getDocPackedLength(dim);
            for (int n : new int[] { 1, 3, 4, 7, 8, 9, 17, 32, 33 }) {
                byte[] query = query(random, dim, encoding);
                byte[] block = new byte[n * packedBytes];
                for (int v = 0; v < n; v++) {
                    System.arraycopy(doc(random, dim, encoding), 0, block, v * packedBytes, packedBytes);
                }

                // A shuffled subset of positions; untouched result slots must keep their sentinel.
                int[] positions = new int[n];
                int count = 0;
                for (int v = 0; v < n; v++) {
                    if (v == 0 || random.nextInt(4) != 0) positions[count++] = v;
                }
                for (int i = count - 1; i > 0; i--) {
                    int j = random.nextInt(i + 1);
                    int t = positions[i];
                    positions[i] = positions[j];
                    positions[j] = t;
                }
                float[] results = new float[n];
                Arrays.fill(results, -1f);

                NativeInt4DotProduct.bulk(query, block, positions, count, results, packedBytes, docBits);

                boolean[] scored = new boolean[n];
                for (int k = 0; k < count; k++)
                    scored[positions[k]] = true;
                for (int v = 0; v < n; v++) {
                    String ctx = "docBits=" + docBits + " dim=" + dim + " n=" + n + " v=" + v;
                    if (!scored[v]) {
                        assertEquals(-1f, results[v], 0f, "untouched slot overwritten " + ctx);
                        continue;
                    }
                    assertEquals(javaDot(query, block, v * packedBytes, packedBytes, docBits), results[v], 0f, ctx);
                }
            }
        }
    }

    @ParameterizedTest(name = "docBits {0}")
    @ValueSource(ints = { 1, 2, 4 })
    void testBulk_allMaxCodes_doesNotOverflow(int docBits) {
        ScalarEncoding encoding = ScalarEncoding.fromNumBits(docBits);
        int dim = 4096;
        byte[] rawQuery = new byte[dim];
        Arrays.fill(rawQuery, (byte) 15);
        byte[] query = packQuery(rawQuery, encoding);
        byte[] raw = new byte[dim];
        Arrays.fill(raw, (byte) ((1 << docBits) - 1));
        byte[] doc = packDoc(raw, encoding);
        int n = 8;
        byte[] block = new byte[n * doc.length];
        for (int v = 0; v < n; v++)
            System.arraycopy(doc, 0, block, v * doc.length, doc.length);
        int[] positions = new int[n];
        for (int v = 0; v < n; v++)
            positions[v] = v;
        float[] results = new float[n];

        NativeInt4DotProduct.bulk(query, block, positions, n, results, doc.length, docBits);

        float expected = (float) ((long) dim * 15 * ((1 << docBits) - 1));
        for (int v = 0; v < n; v++)
            assertEquals(expected, results[v], 0f, "docBits=" + docBits);
    }

    @Test
    void testBulk_rejectsBadArguments() {
        int dim = 64, docBits = 4;
        int packedBytes = ScalarEncoding.PACKED_NIBBLE.getDocPackedLength(dim);
        byte[] query = new byte[2 * packedBytes];
        byte[] block = new byte[2 * packedBytes];
        float[] results = new float[2];
        assertThrows(Exception.class, () -> NativeInt4DotProduct.bulk(query, block, new int[] { 0, 2 }, 2, results, packedBytes, docBits));
        assertThrows(Exception.class, () -> NativeInt4DotProduct.bulk(query, block, new int[] { -1 }, 1, results, packedBytes, docBits));
        assertThrows(Exception.class, () -> NativeInt4DotProduct.bulk(query, block, new int[] { 0 }, 1, results, packedBytes, 3));
        assertThrows(
            Exception.class,
            () -> NativeInt4DotProduct.bulk(new byte[packedBytes], block, new int[] { 0 }, 1, results, packedBytes, docBits)
        );
    }

    // ------------------------------------------------------------------------------------------------------------

    private static byte[] query(Random random, int dim, ScalarEncoding encoding) {
        byte[] raw = new byte[dim];
        for (int i = 0; i < dim; i++)
            raw[i] = (byte) random.nextInt(16);
        return packQuery(raw, encoding);
    }

    private static byte[] doc(Random random, int dim, ScalarEncoding encoding) {
        byte[] raw = new byte[dim];
        for (int i = 0; i < dim; i++)
            raw[i] = (byte) random.nextInt(1 << encoding.getBits());
        return packDoc(raw, encoding);
    }

    /** What {@code ScalarQuantizedCluster.prepareScan} hands the scorer: transposed planes, or unpacked codes for nibbles. */
    private static byte[] packQuery(byte[] rawQuery, ScalarEncoding encoding) {
        int dim = rawQuery.length;
        byte[] codes = Arrays.copyOf(rawQuery, encoding.getDiscreteDimensions(dim));
        if (encoding.isAsymmetric()) {
            byte[] transposed = new byte[encoding.getQueryPackedLength(dim)];
            OptimizedScalarQuantizer.transposeHalfByte(codes, transposed);
            return transposed;
        }
        return codes;
    }

    /** What {@code OptimizedScalarQuantizedBlockWriter} writes for one vector. */
    private static byte[] packDoc(byte[] raw, ScalarEncoding encoding) {
        int dim = raw.length;
        byte[] codes = Arrays.copyOf(raw, encoding.getDiscreteDimensions(dim));
        byte[] packed = new byte[encoding.getDocPackedLength(dim)];
        switch (encoding) {
            case SINGLE_BIT_QUERY_NIBBLE -> OptimizedScalarQuantizer.packAsBinary(codes, packed);
            case DIBIT_QUERY_NIBBLE -> Lucene104Backports.transposeDibit(codes, packed);
            case PACKED_NIBBLE -> {
                int half = codes.length / 2;
                for (int i = 0; i < half; i++)
                    packed[i] = (byte) ((codes[i] << 4) | (codes[i + half] & 0x0F));
            }
            default -> throw new IllegalArgumentException(encoding.toString());
        }
        return packed;
    }

    private static float javaDot(byte[] query, byte[] block, int offset, int packedBytes, int docBits) {
        return switch (docBits) {
            case 1 -> Int4DotProduct.bit(query, block, offset, packedBytes);
            case 2 -> Int4DotProduct.dibit(query, block, offset, packedBytes);
            default -> Int4DotProduct.nibble(query, block, offset, packedBytes);
        };
    }
}
