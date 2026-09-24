/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar;

import org.opensearch.knn.jni.SimdVectorComputeService;

/**
 * Native SIMD counterpart of {@link Int4DotProduct} for one block of codes at a time.
 *
 * <p>The kernel lives in {@code jni/src/simd/similarity_function/clusterann_batch_dot_product.cpp} and reads the same
 * layouts {@link Int4DotProduct} does, so the two are interchangeable per call: same query bytes, same code block, same
 * integer result. It scores several documents per SIMD pass (VNNI {@code vpdpbusd} or NEON {@code udot} for nibbles,
 * AND + popcount for bit planes) and shares the query loads across them, which is where it gets ahead of the scalar
 * Java loops.
 *
 * <p>Whether the native path is used is decided once per JVM: the SIMD library must load, and
 * {@code -Dclusterann.native.dot} must not be {@code false}. Callers check {@link #AVAILABLE} and fall back to
 * {@link Int4DotProduct} otherwise, so a box without {@code jni/build/release} (unit tests, dev laptops) keeps
 * working with identical scores.
 *
 * <p>Stateless and thread-safe: everything is passed per call, nothing is cached natively.
 */
public final class NativeInt4DotProduct {

    /** True when the native kernel is loadable and not disabled. Evaluated once. */
    public static final boolean AVAILABLE = resolve();

    private NativeInt4DotProduct() {}

    private static boolean resolve() {
        if (!Boolean.parseBoolean(System.getProperty("clusterann.native.dot", "true"))) {
            return false;
        }
        try {
            // Class init loads the SIMD library; a missing library surfaces as UnsatisfiedLinkError /
            // ExceptionInInitializerError. Calling the tiny diagnostic native also proves the symbol is present.
            SimdVectorComputeService.bulkQuantizedDotProductKernel();
            return true;
        } catch (Throwable t) {
            return false;
        }
    }

    /** Kernel flavours selected natively, e.g. {@code neon/neon-dotprod}, or a reason when unavailable. */
    public static String kernelName() {
        try {
            return SimdVectorComputeService.bulkQuantizedDotProductKernel();
        } catch (Throwable t) {
            return "unavailable (" + t.getClass().getSimpleName() + ")";
        }
    }

    /**
     * {@code results[positions[k]] = dot(query, codes at positions[k] * packedBytes)} for {@code k in [0, count)}.
     *
     * @param query        the query as {@link Int4DotProduct} expects it for {@code docBits}: four transposed planes
     *                     for 1 and 2 bits, unpacked codes for 4 bits
     * @param codes        one block of packed codes, {@code packedBytes} per vector
     * @param positions    block positions to score
     * @param count        number of valid entries in {@code positions}
     * @param results      receives the raw integer dot products, indexed by position; other slots untouched
     * @param packedBytes  packed bytes per vector
     * @param docBits      1, 2 or 4
     */
    public static void bulk(byte[] query, byte[] codes, int[] positions, int count, float[] results, int packedBytes, int docBits) {
        SimdVectorComputeService.bulkQuantizedDotProduct(query, codes, positions, count, results, packedBytes, docBits);
    }
}
