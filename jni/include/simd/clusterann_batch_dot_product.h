/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPENSEARCH_KNN_SIMD_CLUSTERANN_BATCH_DOT_PRODUCT_H
#define OPENSEARCH_KNN_SIMD_CLUSTERANN_BATCH_DOT_PRODUCT_H

#include <cstdint>

// ClusterANN block-columnar ADC dot products.
//
// 1-bit / 2-bit documents (transposed bit planes):
//   Query: 4-bit codes transposed with OptimizedScalarQuantizer.transposeHalfByte into 4 bit planes of
//          `stripeBytes = (dim + 7) / 8` bytes each: [plane0 | plane1 | plane2 | plane3].
//   Docs : `docBits` bit planes of `stripeBytes` bytes each, stored back to back per vector
//          (packAsBinary for 1 bit, transposeDibit for 2 bits). `bytesPerCode = docBits * stripeBytes`.
//   dot(query, doc) = sum over qp in [0,4), dp in [0,docBits) of popcount(Q[qp] & D[dp]) << (qp + dp)
//
// 4-bit documents (split-half nibbles, identical to Lucene104 PACKED_NIBBLE):
//   Docs : `bytesPerCode = (dim + 1) / 2` bytes; byte i = code[i] << 4 | code[i + bytesPerCode].
//   Query: unpacked 4-bit codes, one byte per dimension, zero padded to 2 * bytesPerCode.
//   dot(query, doc) = sum_i hi(i) * query[i] + lo(i) * query[bytesPerCode + i]
//
// For k in [0, numOffsets): results[offsets[k]] = dot(query, docs + offsets[k] * bytesPerCode).
// Pure functions, no state, thread-safe. The SIMD variant is selected at compile time by the
// library flavour (KNN_HAVE_AVX512 / KNN_HAVE_AVX2_F16C / KNN_HAVE_ARM_FP16), scalar otherwise.
namespace knn_jni::simd::clusterann {

void bulkDotProduct1bit(const uint8_t* query, const uint8_t* docs, const int32_t* offsets, int32_t numOffsets,
                        float* results, int32_t bytesPerCode);

void bulkDotProduct2bit(const uint8_t* query, const uint8_t* docs, const int32_t* offsets, int32_t numOffsets,
                        float* results, int32_t bytesPerCode);

void bulkDotProduct4bit(const uint8_t* query, const uint8_t* docs, const int32_t* offsets, int32_t numOffsets,
                        float* results, int32_t bytesPerCode);

// Scalar reference implementations (used by tests and as the fallback for the last few vectors).
int64_t scalarDotProduct1bit(const uint8_t* query, const uint8_t* doc, int32_t bytesPerCode);
int64_t scalarDotProduct2bit(const uint8_t* query, const uint8_t* doc, int32_t bytesPerCode);
int64_t scalarDotProduct4bit(const uint8_t* query, const uint8_t* doc, int32_t bytesPerCode);

// Name of the compiled-in bit-plane kernel flavour: "avx512", "avx2", "neon" or "scalar".
const char* kernelName();

// Name of the 4-bit nibble kernel actually selected at runtime: "avx512vnni", "avx512bw", "avx2",
// "neon-dotprod", "neon" or "scalar".
const char* nibbleKernelName();

}  // namespace knn_jni::simd::clusterann

#endif  // OPENSEARCH_KNN_SIMD_CLUSTERANN_BATCH_DOT_PRODUCT_H
