#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#define __ARM_FEATURE_FP16_VECTOR_ARITHMETIC 1
#endif
#include <arm_neon.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <stdint.h>
#include <cmath>
#include <iostream>

#include "simd_similarity_function_common.cpp"
#include "faiss_score_to_lucene_transform.cpp"



//
// FP16
//

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct ArmNeonFP16MaxIP final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {
        // Bulk inner product with 4 batch
        int32_t processedCount = 0;
        constexpr int32_t vecBlock = 4;
        const uint8_t* vectors[vecBlock];
        const auto* queryPtr = (const float*) srchContext->queryVectorSimdAligned;
        const int32_t dim = srchContext->dimension;
        constexpr int32_t dimensionBatch = 8;

        for ( ; (processedCount + vecBlock) <= numVectors ; processedCount += vecBlock) {
            srchContext->getVectorPointersInBulk((uint8_t**)vectors, &internalVectorIds[processedCount], vecBlock);

            // Score accumulator per each vector
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);

            // Batch inner product for 8 values
            int32_t i = 0;
            for (; i + dimensionBatch <= dim; i += dimensionBatch) {
                // Load 8 FP32 query elements
                float32x4_t q0 = vld1q_f32(queryPtr + i);
                float32x4_t q1 = vld1q_f32(queryPtr + i + 4);

                // Load 8 FP16 elements from each target and convert to FP32
                float16x8_t h0 = vld1q_f16((const __fp16 *)(vectors[0] + i * 2));
                float16x8_t h1 = vld1q_f16((const __fp16 *)(vectors[1] + i * 2));
                float16x8_t h2 = vld1q_f16((const __fp16 *)(vectors[2] + i * 2));
                float16x8_t h3 = vld1q_f16((const __fp16 *)(vectors[3] + i * 2));
                float32x4_t d0_lo = vcvt_f32_f16(vget_low_f16(h0));
                float32x4_t d0_hi = vcvt_f32_f16(vget_high_f16(h0));
                float32x4_t d1_lo = vcvt_f32_f16(vget_low_f16(h1));
                float32x4_t d1_hi = vcvt_f32_f16(vget_high_f16(h1));
                float32x4_t d2_lo = vcvt_f32_f16(vget_low_f16(h2));
                float32x4_t d2_hi = vcvt_f32_f16(vget_high_f16(h2));
                float32x4_t d3_lo = vcvt_f32_f16(vget_low_f16(h3));
                float32x4_t d3_hi = vcvt_f32_f16(vget_high_f16(h3));

                // Post-load prefetch: next 8 elements
                // By the time in the next loop,
                if (i + dimensionBatch < dim) {
                    __builtin_prefetch(queryPtr + i + 8);
                    __builtin_prefetch(vectors[0] + (i + 8) * 2);
                    __builtin_prefetch(vectors[1] + (i + 8) * 2);
                    __builtin_prefetch(vectors[2] + (i + 8) * 2);
                    __builtin_prefetch(vectors[3] + (i + 8) * 2);
                }

                // Accumulate FMA
                acc0 = vfmaq_f32(acc0, q0, d0_lo);
                acc0 = vfmaq_f32(acc0, q1, d0_hi);

                acc1 = vfmaq_f32(acc1, q0, d1_lo);
                acc1 = vfmaq_f32(acc1, q1, d1_hi);

                acc2 = vfmaq_f32(acc2, q0, d2_lo);
                acc2 = vfmaq_f32(acc2, q1, d2_hi);

                acc3 = vfmaq_f32(acc3, q0, d3_lo);
                acc3 = vfmaq_f32(acc3, q1, d3_hi);
            }

            // Horizontal sum
            scores[processedCount] = vaddvq_f32(acc0);
            scores[processedCount + 1] = vaddvq_f32(acc1);
            scores[processedCount + 2] = vaddvq_f32(acc2);
            scores[processedCount + 3] = vaddvq_f32(acc3);

            // Scalar tail.
            // For example,
            // if dimension was 66 then this loop will take care of remaining 2 values.
            for (; i < dim; i++) {
                __fp16 h0 = *((const __fp16 *)(vectors[0] + i * 2));
                __fp16 h1 = *((const __fp16 *)(vectors[1] + i * 2));
                __fp16 h2 = *((const __fp16 *)(vectors[2] + i * 2));
                __fp16 h3 = *((const __fp16 *)(vectors[3] + i * 2));
                const float qv = queryPtr[i];
                scores[processedCount] += qv * (float)h0;
                scores[processedCount + 1] += qv * (float)h1;
                scores[processedCount + 2] += qv * (float)h2;
                scores[processedCount + 3] += qv * (float)h3;
            }
        }

        // Tail loop for remaining vectors
        for (; processedCount < numVectors; ++processedCount) {
            const auto* vecPtr = (const __fp16*) srchContext->getVectorPointer(internalVectorIds[processedCount]);
            float32x4_t acc = vdupq_n_f32(0.0f);
            int32_t i = 0;
            for (; i <= dim - dimensionBatch; i += dimensionBatch) {
                float32x4_t q0 = vld1q_f32(queryPtr + i);
                float32x4_t q1 = vld1q_f32(queryPtr + i + 4);
                float16x8_t h0 = vld1q_f16((const __fp16 *)(vecPtr + i));
                float32x4_t d0_lo = vcvt_f32_f16(vget_low_f16(h0));
                float32x4_t d0_hi = vcvt_f32_f16(vget_high_f16(h0));
                acc = vfmaq_f32(acc, q0, d0_lo);
                acc = vfmaq_f32(acc, q1, d0_hi);
            }

            float finalSum = vaddvq_f32(acc);
            // Scalar tail for dimensions not divisible by 4
            for (; i < dim; ++i) {
                finalSum += queryPtr[i] * (float)vecPtr[i];
            }
            scores[processedCount] = finalSum;
        }

        BulkScoreTransformFunc(scores, numVectors);
    }
};

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct ArmNeonFP16L2 final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {
        // Prepare similarity calculation
        auto func = dynamic_cast<faiss::ScalarQuantizer::SQDistanceComputer*>(srchContext->faissFunction.get());
        knn_jni::util::ParameterCheck::require_non_null(
            func, "Unexpected distance function acquired. Expected SQDistanceComputer, but it was something else");

        for (int32_t i = 0 ; i < numVectors ; ++i) {
            // Calculate distance
            auto vector = reinterpret_cast<uint8_t*>(srchContext->getVectorPointer(internalVectorIds[i]));
            scores[i] = func->query_to_code(vector);
        }

        // Transform score values if it needs to
        BulkScoreTransformFunc(scores, numVectors);
    }
};



//
// BBQ (ADC: 4-bit query x 1-bit data) - Naive C++ implementation (Phase-1, no SIMD optimization)
//
// The query is 4-bit quantized and transposed into 4 bit planes.
// Each bit plane has `binaryCodeBytes` bytes. The int4BitDotProduct computes:
//   sum over i in [0..3]: popcount(q[i*size..(i+1)*size] AND d[0..size]) << i
// This gives a weighted dot product where each 4-bit query nibble contributes 0..15.
//

static constexpr float FOUR_BIT_SCALE = 1.0f / 15.0f;

// Compute int4BitDotProduct: 4-bit transposed query vs 1-bit binary code
// q has 4 * binaryCodeBytes bytes (4 bit planes), d has binaryCodeBytes bytes
static inline int64_t int4BitDotProduct(const uint8_t* q, const uint8_t* d, const int32_t binaryCodeBytes) {
    int64_t result = 0;
    for (int32_t bitPlane = 0 ; bitPlane < 4 ; ++bitPlane) {
        const auto* qPlane = reinterpret_cast<const uint64_t*>(q + bitPlane * binaryCodeBytes);
        const auto* dPtr = reinterpret_cast<const uint64_t*>(d);
        const int32_t words = binaryCodeBytes >> 3;

        int64_t subResult = 0;
        for (int32_t w = 0 ; w < words ; ++w) {
            subResult += __builtin_popcountll(qPlane[w] & dPtr[w]);
        }

        // Handle remaining bytes (if binaryCodeBytes is not a multiple of 8)
        const int32_t remainStart = words * 8;
        for (int32_t r = remainStart ; r < binaryCodeBytes ; ++r) {
            subResult += __builtin_popcount((q[bitPlane * binaryCodeBytes + r] & d[r]) & 0xFF);
        }

        result += subResult << bitPlane;
    }
    return result;
}

struct ArmNeonBBQIP final : SimilarityFunction {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {
        const auto* queryPtr = reinterpret_cast<const uint8_t*>(srchContext->queryVectorSimdAligned);
        const int32_t dim = srchContext->dimension;
        // Binary code bytes for 1-bit data vectors: discretize(dim, 64) / 8
        const int32_t binaryCodeBytes = ((dim + 63) / 64) * 64 / 8;

        // Read query correction factors from tmpBuffer
        const auto* queryCorrectionPtr = reinterpret_cast<const float*>(srchContext->tmpBuffer.data());
        const float ay = queryCorrectionPtr[0];  // lowerInterval
        // ADC: ly includes FOUR_BIT_SCALE for 4-bit query quantization
        const float ly = (queryCorrectionPtr[1] - queryCorrectionPtr[0]) * FOUR_BIT_SCALE;
        const float queryAdditional = queryCorrectionPtr[2];  // additionalCorrection
        const float y1 = static_cast<float>(*reinterpret_cast<const int32_t*>(&queryCorrectionPtr[3]));  // quantizedComponentSum
        const float centroidDp = queryCorrectionPtr[4];

        // Process vectors in batches of 4
        int32_t processedCount = 0;
        constexpr int32_t vecBlock = 4;
        uint8_t* vectors[vecBlock];

        for ( ; (processedCount + vecBlock) <= numVectors ; processedCount += vecBlock) {
            srchContext->getVectorPointersInBulk(vectors, &internalVectorIds[processedCount], vecBlock);

            for (int32_t v = 0 ; v < vecBlock ; ++v) {
                // Compute ADC distance: int4BitDotProduct(4-bit query, 1-bit data)
                const float qcDist = static_cast<float>(
                    int4BitDotProduct(queryPtr, vectors[v], binaryCodeBytes));

                // Read data vector's correction factors
                // Layout: [binary code | lowerInterval(float) | upperInterval(float) | additionalCorrection(float) | quantizedComponentSum(unsigned short)]
                const auto* dataCorrectionPtr = reinterpret_cast<const float*>(vectors[v] + binaryCodeBytes);
                const float ax = dataCorrectionPtr[0];  // lowerInterval
                const float lx = dataCorrectionPtr[1] - dataCorrectionPtr[0];  // upperInterval - lowerInterval
                const float additional = dataCorrectionPtr[2];  // additionalCorrection
                const float x1 = static_cast<float>(*reinterpret_cast<const uint16_t*>(vectors[v] + binaryCodeBytes + 12));  // quantizedComponentSum

                // BBQ ADC scoring formula
                scores[processedCount + v] = ax * ay * dim
                                           + ay * lx * x1
                                           + ax * ly * y1
                                           + lx * ly * qcDist
                                           + queryAdditional
                                           + additional
                                           - centroidDp;
            }
        }

        // Tail: remaining vectors
        for ( ; processedCount < numVectors ; ++processedCount) {
            const auto* dataVec = srchContext->getVectorPointer(internalVectorIds[processedCount]);

            const float qcDist = static_cast<float>(
                int4BitDotProduct(queryPtr, dataVec, binaryCodeBytes));

            const auto* dataCorrectionPtr = reinterpret_cast<const float*>(dataVec + binaryCodeBytes);
            const float ax = dataCorrectionPtr[0];
            const float lx = dataCorrectionPtr[1] - dataCorrectionPtr[0];
            const float additional = dataCorrectionPtr[2];
            const float x1 = static_cast<float>(*reinterpret_cast<const uint16_t*>(dataVec + binaryCodeBytes + 12));

            scores[processedCount] = ax * ay * dim
                                   + ay * lx * x1
                                   + ax * ly * y1
                                   + lx * ly * qcDist
                                   + queryAdditional
                                   + additional
                                   - centroidDp;
        }

        // Transform to Max IP score (same as VectorUtil.scaleMaxInnerProductScore)
        FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk(scores, numVectors);
    }

    float calculateSimilarity(SimdVectorSearchContext* srchContext, const int32_t internalVectorId) {
        const auto* queryPtr = reinterpret_cast<const uint8_t*>(srchContext->queryVectorSimdAligned);
        const int32_t dim = srchContext->dimension;
        const int32_t binaryCodeBytes = ((dim + 63) / 64) * 64 / 8;

        // Read query correction factors from tmpBuffer
        const auto* queryCorrectionPtr = reinterpret_cast<const float*>(srchContext->tmpBuffer.data());
        const float ay = queryCorrectionPtr[0];
        const float ly = (queryCorrectionPtr[1] - queryCorrectionPtr[0]) * FOUR_BIT_SCALE;
        const float queryAdditional = queryCorrectionPtr[2];
        const float y1 = static_cast<float>(*reinterpret_cast<const int32_t*>(&queryCorrectionPtr[3]));
        const float centroidDp = queryCorrectionPtr[4];

        const auto* dataVec = srchContext->getVectorPointer(internalVectorId);

        const float qcDist = static_cast<float>(
            int4BitDotProduct(queryPtr, dataVec, binaryCodeBytes));

        const auto* dataCorrectionPtr = reinterpret_cast<const float*>(dataVec + binaryCodeBytes);
        const float ax = dataCorrectionPtr[0];
        const float lx = dataCorrectionPtr[1] - dataCorrectionPtr[0];
        const float additional = dataCorrectionPtr[2];
        const float x1 = static_cast<float>(*reinterpret_cast<const uint16_t*>(dataVec + binaryCodeBytes + 12));

        const float score = ax * ay * dim
                          + ay * lx * x1
                          + ax * ly * y1
                          + lx * ly * qcDist
                          + queryAdditional
                          + additional
                          - centroidDp;

        return FaissScoreToLuceneScoreTransform::ipToMaxIpTransform(score);
    }
};



//
// FP16
//
// 1. Max IP
ArmNeonFP16MaxIP<FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk, FaissScoreToLuceneScoreTransform::ipToMaxIpTransform> FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
// 2. L2
ArmNeonFP16L2<FaissScoreToLuceneScoreTransform::l2TransformBulk, FaissScoreToLuceneScoreTransform::l2Transform> FP16_L2_SIMIL_FUNC;
// 3. BBQ IP
ArmNeonBBQIP BBQ_IP_SIMIL_FUNC;

#ifndef __NO_SELECT_FUNCTION
SimilarityFunction* SimilarityFunction::selectSimilarityFunction(const NativeSimilarityFunctionType nativeFunctionType) {
    if (nativeFunctionType == NativeSimilarityFunctionType::FP16_MAXIMUM_INNER_PRODUCT) {
        return &FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::FP16_L2) {
        return &FP16_L2_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::BBQ_IP) {
        return &BBQ_IP_SIMIL_FUNC;
    }

    throw std::runtime_error("Invalid native similarity function type was given, nativeFunctionType="
                             + std::to_string(static_cast<int32_t>(nativeFunctionType)));
}
#endif
