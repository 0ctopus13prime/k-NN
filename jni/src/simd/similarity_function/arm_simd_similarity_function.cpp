#include <algorithm>
#include <array>
#include <cstddef>
#include <stdint.h>
#include <arm_neon.h>
#include <cmath>
#include <iostream>

#include "simd_similarity_function_common.cpp"
#include "jni_util.h"
#include "faiss/impl/ScalarQuantizer.h"

//
// FP16 Inner product
//

std::array<faiss::ScalarQuantizer::SQDistanceComputer*, knn_jni::MAX_DIMENSION> FAISS_FP16_IP_FUNC_TABLE = []{
    std::array<faiss::ScalarQuantizer::SQDistanceComputer*, knn_jni::MAX_DIMENSION> functionTable {};
    for (size_t d = 1 ; d < knn_jni::MAX_DIMENSION ; ++d) {
        functionTable[d] =  faiss::ScalarQuantizer {d, faiss::ScalarQuantizer::QuantizerType::QT_fp16}
                            .get_distance_computer(faiss::MetricType::METRIC_INNER_PRODUCT);
    }
    return functionTable;
}();

struct FP16InnerProductSimilarityFunction final : SimilarityFunction {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   int32_t numVectors) final {
        // Get search context
        uint8_t* vectors[8];
        const int32_t dimension = srchContext->dimension;

        // Bulk SIMD with batch size 8
        int32_t i = 0;
        for ( ; (i + 8) <= numVectors ; i += 8, scores += 8) {
            srchContext->getVectorOffsetsInBulk(vectors, internalVectorIds + i, 8);
            batchInnerProduct8FP16Targets(
                reinterpret_cast<float*>(srchContext->queryVectorSimdAligned),
                vectors[0], vectors[1], vectors[2], vectors[3],
                vectors[4], vectors[5], vectors[6], vectors[7],
                dimension,
                scores[0], scores[1], scores[2], scores[3],
                scores[4], scores[5], scores[6], scores[7]);
        }

        // Bulk SIMD with batch size 4
        for ( ; (i + 4) <= numVectors ; i += 4, scores += 4) {
            srchContext->getVectorOffsetsInBulk(vectors, internalVectorIds + i, 4);
            batchInnerProduct4FP16Targets(
                reinterpret_cast<float*>(srchContext->queryVectorSimdAligned),
                vectors[0], vectors[1], vectors[2], vectors[3],
                dimension,
                scores[0], scores[1], scores[2], scores[3]);
        }

        // Transform score value to MAX_IP
        for (int32_t j = 0 ; j < i ; ++j) {
            scores[j] = scores[j] < 0 ? (1 / (1 - scores[j])) : (scores[j] + 1);
        }

        // Handle remaining vectors
        while (i < numVectors) {
            *scores++ = calculateSimilarity(srchContext, internalVectorIds[i++]);
        }
    }

    float calculateSimilarity(SimdVectorSearchContext* srchContext, int32_t internalVectorId) final {
        // Prepare distance calculation
        const int32_t dimension = srchContext->dimension;
        auto vector = reinterpret_cast<uint8_t*>(srchContext->getVectorOffsets(internalVectorId));
        auto func = FAISS_FP16_IP_FUNC_TABLE[dimension];

        // Set query
        func->q = reinterpret_cast<float*>(srchContext->queryVectorSimdAligned);

        std::cout << "calculateSimilarity: dimension=" << dimension
                  << ", internalVectorId=" << internalVectorId
                  << ", vector[0]=" << int32_t(vector[0])
                  << ", q[0]=" << func->q[0]
                  << std::endl;

        // Calculate distance
        const float ipScore = FAISS_FP16_IP_FUNC_TABLE[dimension]->query_to_code(vector);

        std::cout << "calculateSimilarity: ipScore=" << ipScore
                  << std::endl;

        return ipScore < 0 ? (1 / (1 - ipScore)) : (ipScore + 1);
    }

    static void batchInnerProduct8FP16Targets(
        const float *query,
        const uint8_t *d0, const uint8_t *d1,
        const uint8_t *d2, const uint8_t *d3,
        const uint8_t *d4, const uint8_t *d5,
        const uint8_t *d6, const uint8_t *d7,
        size_t dim,
        float& score0, float& score1, float& score2, float& score3,
        float& score4, float& score5, float& score6, float& score7) {
        // Accumulators (8 targets * 4 elements/register = 32 registers total)
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);
        float32x4_t acc4 = vdupq_n_f32(0.0f);
        float32x4_t acc5 = vdupq_n_f32(0.0f);
        float32x4_t acc6 = vdupq_n_f32(0.0f);
        float32x4_t acc7 = vdupq_n_f32(0.0f);

        const size_t dim_aligned = dim & ~7; // Process 8 elements at a time
        size_t i = 0;

        for (; i < dim_aligned; i += 8) {
            // --- 1. Load Query (8 FP32 values) ---
            float32x4_t q0 = vld1q_f32(query + i);
            float32x4_t q1 = vld1q_f32(query + i + 4);

            // --- 2. Load and Convert Target Vectors (8 targets * 8 FP16 values each) ---

            // Target 0: Load 8 FP16, split into lo/hi, and convert to 2x FP32x4
            float16x8_t h0 = vld1q_f16((const __fp16 *)(d0 + i * 2));
            float32x4_t d0_lo = vcvt_f32_f16(vget_low_f16(h0));
            float32x4_t d0_hi = vcvt_f32_f16(vget_high_f16(h0));

            // Target 1
            float16x8_t h1 = vld1q_f16((const __fp16 *)(d1 + i * 2));
            float32x4_t d1_lo = vcvt_f32_f16(vget_low_f16(h1));
            float32x4_t d1_hi = vcvt_f32_f16(vget_high_f16(h1));

            // Target 2
            float16x8_t h2 = vld1q_f16((const __fp16 *)(d2 + i * 2));
            float32x4_t d2_lo = vcvt_f32_f16(vget_low_f16(h2));
            float32x4_t d2_hi = vcvt_f32_f16(vget_high_f16(h2));

            // Target 3
            float16x8_t h3 = vld1q_f16((const __fp16 *)(d3 + i * 2));
            float32x4_t d3_lo = vcvt_f32_f16(vget_low_f16(h3));
            float32x4_t d3_hi = vcvt_f32_f16(vget_high_f16(h3));

            // Target 4
            float16x8_t h4 = vld1q_f16((const __fp16 *)(d4 + i * 2));
            float32x4_t d4_lo = vcvt_f32_f16(vget_low_f16(h4));
            float32x4_t d4_hi = vcvt_f32_f16(vget_high_f16(h4));

            // Target 5
            float16x8_t h5 = vld1q_f16((const __fp16 *)(d5 + i * 2));
            float32x4_t d5_lo = vcvt_f32_f16(vget_low_f16(h5));
            float32x4_t d5_hi = vcvt_f32_f16(vget_high_f16(h5));

            // Target 6
            float16x8_t h6 = vld1q_f16((const __fp16 *)(d6 + i * 2));
            float32x4_t d6_lo = vcvt_f32_f16(vget_low_f16(h6));
            float32x4_t d6_hi = vcvt_f32_f16(vget_high_f16(h6));

            // Target 7
            float16x8_t h7 = vld1q_f16((const __fp16 *)(d7 + i * 2));
            float32x4_t d7_lo = vcvt_f32_f16(vget_low_f16(h7));
            float32x4_t d7_hi = vcvt_f32_f16(vget_high_f16(h7));

            // --- 3. Accumulate (16 FMA instructions per loop) ---
            // (Uses FMA for superior throughput and reduced rounding error)
            acc0 = vfmaq_f32(acc0, q0, d0_lo); acc0 = vfmaq_f32(acc0, q1, d0_hi);
            acc1 = vfmaq_f32(acc1, q0, d1_lo); acc1 = vfmaq_f32(acc1, q1, d1_hi);
            acc2 = vfmaq_f32(acc2, q0, d2_lo); acc2 = vfmaq_f32(acc2, q1, d2_hi);
            acc3 = vfmaq_f32(acc3, q0, d3_lo); acc3 = vfmaq_f32(acc3, q1, d3_hi);
            acc4 = vfmaq_f32(acc4, q0, d4_lo); acc4 = vfmaq_f32(acc4, q1, d4_hi);
            acc5 = vfmaq_f32(acc5, q0, d5_lo); acc5 = vfmaq_f32(acc5, q1, d5_hi);
            acc6 = vfmaq_f32(acc6, q0, d6_lo); acc6 = vfmaq_f32(acc6, q1, d6_hi);
            acc7 = vfmaq_f32(acc7, q0, d7_lo); acc7 = vfmaq_f32(acc7, q1, d7_hi);

            // Optimization: Prefetching moved to the load/convert section or handled by the compiler
        }

        // --- 4. Horizontal Sum ---
        score0 = vaddvq_f32(acc0);
        score1 = vaddvq_f32(acc1);
        score2 = vaddvq_f32(acc2);
        score3 = vaddvq_f32(acc3);
        score4 = vaddvq_f32(acc4);
        score5 = vaddvq_f32(acc5);
        score6 = vaddvq_f32(acc6);
        score7 = vaddvq_f32(acc7);

        // --- 5. Scalar Tail (using remaining 'i' value) ---
        for (; i < dim; i++) {
            const float qv = query[i];
            score0 += qv * (float)*((const __fp16 *)(d0 + i * 2));
            score1 += qv * (float)*((const __fp16 *)(d1 + i * 2));
            score2 += qv * (float)*((const __fp16 *)(d2 + i * 2));
            score3 += qv * (float)*((const __fp16 *)(d3 + i * 2));
            score4 += qv * (float)*((const __fp16 *)(d4 + i * 2));
            score5 += qv * (float)*((const __fp16 *)(d5 + i * 2));
            score6 += qv * (float)*((const __fp16 *)(d6 + i * 2));
            score7 += qv * (float)*((const __fp16 *)(d7 + i * 2));
        }
    }

    void batchInnerProduct4FP16Targets(
        const float *query,
        const uint8_t *d0,
        const uint8_t *d1,
        const uint8_t *d2,
        const uint8_t *d3,
        size_t dim,
        float& score0,
        float& score1,
        float& score2,
        float& score3) {

        // Score accumulator per each vector
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);

        // Batch inner product for 8 values
        size_t i = 0;
        for (; i + 8 <= dim; i += 8) {
            // Load 8 FP32 query elements
            float32x4_t q0 = vld1q_f32(query + i);
            float32x4_t q1 = vld1q_f32(query + i + 4);

            // Load 8 FP16 elements from each target and convert to FP32
            float16x8_t h0 = vld1q_f16((const __fp16 *)(d0 + i * 2));
            float16x8_t h1 = vld1q_f16((const __fp16 *)(d1 + i * 2));
            float16x8_t h2 = vld1q_f16((const __fp16 *)(d2 + i * 2));
            float16x8_t h3 = vld1q_f16((const __fp16 *)(d3 + i * 2));
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
            if (i + 8 < dim) {
                __builtin_prefetch(query + i + 8);
                __builtin_prefetch(d0 + (i + 8) * 2);
                __builtin_prefetch(d1 + (i + 8) * 2);
                __builtin_prefetch(d2 + (i + 8) * 2);
                __builtin_prefetch(d3 + (i + 8) * 2);
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
        score0 = vaddvq_f32(acc0);
        score1 = vaddvq_f32(acc1);
        score2 = vaddvq_f32(acc2);
        score3 = vaddvq_f32(acc3);

        // Scalar tail.
        // For example,
        // if dimension was 66 then this loop will take care of remaining 2 values.
        for (; i < dim; i++) {
            __fp16 h0 = *((const __fp16 *)(d0 + i * 2));
            __fp16 h1 = *((const __fp16 *)(d1 + i * 2));
            __fp16 h2 = *((const __fp16 *)(d2 + i * 2));
            __fp16 h3 = *((const __fp16 *)(d3 + i * 2));
            const float qv = query[i];
            score0 += qv * (float)h0;
            score1 += qv * (float)h1;
            score2 += qv * (float)h2;
            score3 += qv * (float)h3;
        }
    }
};

FP16InnerProductSimilarityFunction FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;

//
// FP16 L2
//

struct FP16L2SimilarityFunction final : SimilarityFunction {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   int32_t numVectors) final {
        // TODO(KDY)
   }

    float calculateSimilarity(SimdVectorSearchContext* srchContext, int32_t internalVectorId) final {
        // TODO(KDY)
        return 0;
    }
};

FP16L2SimilarityFunction FP16_L2_SIMIL_FUNC;

SimilarityFunction* SimilarityFunction::selectSimilarityFunction(NativeSimilarityFunctionType nativeFunctionType) {
    if (nativeFunctionType == NativeSimilarityFunctionType::FP16_MAXIMUM_INNER_PRODUCT) {
        return &FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::FP16_L2) {
        return &FP16_L2_SIMIL_FUNC;
    }

    throw std::runtime_error("Invalid native similarity function type was given, nativeFunctionType="
                             + std::to_string(static_cast<int32_t>(nativeFunctionType)));
}
