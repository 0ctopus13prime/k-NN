#include <algorithm>
#include <array>
#include <cstddef>
#include <stdint.h>
#include <cmath>

#include "simd_similarity_function_common.cpp"
#include "faiss_score_to_lucene_transform.cpp"
#include "parameter_utils.h"
#include "hamming_distance_calculator.h"

//
// FP16
//

using BulkScoreTransform = void (*)(float*/*scores*/, int32_t/*num scores to transform*/);
using ScoreTransform = float (*)(float/*score*/);

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct DefaultFP16SimilarityFunction final : SimilarityFunction {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) final {

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

    float calculateSimilarity(SimdVectorSearchContext* srchContext, const int32_t internalVectorId) final {
        // Prepare distance calculation
        auto vector = reinterpret_cast<uint8_t*>(srchContext->getVectorPointer(internalVectorId));
        auto func = dynamic_cast<faiss::ScalarQuantizer::SQDistanceComputer*>(srchContext->faissFunction.get());
        knn_jni::util::ParameterCheck::require_non_null(
            func, "Unexpected distance function acquired. Expected SQDistanceComputer, but it was something else");

        // Calculate distance
        const float score = func->query_to_code(vector);

        // Transform score value if it needs to
        return ScoreTransformFunc(score);
    }
};

//
// FP16
//
// 1. Max IP
DefaultFP16SimilarityFunction<FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk, FaissScoreToLuceneScoreTransform::ipToMaxIpTransform> DEFAULT_FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
// 2. L2
DefaultFP16SimilarityFunction<FaissScoreToLuceneScoreTransform::l2TransformBulk, FaissScoreToLuceneScoreTransform::l2Transform> DEFAULT_FP16_L2_SIMIL_FUNC;

//
// Binary
//

struct DefaultBinaryHammingSimilarityFunction final : SimilarityFunction {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) final {
        // Prepare similarity calculation
        auto func = dynamic_cast<HammingDistanceCalculatorInterface*>(srchContext->faissFunction.get());
        knn_jni::util::ParameterCheck::require_non_null(
            func, "Unexpected distance function acquired. Expected HammingDistanceCalculatorInterface, but it was something else");

        int32_t i = 0 ;
        uint8_t* vectorPointers[8];
        for ( ; (i + 8) < numVectors ; i += 8) {
            // Calculate distance
            srchContext->getVectorPointersInBulk(&vectorPointers[0], &internalVectorIds[i], 8);
            func->calculateBatch8(
                &scores[i], &scores[i + 1], &scores[i + 2], &scores[i + 3], &scores[i + 4], &scores[i + 5], &scores[i + 6], &scores[i + 7],
                vectorPointers[0], vectorPointers[1], vectorPointers[2], vectorPointers[3],
                vectorPointers[4], vectorPointers[5], vectorPointers[6], vectorPointers[7]);
        }

        for ( ; (i + 4) < numVectors ; i += 4) {
            // Calculate distance
            srchContext->getVectorPointersInBulk(&vectorPointers[0], &internalVectorIds[i], 4);
            func->calculateBatch4(
                &scores[i], &scores[i + 1], &scores[i + 2], &scores[i + 3],
                vectorPointers[0], vectorPointers[1], vectorPointers[2], vectorPointers[3]);
        }

        while (i < numVectors) {
            auto vector = reinterpret_cast<uint8_t*>(srchContext->getVectorPointer(internalVectorIds[i]));
            scores[i] = func->calculate(vector);
            ++i;
        }

        FaissScoreToLuceneScoreTransform::hammingBitsTransformBulk(scores, numVectors);
    }

    float calculateSimilarity(SimdVectorSearchContext* srchContext, const int32_t internalVectorId) final {
        // Prepare distance calculation
        auto vector = reinterpret_cast<uint8_t*>(srchContext->getVectorPointer(internalVectorId));
        auto func = dynamic_cast<HammingDistanceCalculatorInterface*>(srchContext->faissFunction.get());
        knn_jni::util::ParameterCheck::require_non_null(
            func, "Unexpected distance function acquired. Expected HammingDistanceCalculatorInterface, but it was something else");

        // Calculate distance
        const float score = func->calculate(vector);
        return FaissScoreToLuceneScoreTransform::hammingBitsTransform(score);
    }
};

DefaultBinaryHammingSimilarityFunction DEFAULT_BINARY_HAMMING_SIMIL_FUNC;

#ifndef __NO_SELECT_FUNCTION
SimilarityFunction* SimilarityFunction::selectSimilarityFunction(const NativeSimilarityFunctionType nativeFunctionType) {
    if (nativeFunctionType == NativeSimilarityFunctionType::FP16_MAXIMUM_INNER_PRODUCT) {
        return &DEFAULT_FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::FP16_L2) {
        return &DEFAULT_FP16_L2_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::HAMMING) {
        return &DEFAULT_BINARY_HAMMING_SIMIL_FUNC;
    }

    throw std::runtime_error("Invalid native similarity function type was given, nativeFunctionType="
                             + std::to_string(static_cast<int32_t>(nativeFunctionType)));
}
#endif
