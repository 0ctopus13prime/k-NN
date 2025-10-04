#include <algorithm>
#include <array>
#include <cstddef>
#include <stdint.h>
#include <arm_neon.h>
#include <cmath>

#include "simd_similarity_function_common.cpp"
#include "jni_util.h"
#include "faiss/impl/ScalarQuantizer.h"

//
// FP16 Inner product
//

std::array<faiss::ScalarQuantizer::SQDistanceComputer*, knn_jni::MAX_DIMENSION> FAISS_FP16_IP_FUNC_TABLE = []{
    std::array<faiss::ScalarQuantizer::SQDistanceComputer*, knn_jni::MAX_DIMENSION> functionTable {};
    for (int32_t d = 1 ; d < knn_jni::MAX_DIMENSION ; ++d) {
        functionTable[d] =  faiss::ScalarQuantizer {d, faiss::ScalarQuantizer::QuantizerType::QT_fp16}
                            .get_distance_computer(faiss::MetricType::METRIC_INNER_PRODUCT);
    }
    return functionTable;
}();

float SimilarityFunction::calculateSimilarityInBulk(int32_t* internalVectorIds,
                                                    int32_t numVectors,
                                                    float* scores) {
    for (int32_t i = 0 ; i < numVectors ; ++i) {
        scores[i] = calculateSimilarity(internalVectorId[i]);
    }
}

float SimilarityFunction::calculateSimilarity(SimdVectorSearchContext* srchContext, int32_t internalVectorId) {
    // Prepare distance calculation
    const int32_t dimension = srchContext->oneVectorByteSize / 2;
    auto vector = reinterpret_cast<uint8_t*>(srchContext->getVectorOffsets(internalVectorId));
    auto func = FAISS_FP16_IP_FUNC_TABLE[dimension];

    // Set query
    func->q = reinterpret_cast<float*>(srchContext->queryVectorSimdAligned);

    // Calculate distance
    float ipScore = FAISS_FP16_IP_FUNC_TABLE[dimension]->query_to_code(vector);
    return ipScore < 0 ? (1 / (1 - ipScore)) : (ipScore + 1);
}
