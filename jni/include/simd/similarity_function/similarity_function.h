#ifndef OPENSEARCH_KNN_SIMD_SIMILARITY_FUNCTION_H
#define OPENSEARCH_KNN_SIMD_SIMILARITY_FUNCTION_H

#include <cstdint>
#include <vector>

namespace knn_jni::simd::similarity_function {
    struct SimdVectorSearchContext {
        void* queryVectorSimdAligned = nullptr;
        int32_t oneVectorByteSize = 0;
        std::vector<void*> mmapPages;
        std::vector<int64_t> mmapPageSizes;
        int32_t nativeFunctionTypeOrd = -1;

        ~SimdVectorSearchContext();
    };

    struct SimilarityFunction {
        static SimdVectorSearchContext* saveSearchContext(
                   uint8_t* queryPtr,
                   int32_t queryByteSize,
                   int64_t* mmapAddressAndSize,
                   int32_t numAddressAndSize,
                   int32_t nativeFunctionTypeOrd);

        static float calculateSimilarityInBulk(int32_t** internalVectorIds,
                                               int32_t numVectors,
                                               float* scores);

        static float calculateSimilarity(int32_t internalVectorId);

      private:
        SimilarityFunction() = delete;
    };
}

#endif  // OPENSEARCH_KNN_SIMD_SIMILARITY_FUNCTION_H
