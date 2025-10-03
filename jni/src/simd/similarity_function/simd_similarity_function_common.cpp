#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "simd/similarity_function/similarity_function.h"

using knn_jni::simd::similarity_function::SimdVectorSearchContext;
using knn_jni::simd::similarity_function::SimilarityFunction;

SimdVectorSearchContext::~SimdVectorSearchContext() {
    if (queryVectorSimdAligned) {
        // Use free() because we used aligned_alloc
        std::free(queryVectorSimdAligned);
    }
}

thread_local SimdVectorSearchContext THREAD_LOCAL_SIMD_VEC_SRCH_CTX {};

SimdVectorSearchContext* SimilarityFunction::saveSearchContext(
           uint8_t* queryPtr,
           int32_t queryByteSize,
           int64_t* mmapAddressAndSize,
           int32_t numAddressAndSize,
           int32_t nativeFunctionTypeOrd) {
    // Set query vector
    if (THREAD_LOCAL_SIMD_VEC_SRCH_CTX.oneVectorByteSize < queryByteSize) {
        // We need to allocate or re-allocate the space.
        // Allocating 64 bytes aligned memory.
        // Since 16000 dimension is the maximum, therefore at most 62.6KB will be allocated per thread.
        void* alignedPtr = std::aligned_alloc(64, queryByteSize);
        if (!alignedPtr) {
            throw std::runtime_error("Failed to allocate space for SIMD aligned query vector with size=" + std::to_string(queryByteSize));
        }

        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorSimdAligned = alignedPtr;
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.oneVectorByteSize = queryByteSize;
    }

    // Copy query
    std::memcpy(THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorSimdAligned, queryPtr, queryByteSize);

    // Set mmap pages
    THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPages.clear();
    THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPageSizes.clear();
    for (int32_t i = 0 ; i < numAddressAndSize ; i += 2) {
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPages.emplace_back(reinterpret_cast<void*>(mmapAddressAndSize[i]));
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPageSizes.emplace_back(mmapAddressAndSize[i + 1]);
    }

    // Set function type
    THREAD_LOCAL_SIMD_VEC_SRCH_CTX.nativeFunctionTypeOrd = nativeFunctionTypeOrd;

    // Return thread_local object
    return &THREAD_LOCAL_SIMD_VEC_SRCH_CTX;
}
