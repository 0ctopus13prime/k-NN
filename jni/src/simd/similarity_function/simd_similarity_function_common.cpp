#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <iostream>

#include "simd/similarity_function/similarity_function.h"
#include "faiss/impl/ScalarQuantizer.h"
#include "jni_util.h"

using knn_jni::simd::similarity_function::SimdVectorSearchContext;
using knn_jni::simd::similarity_function::SimilarityFunction;

//
// SimdVectorSearchContext
//
void SimdVectorSearchContext::getVectorOffsetsInBulk(uint8_t* vector[], int32_t* internalVectorIds, int32_t numVectors) {
    std::cout << "getVectorOffsetsInBulk: mmapPages.size()=" << mmapPages.size()
              << ", numVectors=" << numVectors
              << ", oneVectorByteSize=" << oneVectorByteSize
              << std::endl;
    if (mmapPages.size() == 1) {
        // Fast case, there's only one mmap area.
        auto base = mmapPages[0];
        for (int32_t i = 0 ; i < numVectors ; ++i) {
            vector[i] = reinterpret_cast<uint8_t*>(base) + (oneVectorByteSize * internalVectorIds[i]);
        }
        return;
    }

    // TODO(KDY)

    if (mmapPages.empty() == false) {
    }

    throw std::runtime_error("Search context has not been initialized, mmapPages was empty.");
}

uint8_t* SimdVectorSearchContext::getVectorOffsets(int32_t internalVectorId) {
    std::cout << "getVectorOffsets: mmapPages.size()=" << mmapPages.size()
              << ", internalVectorId=" << internalVectorId
              << ", oneVectorByteSize=" << oneVectorByteSize
              << std::endl;
    if (mmapPages.size() == 1) {
        // Fast case, there's only one mmap area.
        return reinterpret_cast<uint8_t*>(mmapPages[0]) + (oneVectorByteSize * internalVectorId);
    }

    // TODO(KDY)

    if (mmapPages.empty() == false) {
    }

    throw std::runtime_error("Search context has not been initialized, mmapPages was empty.");
}

SimdVectorSearchContext::~SimdVectorSearchContext() {
    if (queryVectorSimdAligned) {
        // Use free() because we used aligned_alloc
        std::free(queryVectorSimdAligned);
    }
}

// Thread static local SimdVectorSearchContext
thread_local SimdVectorSearchContext THREAD_LOCAL_SIMD_VEC_SRCH_CTX {};



//
// SimilarityFunction
//
SimdVectorSearchContext* SimilarityFunction::saveSearchContext(
           uint8_t* queryPtr,
           int32_t queryByteSize,
           int32_t dimension,
           int64_t* mmapAddressAndSize,
           int32_t numAddressAndSize,
           int32_t nativeFunctionTypeOrd) {
    // Set function
    if (nativeFunctionTypeOrd == static_cast<int32_t>(NativeSimilarityFunctionType::FP16_MAXIMUM_INNER_PRODUCT)) {
        std::cout << "saveSearchContext: funcType=FP16_MAXIMUM_INNER_PRODUCT" << std::endl;
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.similarityFunction = selectSimilarityFunction(
            NativeSimilarityFunctionType::FP16_MAXIMUM_INNER_PRODUCT);
         THREAD_LOCAL_SIMD_VEC_SRCH_CTX.oneVectorByteSize = 2 * dimension;
    } else if (nativeFunctionTypeOrd == static_cast<int32_t>(NativeSimilarityFunctionType::FP16_L2)) {
        std::cout << "saveSearchContext: funcType=FP16_L2" << std::endl;
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.similarityFunction = selectSimilarityFunction(
            NativeSimilarityFunctionType::FP16_L2);
         THREAD_LOCAL_SIMD_VEC_SRCH_CTX.oneVectorByteSize = 2 * dimension;
    } else {
        throw std::runtime_error("Invalid native similarity function type was given, nativeFunctionTypeOrd="
                                 + std::to_string(nativeFunctionTypeOrd));
    }

    THREAD_LOCAL_SIMD_VEC_SRCH_CTX.nativeFunctionTypeOrd = nativeFunctionTypeOrd;
    std::cout << "saveSearchContext: nativeFunctionTypeOrd=" << nativeFunctionTypeOrd
              << ", queryVectorByteSize=" << THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorByteSize
              << ", queryByteSize=" << queryByteSize
              << std::endl;

    // Set dimension
    THREAD_LOCAL_SIMD_VEC_SRCH_CTX.dimension = dimension;

    // Set query vector
    if (THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorByteSize < queryByteSize) {
        // We need to allocate or re-allocate the space.
        // Allocating 64 bytes aligned memory.
        // Since 16000 dimension is the maximum, therefore at most 62.6KB will be allocated per thread.
        const auto roundedUpQueryByteSize = ((queryByteSize + 63) / 64) * 64;
        void* alignedPtr = std::aligned_alloc(64, roundedUpQueryByteSize);
        if (!alignedPtr) {
            throw std::runtime_error("Failed to allocate space for SIMD aligned query vector with size=" + std::to_string(queryByteSize));
        }

        std::cout << "saveSearchContext: allocating [" << queryByteSize << "] bytes for a query"
                  << std::endl;

        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorSimdAligned = alignedPtr;
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorByteSize = queryByteSize;
    }

    // Copy query
    std::memcpy(THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorSimdAligned, queryPtr, queryByteSize);

    // Set mmap pages
    THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPages.clear();
    THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPageSizes.clear();
    std::cout << "saveSearchContext: numAddressAndSize=" << numAddressAndSize << std::endl;
    for (int32_t i = 0 ; i < numAddressAndSize ; i += 2) {
        std::cout << "saveSearchContext: mmapPages=" << mmapAddressAndSize[i]
                  << ", size=" << mmapAddressAndSize[i + 1]
                  << std::endl;
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPages.emplace_back(reinterpret_cast<void*>(mmapAddressAndSize[i]));
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPageSizes.emplace_back(mmapAddressAndSize[i + 1]);
    }

    // Return thread_local object
    return &THREAD_LOCAL_SIMD_VEC_SRCH_CTX;
}

SimdVectorSearchContext* SimilarityFunction::getSearchContext() {
    return &THREAD_LOCAL_SIMD_VEC_SRCH_CTX;
}
