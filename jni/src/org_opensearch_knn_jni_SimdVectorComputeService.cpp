#include <string>
#include "org_opensearch_knn_jni_SimdVectorComputeService.h"
#include "jni_util.h"
#include "simd/similarity_function/similarity_function.h"
#include "simd/clusterann_batch_dot_product.h"

static knn_jni::JNIUtil JNI_UTIL;
static constexpr jint KNN_SIMD_COMPUTE_JNI_VERSION = JNI_VERSION_1_1;

using knn_jni::simd::similarity_function::SimilarityFunction;
using knn_jni::simd::similarity_function::SimdVectorSearchContext;

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    // Obtain the JNIEnv from the VM and confirm JNI_VERSION
    JNIEnv* env;
    if (vm->GetEnv((void**)&env, KNN_SIMD_COMPUTE_JNI_VERSION) != JNI_OK) {
        return JNI_ERR;
    }

    JNI_UTIL.Initialize(env, vm);

    return KNN_SIMD_COMPUTE_JNI_VERSION;
}

void JNI_OnUnload(JavaVM *vm, void *reserved) {
    JNIEnv* env;
    vm->GetEnv((void**)&env, KNN_SIMD_COMPUTE_JNI_VERSION);
    JNI_UTIL.Uninitialize(env);
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_scoreSimilarityInBulk
  (JNIEnv *env, jclass clazz, jintArray internalVectorIds, jfloatArray jscores, const jint numVectors) {

    try {
      // Get search context
      SimdVectorSearchContext* srchContext = SimilarityFunction::getSearchContext();
      if (srchContext == nullptr || srchContext->similarityFunction == nullptr) {
          throw std::runtime_error("No search context has been initialized, SimdVectorSearchContext* was empty.");
      }

      // Get pointers of vectorIds and scores
      jint* vectorIds = static_cast<jint*>(JNI_UTIL.GetPrimitiveArrayCritical(env, internalVectorIds, nullptr));
      jfloat* scores = static_cast<jfloat*>(JNI_UTIL.GetPrimitiveArrayCritical(env, jscores, nullptr));

      // Bulk similarity calculation
      srchContext->similarityFunction->calculateSimilarityInBulk(
          srchContext,
          reinterpret_cast<int32_t*>(vectorIds),
          reinterpret_cast<float*>(scores),
          numVectors);

      // Release pinned pointers
      JNI_UTIL.ReleasePrimitiveArrayCritical(env, internalVectorIds, vectorIds, 0);
      JNI_UTIL.ReleasePrimitiveArrayCritical(env, jscores, scores, 0);
    } catch (...) {
      JNI_UTIL.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_saveSearchContext
  (JNIEnv *env, jclass clazz, jfloatArray query, jlongArray addressAndSize, const jint nativeFunctionTypeOrd) {
    try {
      // Get raw pointer of query vector + size
      const jsize queryVecSize = JNI_UTIL.GetJavaFloatArrayLength(env, query);
      jfloat* queryVecPtr = static_cast<jfloat*>(JNI_UTIL.GetPrimitiveArrayCritical(env, query, nullptr));

      // Get mmap address and size
      const jsize mmapAddressAndSizeLength = JNI_UTIL.GetJavaLongArrayLength(env, addressAndSize);
      jlong* mmapAddressAndSize = static_cast<jlong*>(JNI_UTIL.GetPrimitiveArrayCritical(env, addressAndSize, nullptr));

      // Save search context
      SimilarityFunction::saveSearchContext(
          (uint8_t*) queryVecPtr, sizeof(jfloat) * queryVecSize,
          queryVecSize,
          (int64_t*) mmapAddressAndSize, mmapAddressAndSizeLength,
          nativeFunctionTypeOrd);

      // Release query vector
      JNI_UTIL.ReleasePrimitiveArrayCritical(env, query, queryVecPtr, 0);
      JNI_UTIL.ReleasePrimitiveArrayCritical(env, addressAndSize, mmapAddressAndSize, 0);
    } catch (...) {
      JNI_UTIL.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT jfloat JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_scoreSimilarity
  (JNIEnv *env, jclass clazz, const jint internalVectorId) {

    try {
      // Get search context
      SimdVectorSearchContext* srchContext = SimilarityFunction::getSearchContext();

      // Single vector similarity calculation.
      return srchContext->similarityFunction->calculateSimilarity(srchContext, internalVectorId);
    } catch (...) {
      JNI_UTIL.CatchCppExceptionAndThrowJava(env);
    }

    return 0;
}

// ===== ClusterANN bulk operations (pure functions, no state, thread-safe) =====

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_bulkQuantizedDotProduct(
    JNIEnv* env, jclass,
    jbyteArray queryArr, jbyteArray docsArr, jintArray offsetsArr, const jint numOffsets,
    jfloatArray resultsArr, const jint bytesPerCode, const jint docBits
) {
    if (numOffsets <= 0) {
        return;
    }

    try {
        // Validate shapes before touching any array so that a bad call cannot read out of bounds.
        // All checks throw std::runtime_error, which CatchCppExceptionAndThrowJava maps to java.lang.Exception.
        if (docBits != 1 && docBits != 2 && docBits != 4) {
            throw std::runtime_error("docBits must be 1, 2 or 4, got " + std::to_string(docBits));
        }
        if (bytesPerCode <= 0 || (docBits != 4 && (bytesPerCode % docBits) != 0)) {
            throw std::runtime_error("bytesPerCode=" + std::to_string(bytesPerCode)
                                     + " must be a positive multiple of docBits=" + std::to_string(docBits));
        }
        // 1/2-bit: 4 transposed query planes of (bytesPerCode / docBits) bytes. 4-bit: unpacked query codes,
        // 2 * bytesPerCode bytes (high-nibble half followed by low-nibble half).
        const int32_t requiredQueryLen = docBits == 4 ? 2 * bytesPerCode : 4 * (bytesPerCode / docBits);
        const jsize queryLen = JNI_UTIL.GetJavaBytesArrayLength(env, queryArr);
        if (queryLen < requiredQueryLen) {
            throw std::runtime_error("query length " + std::to_string(queryLen) + " < required "
                                     + std::to_string(requiredQueryLen) + " for docBits=" + std::to_string(docBits));
        }
        const jsize docsLen = JNI_UTIL.GetJavaBytesArrayLength(env, docsArr);
        const jsize offsetsLen = JNI_UTIL.GetJavaIntArrayLength(env, offsetsArr);
        const jsize resultsLen = JNI_UTIL.GetJavaFloatArrayLength(env, resultsArr);
        if (offsetsLen < numOffsets) {
            throw std::runtime_error("offsets length " + std::to_string(offsetsLen)
                                     + " < numOffsets " + std::to_string(numOffsets));
        }

        auto* offsets = static_cast<jint*>(JNI_UTIL.GetPrimitiveArrayCritical(env, offsetsArr, nullptr));
        knn_jni::JNIReleaseElements releaseOffsets {[=] {
            JNI_UTIL.ReleasePrimitiveArrayCritical(env, offsetsArr, offsets, JNI_ABORT);
        }};

        // Every offset must address a whole record inside `docs` and a slot inside `results`.
        const int64_t maxOffset = (docsLen / bytesPerCode) - 1;
        for (jint k = 0; k < numOffsets; ++k) {
            if (offsets[k] < 0 || offsets[k] > maxOffset || offsets[k] >= resultsLen) {
                // The critical array is released by the guard's destructor during unwinding, before the catch runs.
                throw std::runtime_error("offsets[" + std::to_string(k) + "]=" + std::to_string(offsets[k])
                                         + " out of range (docs records=" + std::to_string(maxOffset + 1)
                                         + ", results=" + std::to_string(resultsLen) + ")");
            }
        }

        auto* q = static_cast<uint8_t*>(JNI_UTIL.GetPrimitiveArrayCritical(env, queryArr, nullptr));
        knn_jni::JNIReleaseElements releaseQuery {[=] {
            JNI_UTIL.ReleasePrimitiveArrayCritical(env, queryArr, q, JNI_ABORT);
        }};
        auto* d = static_cast<uint8_t*>(JNI_UTIL.GetPrimitiveArrayCritical(env, docsArr, nullptr));
        knn_jni::JNIReleaseElements releaseDocs {[=] {
            JNI_UTIL.ReleasePrimitiveArrayCritical(env, docsArr, d, JNI_ABORT);
        }};
        auto* r = static_cast<jfloat*>(JNI_UTIL.GetPrimitiveArrayCritical(env, resultsArr, nullptr));
        knn_jni::JNIReleaseElements releaseResults {[=] {
            JNI_UTIL.ReleasePrimitiveArrayCritical(env, resultsArr, r, 0);
        }};

        using namespace knn_jni::simd::clusterann;
        switch (docBits) {
            case 1: bulkDotProduct1bit(q, d, offsets, numOffsets, r, bytesPerCode); break;
            case 2: bulkDotProduct2bit(q, d, offsets, numOffsets, r, bytesPerCode); break;
            default: bulkDotProduct4bit(q, d, offsets, numOffsets, r, bytesPerCode); break;
        }
    } catch (...) {
        JNI_UTIL.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT jstring JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_bulkQuantizedDotProductKernel(
    JNIEnv* env, jclass
) {
    // "<bit-plane kernel>/<nibble kernel>", e.g. "avx512/avx512vnni" or "neon/neon-dotprod".
    const std::string name = std::string(knn_jni::simd::clusterann::kernelName()) + "/"
                             + knn_jni::simd::clusterann::nibbleKernelName();
    return env->NewStringUTF(name.c_str());
}
