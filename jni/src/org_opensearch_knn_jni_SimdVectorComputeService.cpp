#include <cstring>
#include <limits>
#include <algorithm>
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

JNIEXPORT jfloat JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_scoreSimilarityInBulk
  (JNIEnv *env, jclass clazz, jintArray internalVectorIds, jfloatArray jscores, const jint numVectors) {
    if (numVectors <= 0) {
      return std::numeric_limits<float>::min();
    }

    try {
      // Get search context
      SimdVectorSearchContext* srchContext = SimilarityFunction::getSearchContext();
      if (srchContext == nullptr || srchContext->similarityFunction == nullptr) {
          throw std::runtime_error("No search context has been initialized, SimdVectorSearchContext* was empty.");
      }

      // Get pointers of vectorIds and scores
      jint* vectorIds = static_cast<jint*>(JNI_UTIL.GetPrimitiveArrayCritical(env, internalVectorIds, nullptr));
      knn_jni::JNIReleaseElements releaseVectorIds {[=]{
        JNI_UTIL.ReleasePrimitiveArrayCritical(env, internalVectorIds, vectorIds, 0);
      }};

      jfloat* scores = static_cast<jfloat*>(JNI_UTIL.GetPrimitiveArrayCritical(env, jscores, nullptr));
      knn_jni::JNIReleaseElements releaseScores {[=]{
        JNI_UTIL.ReleasePrimitiveArrayCritical(env, jscores, scores, 0);
      }};

      // Bulk similarity calculation
      srchContext->similarityFunction->calculateSimilarityInBulk(
          srchContext,
          reinterpret_cast<int32_t*>(vectorIds),
          reinterpret_cast<float*>(scores),
          numVectors);

      jfloat maxScore = *std::max_element(scores, scores + numVectors);
      return maxScore;
    } catch (...) {
      JNI_UTIL.CatchCppExceptionAndThrowJava(env);
      return 0.0f;   // value ignored if exception pending
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

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_saveSQSearchContext
  (JNIEnv *env, jclass clazz, jbyteArray quantizedQuery,
   jfloat lowerInterval, jfloat upperInterval, jfloat additionalCorrection,
   jint quantizedComponentSum, jlongArray addressAndSize,
   jint functionTypeOrd, jint dimension, jfloat centroidDp) {
    try {
      // Get quantized query bytes
      const jsize queryByteSize = JNI_UTIL.GetJavaBytesArrayLength(env, quantizedQuery);
      jbyte* queryPtr = static_cast<jbyte*>(JNI_UTIL.GetPrimitiveArrayCritical(env, quantizedQuery, nullptr));
      knn_jni::JNIReleaseElements queryRelease {[=]{
        JNI_UTIL.ReleasePrimitiveArrayCritical(env, quantizedQuery, queryPtr, 0);
      }};

      // Get mmap address and size
      const jsize mmapAddressAndSizeLength = JNI_UTIL.GetJavaLongArrayLength(env, addressAndSize);
      jlong* mmapAddressAndSize = static_cast<jlong*>(JNI_UTIL.GetPrimitiveArrayCritical(env, addressAndSize, nullptr));
      knn_jni::JNIReleaseElements mmapAddressAndSizeRelease {[=]{
        JNI_UTIL.ReleasePrimitiveArrayCritical(env, addressAndSize, mmapAddressAndSize, 0);
      }};

      // Store correction factors in tmpBuffer before calling saveSearchContext.
      // saveSearchContext will reset tmpBuffer at the beginning, so we need to call it first,
      // then write correction factors after.
      SimilarityFunction::saveSearchContext(
          reinterpret_cast<uint8_t*>(queryPtr), queryByteSize,
          dimension,
          reinterpret_cast<int64_t*>(mmapAddressAndSize), mmapAddressAndSizeLength,
          functionTypeOrd);

      // Now store correction factors in tmpBuffer (saveSearchContext clears it, then SQ_IP branch leaves it empty)
      SimdVectorSearchContext* ctx = SimilarityFunction::getSearchContext();
      ctx->tmpBuffer.resize(5 * sizeof(float));
      auto* correctionPtr = reinterpret_cast<float*>(ctx->tmpBuffer.data());
      correctionPtr[0] = lowerInterval;
      correctionPtr[1] = upperInterval;
      correctionPtr[2] = additionalCorrection;
      std::memcpy(&correctionPtr[3], &quantizedComponentSum, sizeof(int32_t));
      correctionPtr[4] = centroidDp;
    } catch (...) {
      JNI_UTIL.CatchCppExceptionAndThrowJava(env);
    }
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
                // Release the critical array before throwing: destructors run during unwinding, before the catch.
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

JNIEXPORT jint JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_bulkCentroidDistance(
    JNIEnv* env, jclass,
    jfloatArray vectorArr, jfloatArray centroidsArr, jfloatArray distancesArr,
    jint dimension, jint numCentroids, jint metricOrd
) {
    try {
        float* vec = env->GetFloatArrayElements(vectorArr, nullptr);
        float* cents = env->GetFloatArrayElements(centroidsArr, nullptr);
        float* dists = env->GetFloatArrayElements(distancesArr, nullptr);

        int bestIdx = 0;
        float bestDist = std::numeric_limits<float>::max();

        for (int32_t i = 0; i < numCentroids; i++) {
            float* c = cents + (int64_t)i * dimension;
            float d = 0;
            if (metricOrd == 0) { // L2
                for (int32_t j = 0; j < dimension; j++) {
                    float diff = vec[j] - c[j];
                    d += diff * diff;
                }
            } else { // DOT_PRODUCT (negated for distance)
                for (int32_t j = 0; j < dimension; j++) {
                    d -= vec[j] * c[j];
                }
            }
            dists[i] = d;
            if (d < bestDist) { bestDist = d; bestIdx = i; }
        }

        env->ReleaseFloatArrayElements(distancesArr, dists, 0);
        env->ReleaseFloatArrayElements(centroidsArr, cents, JNI_ABORT);
        env->ReleaseFloatArrayElements(vectorArr, vec, JNI_ABORT);
        return bestIdx;
    } catch (...) {
        JNI_UTIL.CatchCppExceptionAndThrowJava(env);
        return -1;
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_bulkSOARDistance(
    JNIEnv* env, jclass,
    jfloatArray vectorArr, jfloatArray primaryArr, jfloatArray candidatesArr,
    jfloatArray distancesArr, jint dimension, jint numCandidates, jfloat soarLambda
) {
    try {
        float* vec = env->GetFloatArrayElements(vectorArr, nullptr);
        float* primary = env->GetFloatArrayElements(primaryArr, nullptr);
        float* cands = env->GetFloatArrayElements(candidatesArr, nullptr);
        float* dists = env->GetFloatArrayElements(distancesArr, nullptr);

        // Compute residual and norm (stack-allocated for thread safety)
        std::vector<float> residual(dimension);
        float residualNormSq = 0;
        for (int32_t d = 0; d < dimension; d++) {
            residual[d] = vec[d] - primary[d];
            residualNormSq += residual[d] * residual[d];
        }

        if (residualNormSq < 1e-20f) {
            for (int32_t i = 0; i < numCandidates; i++) dists[i] = std::numeric_limits<float>::max();
        } else {
            float invNorm = soarLambda / residualNormSq;
            for (int32_t i = 0; i < numCandidates; i++) {
                float* c = cands + (int64_t)i * dimension;
                float dsq = 0, proj = 0;
                for (int32_t d = 0; d < dimension; d++) {
                    float diff = vec[d] - c[d];
                    dsq += diff * diff;
                    proj += residual[d] * diff;
                }
                dists[i] = dsq + invNorm * proj * proj;
            }
        }

        env->ReleaseFloatArrayElements(distancesArr, dists, 0);
        env->ReleaseFloatArrayElements(candidatesArr, cands, JNI_ABORT);
        env->ReleaseFloatArrayElements(primaryArr, primary, JNI_ABORT);
        env->ReleaseFloatArrayElements(vectorArr, vec, JNI_ABORT);
    } catch (...) {
        JNI_UTIL.CatchCppExceptionAndThrowJava(env);
    }
}
