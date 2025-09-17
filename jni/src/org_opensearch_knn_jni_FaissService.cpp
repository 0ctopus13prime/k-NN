/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

#include "org_opensearch_knn_jni_FaissService.h"
#include "faiss/impl/ScalarQuantizer.h"

#include <jni.h>

#include <vector>

#include "faiss_wrapper.h"
#include "jni_util.h"
#include "faiss_stream_support.h"

static knn_jni::JNIUtil jniUtil;
static const jint KNN_FAISS_JNI_VERSION = JNI_VERSION_1_1;

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    // Obtain the JNIEnv from the VM and confirm JNI_VERSION
    JNIEnv* env;
    if (vm->GetEnv((void**)&env, KNN_FAISS_JNI_VERSION) != JNI_OK) {
        return JNI_ERR;
    }

    jniUtil.Initialize(env);

    return KNN_FAISS_JNI_VERSION;
}

void JNI_OnUnload(JavaVM *vm, void *reserved) {
    JNIEnv* env;
    vm->GetEnv((void**)&env, KNN_FAISS_JNI_VERSION);
    jniUtil.Uninitialize(env);
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_initIndex(JNIEnv * env, jclass cls,
                                                                           jlong numDocs, jint dimJ,
                                                                           jobject parametersJ)
{
    try {
        std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
        knn_jni::faiss_wrapper::IndexService indexService(std::move(faissMethods));
        return knn_jni::faiss_wrapper::InitIndex(&jniUtil, env, numDocs, dimJ, parametersJ, &indexService);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return (jlong)0;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_initBinaryIndex(JNIEnv * env, jclass cls,
                                                                                 jlong numDocs, jint dimJ,
                                                                                 jobject parametersJ)
{
    try {
        std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
        knn_jni::faiss_wrapper::BinaryIndexService binaryIndexService(std::move(faissMethods));
        return knn_jni::faiss_wrapper::InitIndex(&jniUtil, env, numDocs, dimJ, parametersJ, &binaryIndexService);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return (jlong)0;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_initByteIndex(JNIEnv * env, jclass cls,
                                                                               jlong numDocs, jint dimJ,
                                                                               jobject parametersJ)
{
    try {
        std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
        knn_jni::faiss_wrapper::ByteIndexService byteIndexService(std::move(faissMethods));
        return knn_jni::faiss_wrapper::InitIndex(&jniUtil, env, numDocs, dimJ, parametersJ, &byteIndexService);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return (jlong)0;
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_insertToIndex(JNIEnv * env, jclass cls, jintArray idsJ,
                                                                              jlong vectorsAddressJ, jint dimJ,
                                                                              jlong indexAddress, jint threadCount)
{
    try {
        std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
        knn_jni::faiss_wrapper::IndexService indexService(std::move(faissMethods));
        knn_jni::faiss_wrapper::InsertToIndex(&jniUtil, env, idsJ, vectorsAddressJ, dimJ, indexAddress, threadCount, &indexService);
    } catch (...) {
        // NOTE: ADDING DELETE STATEMENT HERE CAUSES A CRASH!
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_insertToBinaryIndex(JNIEnv * env, jclass cls, jintArray idsJ,
                                                                                    jlong vectorsAddressJ, jint dimJ,
                                                                                    jlong indexAddress, jint threadCount)
{
    try {
        std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
        knn_jni::faiss_wrapper::BinaryIndexService binaryIndexService(std::move(faissMethods));
        knn_jni::faiss_wrapper::InsertToIndex(&jniUtil, env, idsJ, vectorsAddressJ, dimJ, indexAddress, threadCount, &binaryIndexService);
    } catch (...) {
        // NOTE: ADDING DELETE STATEMENT HERE CAUSES A CRASH!
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_insertToByteIndex(JNIEnv * env, jclass cls, jintArray idsJ,
                                                                                  jlong vectorsAddressJ, jint dimJ,
                                                                                  jlong indexAddress, jint threadCount)
{
    try {
        std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
        knn_jni::faiss_wrapper::ByteIndexService byteIndexService(std::move(faissMethods));
        knn_jni::faiss_wrapper::InsertToIndex(&jniUtil, env, idsJ, vectorsAddressJ, dimJ, indexAddress, threadCount, &byteIndexService);
    } catch (...) {
        // NOTE: ADDING DELETE STATEMENT HERE CAUSES A CRASH!
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_writeIndex(JNIEnv * env,
                                                                           jclass cls,
                                                                           jlong indexAddress,
                                                                           jobject output)
{
  try {
      std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
      knn_jni::faiss_wrapper::IndexService indexService(std::move(faissMethods));
      knn_jni::faiss_wrapper::WriteIndex(&jniUtil, env, output, indexAddress, &indexService);
  } catch (...) {
      jniUtil.CatchCppExceptionAndThrowJava(env);
  }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_writeBinaryIndex(JNIEnv * env,
                                                                                 jclass cls,
                                                                                 jlong indexAddress,
                                                                                 jobject output)
{
  try {
      std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
      knn_jni::faiss_wrapper::BinaryIndexService binaryIndexService(std::move(faissMethods));
      knn_jni::faiss_wrapper::WriteIndex(&jniUtil, env, output, indexAddress, &binaryIndexService);
  } catch (...) {
      jniUtil.CatchCppExceptionAndThrowJava(env);
  }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_writeByteIndex(JNIEnv * env,
                                                                               jclass cls,
                                                                               jlong indexAddress,
                                                                               jobject output)
{
  try {
      std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
      knn_jni::faiss_wrapper::ByteIndexService byteIndexService(std::move(faissMethods));
      knn_jni::faiss_wrapper::WriteIndex(&jniUtil, env, output, indexAddress, &byteIndexService);
  } catch (...) {
      jniUtil.CatchCppExceptionAndThrowJava(env);
  }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_createIndexFromTemplate(JNIEnv * env,
                                                                                        jclass cls,
                                                                                        jintArray idsJ,
                                                                                        jlong vectorsAddressJ,
                                                                                        jint dimJ,
                                                                                        jobject output,
                                                                                        jbyteArray templateIndexJ,
                                                                                        jobject parametersJ)
{
    try {
        knn_jni::faiss_wrapper::CreateIndexFromTemplate(&jniUtil,
                                                        env,
                                                        idsJ,
                                                        vectorsAddressJ,
                                                        dimJ,
                                                        output,
                                                        templateIndexJ,
                                                        parametersJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_createBinaryIndexFromTemplate(JNIEnv * env,
                                                                                              jclass cls,
                                                                                              jintArray idsJ,
                                                                                              jlong vectorsAddressJ,
                                                                                              jint dimJ,
                                                                                              jobject output,
                                                                                              jbyteArray templateIndexJ,
                                                                                              jobject parametersJ)
{
    try {
        knn_jni::faiss_wrapper::CreateBinaryIndexFromTemplate(&jniUtil,
                                                              env,
                                                              idsJ,
                                                              vectorsAddressJ,
                                                              dimJ,
                                                              output,
                                                              templateIndexJ,
                                                              parametersJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_createByteIndexFromTemplate(JNIEnv * env,
                                                                                            jclass cls,
                                                                                            jintArray idsJ,
                                                                                            jlong vectorsAddressJ,
                                                                                            jint dimJ,
                                                                                            jobject output,
                                                                                            jbyteArray templateIndexJ,
                                                                                            jobject parametersJ)
{
    try {
        knn_jni::faiss_wrapper::CreateByteIndexFromTemplate(&jniUtil,
                                                            env,
                                                            idsJ,
                                                            vectorsAddressJ,
                                                            dimJ,
                                                            output,
                                                            templateIndexJ,
                                                            parametersJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_loadIndex(JNIEnv * env, jclass cls, jstring indexPathJ)
{
  try {
      return knn_jni::faiss_wrapper::LoadIndex(&jniUtil, env, indexPathJ);
  } catch (...) {
      jniUtil.CatchCppExceptionAndThrowJava(env);
  }
  return NULL;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_loadIndexWithStream(JNIEnv * env,
                                                                                     jclass cls,
                                                                                     jobject readStream)
{
    try {
        // Create a mediator locally.
        // Note that `indexInput` is `IndexInputWithBuffer` type.
        knn_jni::stream::NativeEngineIndexInputMediator mediator {&jniUtil, env, readStream};

        // Wrap the mediator with a glue code inheriting IOReader.
        knn_jni::stream::FaissOpenSearchIOReader faissOpenSearchIOReader {&mediator};

        // Pass IOReader to Faiss for loading vector index.
        return knn_jni::faiss_wrapper::LoadIndexWithStream(
                 &faissOpenSearchIOReader);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }

    return NULL;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_loadBinaryIndex(JNIEnv * env, jclass cls, jstring indexPathJ)
{
    try {
        return knn_jni::faiss_wrapper::LoadBinaryIndex(&jniUtil, env, indexPathJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return NULL;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_loadBinaryIndexWithStream(JNIEnv * env,
                                                                                           jclass cls,
                                                                                           jobject readStream)
{
    try {
        // Create a mediator locally.
        // Note that `indexInput` is `IndexInputWithBuffer` type.
        knn_jni::stream::NativeEngineIndexInputMediator mediator {&jniUtil, env, readStream};

        // Wrap the mediator with a glue code inheriting IOReader.
        knn_jni::stream::FaissOpenSearchIOReader faissOpenSearchIOReader {&mediator};

        // Pass IOReader to Faiss for loading vector index.
        return knn_jni::faiss_wrapper::LoadBinaryIndexWithStream(
            &faissOpenSearchIOReader);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }

    return NULL;
}
JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_loadIndexWithStreamADCParams
(JNIEnv * env, jclass cls, jobject readStreamJ, jobject parametersJ) {
    try {
        knn_jni::stream::NativeEngineIndexInputMediator mediator {&jniUtil, env, readStreamJ};

        // Wrap the mediator with a glue code inheriting IOReader.
        knn_jni::stream::FaissOpenSearchIOReader faissOpenSearchIOReader {&mediator};

        return knn_jni::faiss_wrapper::LoadIndexWithStreamADCParams(&faissOpenSearchIOReader, &jniUtil, env, parametersJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return NULL;
}

JNIEXPORT jboolean JNICALL Java_org_opensearch_knn_jni_FaissService_isSharedIndexStateRequired(JNIEnv * env,
                                                                                               jclass cls,
                                                                                               jlong indexPointerJ)
{
    try {
        return knn_jni::faiss_wrapper::IsSharedIndexStateRequired(indexPointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return NULL;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_initSharedIndexState
        (JNIEnv * env, jclass cls, jlong indexPointerJ)
{
    try {
        return knn_jni::faiss_wrapper::InitSharedIndexState(indexPointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return NULL;
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_setSharedIndexState
        (JNIEnv * env, jclass cls, jlong indexPointerJ, jlong shareIndexStatePointerJ)
{
    try {
        knn_jni::faiss_wrapper::SetSharedIndexState(indexPointerJ, shareIndexStatePointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_queryIndex(JNIEnv * env, jclass cls,
                                                                                   jlong indexPointerJ,
                                                                                   jfloatArray queryVectorJ, jint kJ, jobject methodParamsJ, jintArray parentIdsJ)
{
    try {
        return knn_jni::faiss_wrapper::QueryIndex(&jniUtil, env, indexPointerJ, queryVectorJ, kJ, methodParamsJ, parentIdsJ);

    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_queryIndexWithFilter
  (JNIEnv * env, jclass cls, jlong indexPointerJ, jfloatArray queryVectorJ, jint kJ, jobject methodParamsJ, jlongArray filteredIdsJ, jint filterIdsTypeJ,  jintArray parentIdsJ) {

      try {
          return knn_jni::faiss_wrapper::QueryIndex_WithFilter(&jniUtil, env, indexPointerJ, queryVectorJ, kJ, methodParamsJ, filteredIdsJ, filterIdsTypeJ, parentIdsJ);
      } catch (...) {
          jniUtil.CatchCppExceptionAndThrowJava(env);
      }
      return nullptr;

}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_queryBinaryIndexWithFilter
  (JNIEnv * env, jclass cls, jlong indexPointerJ, jbyteArray queryVectorJ, jint kJ, jobject methodParamsJ, jlongArray filteredIdsJ, jint filterIdsTypeJ,  jintArray parentIdsJ) {

      try {
          return knn_jni::faiss_wrapper::QueryBinaryIndex_WithFilter(&jniUtil, env, indexPointerJ, queryVectorJ, kJ, methodParamsJ, filteredIdsJ, filterIdsTypeJ, parentIdsJ);
      } catch (...) {
          jniUtil.CatchCppExceptionAndThrowJava(env);
      }
      return nullptr;

}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_free(JNIEnv * env, jclass cls, jlong indexPointerJ, jboolean isBinaryIndexJ)
{
    try {
        return knn_jni::faiss_wrapper::Free(indexPointerJ, isBinaryIndexJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_freeSharedIndexState
        (JNIEnv * env, jclass cls, jlong shareIndexStatePointerJ)
{
    try {
        knn_jni::faiss_wrapper::FreeSharedIndexState(shareIndexStatePointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_initLibrary(JNIEnv * env, jclass cls)
{
    try {
        knn_jni::faiss_wrapper::InitLibrary();
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT jbyteArray JNICALL Java_org_opensearch_knn_jni_FaissService_trainIndex(JNIEnv * env, jclass cls,
                                                                                 jobject parametersJ,
                                                                                 jint dimensionJ,
                                                                                 jlong trainVectorsPointerJ)
{
    try {
        return knn_jni::faiss_wrapper::TrainIndex(&jniUtil, env, parametersJ, dimensionJ, trainVectorsPointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT jbyteArray JNICALL Java_org_opensearch_knn_jni_FaissService_trainBinaryIndex(JNIEnv * env, jclass cls,
                                                                                 jobject parametersJ,
                                                                                 jint dimensionJ,
                                                                                 jlong trainVectorsPointerJ)
{
    try {
        return knn_jni::faiss_wrapper::TrainBinaryIndex(&jniUtil, env, parametersJ, dimensionJ, trainVectorsPointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT jbyteArray JNICALL Java_org_opensearch_knn_jni_FaissService_trainByteIndex(JNIEnv * env, jclass cls,
                                                                                 jobject parametersJ,
                                                                                 jint dimensionJ,
                                                                                 jlong trainVectorsPointerJ)
{
    try {
        return knn_jni::faiss_wrapper::TrainByteIndex(&jniUtil, env, parametersJ, dimensionJ, trainVectorsPointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_transferVectors(JNIEnv * env, jclass cls,
                                                                                 jlong vectorsPointerJ,
                                                                                 jobjectArray vectorsJ)
{
    std::vector<float> *vect;
    if ((long) vectorsPointerJ == 0) {
        vect = new std::vector<float>;
    } else {
        vect = reinterpret_cast<std::vector<float>*>(vectorsPointerJ);
    }

    int dim = jniUtil.GetInnerDimensionOf2dJavaFloatArray(env, vectorsJ);
    auto dataset = jniUtil.Convert2dJavaObjectArrayToCppFloatVector(env, vectorsJ, dim);
    vect->insert(vect->begin(), dataset.begin(), dataset.end());

    return (jlong) vect;
}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_rangeSearchIndex(JNIEnv * env, jclass cls,
                                                                                         jlong indexPointerJ,
                                                                                         jfloatArray queryVectorJ,
                                                                                         jfloat radiusJ, jobject methodParamsJ,
                                                                                         jint maxResultWindowJ, jintArray parentIdsJ)
{
    try {
        return knn_jni::faiss_wrapper::RangeSearch(&jniUtil, env, indexPointerJ, queryVectorJ, radiusJ, methodParamsJ, maxResultWindowJ, parentIdsJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_rangeSearchIndexWithFilter(JNIEnv * env, jclass cls,
                                                                                                   jlong indexPointerJ,
                                                                                                   jfloatArray queryVectorJ,
                                                                                                   jfloat radiusJ, jobject methodParamsJ, jint maxResultWindowJ,
                                                                                                   jlongArray filterIdsJ, jint filterIdsTypeJ, jintArray parentIdsJ)
{
    try {
        return knn_jni::faiss_wrapper::RangeSearchWithFilter(&jniUtil, env, indexPointerJ, queryVectorJ, radiusJ, methodParamsJ, maxResultWindowJ, filterIdsJ, filterIdsTypeJ, parentIdsJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

// TMP

void batch_inner_product_4_fp16_targets(
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

//     std::cout << "_____________ q=" << ((uint64_t) query)
//            << ", d0=" << ((uint64_t) d0)
//            << ", d1=" << ((uint64_t) d1)
//            << ", d2=" << ((uint64_t) d2)
//            << ", d3=" << ((uint64_t) d3)
//            << ", dim=" << dim
//            << std::endl;
//

    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

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

    // Scalar tail
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

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    bulkScoring2
 * Signature: (JJIJJI)V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_bulkScoring1
  (JNIEnv *env, jclass clazz, jlong queryAddr, jlong neighborsAddr, jint numNeighbors, jlong flatVectorSectionAddr, jlong scoresAddr, jint oneVectorByteSize) {
    thread_local static auto dist = faiss::ScalarQuantizer {768, faiss::ScalarQuantizer::QuantizerType::QT_fp16}.get_distance_computer(faiss::MetricType::METRIC_INNER_PRODUCT);

    // Set query
    dist->q = (float*) queryAddr;

    // Get scores[]
    auto scores = (float*) scoresAddr;

    // Bulk scoring
    auto* neighbors = (const int32_t*) neighborsAddr;
    auto* flatVectors = (const uint8_t*) flatVectorSectionAddr;
    int scoreIdx  = 0;
    const uint64_t oneVectorByteSizeL = oneVectorByteSize;

    for (int i = 0 ; i < numNeighbors ; ++i) {
        auto ptr_vec = flatVectors + (neighbors[i] * oneVectorByteSizeL);
        const float score = dist->query_to_code(ptr_vec);
        scores[scoreIdx++] = score;
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_bulkScoring2
  (JNIEnv *env, jclass clazz, jlong queryAddr, jlong neighborsAddr, jint numNeighbors, jlong flatVectorSectionAddr, jlong scoresAddr, jint oneVectorByteSize) {
    thread_local static auto dist = faiss::ScalarQuantizer {768, faiss::ScalarQuantizer::QuantizerType::QT_fp16}.get_distance_computer(faiss::MetricType::METRIC_INNER_PRODUCT);
    // Get scores[]
    auto scores = (float*) scoresAddr;

    // Bulk scoring
    float fourScores[4];
    uint8_t* fp16Vecs[4];
    auto* neighbors = (const int32_t*) neighborsAddr;
    auto* flatVectors = (const uint8_t*) flatVectorSectionAddr;
    int scoreIdx  = 0;
    const size_t dimension = 768;
    const uint64_t oneVectorByteSizeL = oneVectorByteSize;

    int i = 0;
    for ( ; (i + 4) <= numNeighbors ; i += 4) {
        for (int j = 0 ; j < 4 ; ++j) {
            fp16Vecs[j] = (uint8_t*) (flatVectorSectionAddr + (neighbors[i + j] * oneVectorByteSizeL));
        }

        batch_inner_product_4_fp16_targets((float*) queryAddr, fp16Vecs[0], fp16Vecs[1], fp16Vecs[2], fp16Vecs[3], dimension,
                                           fourScores[0], fourScores[1], fourScores[2], fourScores[3]);

        for (int j = 0 ; j < 4 ; ++j) {
            scores[scoreIdx++] = fourScores[j];
        }
    }

    dist->q = (float*) queryAddr;
    while (i < numNeighbors) {
        auto ptr_vec = (uint8_t*) (flatVectorSectionAddr + (neighbors[i] * oneVectorByteSizeL));
        const float score = dist->query_to_code((const uint8_t*) ptr_vec);
        scores[scoreIdx++] = score;
     i += 1;
    }
}

// TMP


