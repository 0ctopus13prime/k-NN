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

#include <jni.h>

#include <vector>

#include "faiss_wrapper.h"
#include "jni_util.h"
#include "faiss_stream_support.h"
#include "faiss/utils/distances.h"

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

const uint32_t PAGE_SIZE = 16 * 1024 * 1024;  // 16MB

struct FlatVectorManager {
    std::vector<std::vector<float>> floatValues;
};

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_allocateFlatVectorsManager
  (JNIEnv* env, jclass cls, jobject input, jint dimension, jint numVectors) {
    const int oneVecSize = sizeof(float) * dimension;
    const int numVecsInPage = PAGE_SIZE / oneVecSize;
    const int numPages = numVectors / numVecsInPage + ((numVectors % numVecsInPage) > 0 ? 1 : 0);

    knn_jni::stream::NativeEngineIndexInputMediator mediator {&jniUtil, env, input};
    auto manager = new FlatVectorManager();
    manager->floatValues.resize(numPages);
    int addedVecs = 0;
    for (int i = 0 ; i  < numPages ; ++i) {
        int numVecsToAdd = numVecsInPage;
        if ((addedVecs + numVecsInPage) > numVectors) {
            numVecsToAdd = numVectors - addedVecs;
        }
        manager->floatValues[i].resize(numVecsToAdd * dimension);
        auto data = (uint8_t*) manager->floatValues[i].data();
        mediator.copyBytes(numVecsToAdd * oneVecSize, data);
        addedVecs += numVecsToAdd;
    }

    // std::cout << "____________________ dimension=" << dimension
    //           << ", numVectors=" << numVectors
    //           << ", oneVecSize=" << oneVecSize
    //           << ", numVecsInPage=" << numVecsInPage
    //           << ", numPages=" << numPages
    //           << ", manager addr=" << ((uint64_t) manager)
    //           << std::endl;

    return (jlong) manager;
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_deallocateFlatVectorsManager
  (JNIEnv * env, jclass cls, jlong flatVectorsManagerAddress) {
    delete ((FlatVectorManager*) flatVectorsManagerAddress);
}

float convert_dist_to_lucene_score(float dist) {
    if (dist < 0) {
      return 1 / (1 - dist);
    } else {
      return dist + 1;
    }
}

void Java_org_opensearch_knn_jni_FaissService_bulkScoring(
   uint64_t flatVectorsManagerAddress,
   float* query,
   uint32_t* neighborList,
   float* scores,
   uint32_t size,
   uint32_t dimension) {
    auto manager = (FlatVectorManager*) flatVectorsManagerAddress;

    const int32_t oneVecSize = sizeof(float) * dimension;
    const int32_t numVecsInPage = PAGE_SIZE / oneVecSize;

    // std::cout << "_________ c++ bulk scoring"
    //           << ", flatVectorsManagerAddress=" << flatVectorsManagerAddress
    //           << ", query=" << query
    //           << ", neighborList=" << neighborList
    //           << ", scores=" << scores
    //           << ", size=" << size
    //           << ", dimension=" << dimension
    //           << std::endl;

    float* vectors[4];
    int idx = 0;
    int score_idx = 0;

    for (int i = 0 ; i < size ; ++i) {
        const long id = neighborList[i];
        const int page_index = id / numVecsInPage;
        const int vec_index = id % numVecsInPage;

        auto vec = manager->floatValues[page_index].data() + (vec_index * dimension);

        // std::cout << "__________________ c++ neighbor_id=" << id
        //           << ", page_index=" << page_index
        //           << ", vec_index=" << vec_index
        //           << ", score_idx=" << score_idx
        //           << ", q[0]=" << query[0]
        //           << std::endl;

        vectors[idx++] = vec;

        if (idx == 4) {
            // std::cout << "+++++++++++++++++++ before c++ bulk"
            //           << ", v[0]=" << vectors[0][0]
            //           << ", v[1]=" << vectors[1][0]
            //           << ", v[2]=" << vectors[2][0]
            //           << ", v[3]=" << vectors[3][0]
            //           << ", q=" << ((uint64_t) query)
            //           << std::endl;

            faiss::fvec_inner_product_batch_4(
                query,
                vectors[0],
                vectors[1],
                vectors[2],
                vectors[3],
                dimension,
                scores[score_idx],
                scores[score_idx + 1],
                scores[score_idx + 2],
                scores[score_idx + 3]);

            scores[score_idx] = convert_dist_to_lucene_score(scores[score_idx]);
            scores[score_idx + 1] = convert_dist_to_lucene_score(scores[score_idx + 1]);
            scores[score_idx + 2] = convert_dist_to_lucene_score(scores[score_idx + 2]);
            scores[score_idx + 3] = convert_dist_to_lucene_score(scores[score_idx + 3]);

            idx = 0;
            score_idx += 4;
        }
    }

    for (int i = 0 ; i < idx ; ++i) {
        scores[score_idx++] =
            convert_dist_to_lucene_score(
                faiss::fvec_inner_product(query, vectors[i], dimension));
    }
}

float Java_org_opensearch_knn_jni_FaissService_singleScoring
  (int64_t flatVectorsManagerAddress,
   float* query,
   int32_t vector_id,
   int32_t dimension) {

    auto manager = (FlatVectorManager*) flatVectorsManagerAddress;
    const int32_t oneVecSize = sizeof(float) * dimension;
    const int32_t numVecsInPage = PAGE_SIZE / oneVecSize;
    const int page_index = vector_id / numVecsInPage;
    const int vec_index = vector_id % numVecsInPage;
    auto vec = manager->floatValues[page_index].data() + (vec_index * dimension);

    float dist = faiss::fvec_inner_product(query, vec, dimension);

    // std::cout << "_______________ single score in c++ -> "
    //           << dist
    //           << ", manager addr=" << flatVectorsManagerAddress
    //           << ", dim=" << dimension
    //           << ", vector_id=" << vector_id
    //           << ", page_index=" << page_index
    //           << ", vec_index=" << vec_index
    //           << std::endl;
    // std::cout << "query -> [";
    // dimension = dimension > 8 ? 8 : dimension;
    // for (int i = 0 ; i < dimension ; ++i) {
    //     std::cout << ", " << query[i];
    // }
    // std::cout << std::endl;

    // std::cout << "vec -> [";
    // for (int i = 0 ; i < dimension ; ++i) {
    //     std::cout << ", " << vec[i];
    // }
    // std::cout << std::endl;

    return convert_dist_to_lucene_score(dist);
}

// TMP
