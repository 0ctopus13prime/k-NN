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

#include "org_opensearch_knn_jni_NmslibService.h"

#include <jni.h>

#include "jni_util.h"
#include "nmslib_wrapper.h"

// TMP
#include <chrono>
// TMP

static knn_jni::JNIUtil jniUtil;
static const jint KNN_NMSLIB_JNI_VERSION = JNI_VERSION_1_1;

jint JNI_OnLoad(JavaVM *vm, void *reserved) {
  JNIEnv *env;
  if (vm->GetEnv((void **) &env, KNN_NMSLIB_JNI_VERSION) != JNI_OK) {
    return JNI_ERR;
  }

  jniUtil.Initialize(env);

  return KNN_NMSLIB_JNI_VERSION;
}

void JNI_OnUnload(JavaVM *vm, void *reserved) {
  JNIEnv *env;
  vm->GetEnv((void **) &env, KNN_NMSLIB_JNI_VERSION);
  jniUtil.Uninitialize(env);
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_NmslibService_createIndex(JNIEnv *env,
                                                                             jclass cls,
                                                                             jintArray idsJ,
                                                                             jlong vectorsAddressJ,
                                                                             jint dimJ,
                                                                             jobject output,
                                                                             jobject parametersJ) {
  try {
    knn_jni::nmslib_wrapper::CreateIndex(&jniUtil, env, idsJ, vectorsAddressJ, dimJ, output, parametersJ);
  } catch (...) {
    jniUtil.CatchCppExceptionAndThrowJava(env);
  }
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_NmslibService_loadIndex(JNIEnv *env, jclass cls,
                                                                            jstring indexPathJ, jobject parametersJ) {
  try {
    return knn_jni::nmslib_wrapper::LoadIndex(&jniUtil, env, indexPathJ, parametersJ);
  } catch (...) {
    jniUtil.CatchCppExceptionAndThrowJava(env);
  }
  return NULL;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_NmslibService_loadIndexWithStream(JNIEnv *env,
                                                                                      jclass cls,
                                                                                      jobject readStream,
                                                                                      jobject parametersJ) {
  try {
    return knn_jni::nmslib_wrapper::LoadIndexWithStream(&jniUtil, env, readStream, parametersJ);
  } catch (...) {
    jniUtil.CatchCppExceptionAndThrowJava(env);
  }
  return NULL;
}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_NmslibService_queryIndex(JNIEnv *env,
                                                                                    jclass cls,
                                                                                    jlong indexPointerJ,
                                                                                    jfloatArray queryVectorJ,
                                                                                    jint kJ,
                                                                                    jobject methodParamsJ) {
  try {
    return knn_jni::nmslib_wrapper::QueryIndex(&jniUtil, env, indexPointerJ, queryVectorJ, kJ, methodParamsJ);
  } catch (...) {
    jniUtil.CatchCppExceptionAndThrowJava(env);
  }
  return nullptr;
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_NmslibService_free(JNIEnv *env, jclass cls, jlong indexPointerJ) {
  try {
    return knn_jni::nmslib_wrapper::Free(indexPointerJ);
  } catch (...) {
    jniUtil.CatchCppExceptionAndThrowJava(env);
  }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_NmslibService_initLibrary(JNIEnv *env, jclass cls) {
  try {
    knn_jni::nmslib_wrapper::InitLibrary();
  } catch (...) {
    jniUtil.CatchCppExceptionAndThrowJava(env);
  }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_NmslibService_kdyBench
    (JNIEnv * env, jclass cls, jlong numData, jint dim, jstring inputDataPathJ, jintArray ids,
     jobject parameters, jobject indexOutput, jstring fullPathj) {
  const std::string dataPath = jniUtil.ConvertJavaStringToCppString(env, inputDataPathJ);
  std::ifstream in (dataPath);
  std::string line;
  std::vector<float> vectors;
  while (std::getline(in, line)) {
    int s = 1;
    for (int i = s ; i < line.size() ; ) {
      while (line[i] != ',' && line[i] != ']') {
        ++i;
      }
      while (s < line.size() && line[s] == ' ') {
        ++s;
      }
      std::string value = line.substr(s, (i - s));
      const float fvalue = std::stod(value);
      vectors.push_back(fvalue);
      s = ++i;
    }
  }

  std::cout << dim << ", " << numData << ", " << vectors.size() << std::endl;

  // Stream
  try {
    auto start = std::chrono::high_resolution_clock::now();
    knn_jni::nmslib_wrapper::CreateIndex(&jniUtil, env, ids, (jlong) &vectors,
                                         dim, indexOutput, parameters);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = end - start;
    std::cout << "Stream file version -> Execution time: " << duration.count() << " microseconds" << std::endl;
  } catch (...) {
    jniUtil.CatchCppExceptionAndThrowJava(env);
  }

  // File based.
//  try {
//    const std::string fullPath = jniUtil.ConvertJavaStringToCppString(env, fullPathj);
//    auto start = std::chrono::high_resolution_clock::now();
//    knn_jni::nmslib_wrapper::CreateIndexLegacy(&jniUtil, env, ids, (jlong) &vectors,
//                                         dim, fullPath, parameters);
//    auto end = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double, std::micro> duration = end - start;
//    std::cout << "Stream file version -> Execution time: " << duration.count() << " microseconds" << std::endl;
//  } catch (...) {
//    jniUtil.CatchCppExceptionAndThrowJava(env);
//  }
}
