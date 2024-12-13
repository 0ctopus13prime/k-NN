//
// Created by Kim, Dooyong on 11/27/24.
//

#ifndef KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_CONTEXT_H_
#define KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_CONTEXT_H_

#include <optional>

#include "jni_util.h"
#include "parameter_utils.h"
#include "memory_util.h"

#include <jni.h>

namespace knn_jni::partial_loading {

struct PartialLoadingContext {
  JNIUtilInterface *jni_interface_;
  JNIEnv *env_;
  jobject partial_loading_context_;
  jobject index_input_thread_local_getter_;
  static thread_local std::optional<knn_jni::stream::NativeEngineIndexInputMediator> mediator;

  PartialLoadingContext(JNIUtilInterface *jni_interface, JNIEnv *env, jobject partial_loading_context)
      : jni_interface_(jni_interface),
        env_(env),
        partial_loading_context_(partial_loading_context),
        index_input_thread_local_getter_(jni_interface->GetObjectField(env,
                                                                       partial_loading_context,
                                                                       getIndexInputThreadLocalGetterFieldId(
                                                                           jni_interface,
                                                                           env))) {
  }

  void setIndexInputWithBufferInThreadLocal() {
    jobject index_input_with_buffer = getIndexInputWithBuffer(jni_interface_, env_, index_input_thread_local_getter_);
    mediator = knn_jni::stream::NativeEngineIndexInputMediator(jni_interface_, env_, index_input_with_buffer);
  }

  std::string getMMapFilePathIfAvailable() {
    static jmethodID GET_MMAP_FILE_PATH_IF_AVAILABLE_METHOD_ID =
        jni_interface_->GetMethodID(env_,
                                    getPartialContextClass(jni_interface_, env_),
                                    "getMMapFilePathIfAvailable",
                                    "()Ljava/lang/String;");
    jobject file_path = jni_interface_->CallNonvirtualObjectMethodA(
        env_, partial_loading_context_, getPartialContextClass(jni_interface_, env_),
        GET_MMAP_FILE_PATH_IF_AVAILABLE_METHOD_ID, nullptr);

    if (file_path) {
      return jni_interface_->ConvertJavaStringToCppString(env_, (jstring) file_path);
    }

    return std::string{};
  }

  static jobject getIndexInputWithBuffer(JNIUtilInterface *jni_interface,
                                         JNIEnv *env,
                                         jobject index_input_thread_local_getter) {
    static jclass INDEX_INPUT_THREAD_LOCAL_GETTER_CLASS =
        jni_interface->FindClassFromJNIEnv(env, "org/opensearch/knn/index/store/IndexInputThreadLocalGetter");
    static jmethodID GET_INDEX_INPUT_WITH_BUFFER_METHOD_ID =
        jni_interface->GetMethodID(env,
                                   INDEX_INPUT_THREAD_LOCAL_GETTER_CLASS,
                                   "getIndexInputWithBuffer",
                                   "()Lorg/opensearch/knn/index/store/IndexInputWithBuffer;");
    jobject index_input_with_buffer = jni_interface->CallNonvirtualObjectMethodA(
        env, index_input_thread_local_getter, INDEX_INPUT_THREAD_LOCAL_GETTER_CLASS,
        GET_INDEX_INPUT_WITH_BUFFER_METHOD_ID, nullptr);
    return index_input_with_buffer;
  }

  static knn_jni::stream::NativeEngineIndexInputMediator *getIndexInputWithBufferFromThreadLocal() {
    if (mediator) {
      return &(*mediator);
    }
    throw std::runtime_error("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO mediator was empty");
  }

  static jfieldID getIndexInputThreadLocalGetterFieldId(JNIUtilInterface *jni_interface, JNIEnv *env) {
    static jfieldID INDEX_INPUT_THREAD_LOCAL_GETTER_FIELD_ID =
        jni_interface->GetFieldID(env,
                                  getPartialContextClass(jni_interface, env),
                                  "indexInputThreadLocalGetter",
                                  "Lorg/opensearch/knn/index/store/IndexInputThreadLocalGetter;");
    return INDEX_INPUT_THREAD_LOCAL_GETTER_FIELD_ID;
  }

  static jclass getPartialContextClass(JNIUtilInterface *jni_interface, JNIEnv *env) {
    static jclass PARTIAL_CONTEXT_CLASS =
        jni_interface->FindClassFromJNIEnv(env, "org/opensearch/knn/index/util/PartialLoadingContext");
    return PARTIAL_CONTEXT_CLASS;
  }
};  // class PartialLoadingContext



}

#endif //KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_CONTEXT_H_
