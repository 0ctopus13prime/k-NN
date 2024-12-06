//
// Created by Kim, Dooyong on 12/9/24.
//

#ifndef KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_THREAD_LOCALS_H_
#define KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_THREAD_LOCALS_H_

#include <vector>

namespace knn_jni::partial_loading {
  static thread_local std::vector<uint8_t> __partial_loading_buffer {};
}

#endif //KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_THREAD_LOCALS_H_
