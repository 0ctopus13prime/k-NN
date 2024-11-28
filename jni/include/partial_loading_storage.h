// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

#ifndef KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_STORAGE_H_
#define KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_STORAGE_H_

#include <faiss/impl/io.h>

namespace knn_jni::partial_loading {

template <typename T>
class VectorStorage {
 public:
  void LoadBlock(faiss::IOReader* io_reader, size_t nitems) {
//    std::cout << "VectorStorage::LoadBlock, nitems=" << nitems
//              << ", bytes=" << (sizeof(T) * nitems)
//              << ", io_reader=" << io_reader << std::endl;
    vec.resize(nitems);
    io_reader->operator()(vec.data(), sizeof(T), nitems);
  }

  const T& operator[](const int idx) const {
    return vec[idx];
  }

  void setMinimumItemsToLoad(int32_t nitems) {
    // No-op
  }

  size_t size() const {
    return vec.size();
  }

 private:
  std::vector<T> vec;
};  // class VectorStorage



template <typename T>
class MMapStorage {
 public:

};  // class MMapStorage



}

#endif //KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_STORAGE_H_
