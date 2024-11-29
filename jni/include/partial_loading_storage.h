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
#include "partial_loading_mmap.h"

namespace knn_jni::partial_loading {

template<typename T>
class VectorStorage {
 public:
  void LoadBlock(faiss::IOReader *io_reader, size_t nitems) {
//    std::cout << "VectorStorage::LoadBlock, nitems=" << nitems
//              << ", bytes=" << (sizeof(T) * nitems)
//              << ", io_reader=" << io_reader << std::endl;
    vec.resize(nitems);
    io_reader->operator()(vec.data(), sizeof(T), nitems);
  }

  const T &operator[](const int idx) const {
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



template<typename T>
struct MMapStorage {
  MMapStorage()
      : mmaper_(),
        mapped_pointer_holder_(),
        nitems_(),
        minimum_nitems_to_load_(1),
        buffer(1) {
  }

  void LoadBlock(faiss::IOReader *io_reader, size_t _nitems) {
    auto index_input_mediator = knn_jni::util::ParameterCheck::require_non_null(
        dynamic_cast<knn_jni::stream::FaissOpenSearchIOReader *>(io_reader),
        "dynamic_cast<FaissOpenSearchIOReader*>(io_reader)")->mediator;
    const auto offset = index_input_mediator->getOffset();
    std::cout << "------------ MMapStorage offset=" << offset << std::endl;
    const auto num_bytes = sizeof(T) * _nitems;
    std::cout << "------------ MMapStorage num_bytes=" << num_bytes << std::endl;
    mapped_pointer_holder_ = mmaper_->fileMapping(offset, num_bytes);
    std::cout << "------------ MMapStorage fileMapping done. pointer="
              << ((size_t) mapped_pointer_holder_.mapped_pointer_) << std::endl;
    nitems_ = _nitems;
    index_input_mediator->seek(offset + num_bytes);
  }

  const T &operator[](const int32_t index) const {
    // std::cout << "------------ MMapStorage index=" << index
    //           << ", nitems=" << nitems_
    //           << ", sizeof(T)=" << sizeof(T)
    //           << ", pointer=" << ((size_t) mapped_pointer_holder_.calibrated_mapped_pointer_)
    //           << ", minimum_nitems_to_load_=" << minimum_nitems_to_load_
    //           << ", size(buffer)=" << buffer.size()
    //           << std::endl;
    std::memcpy(reinterpret_cast<uint8_t *>(const_cast<T *>(buffer.data())),
                mapped_pointer_holder_.calibrated_mapped_pointer_ + sizeof(T) * index,
                sizeof(T) * minimum_nitems_to_load_);
    return buffer[0];
  }

  void setMinimumItemsToLoad(int32_t nitems) {
    minimum_nitems_to_load_ = nitems;
    buffer.resize(minimum_nitems_to_load_);
  }

  void init(std::shared_ptr<MMaper> &mmaper) {
    mmaper_ = mmaper;
  }

  size_t size() const {
    return nitems_;
  }

  std::shared_ptr<MMaper> mmaper_;
  MMaper::MappedPointerHolder mapped_pointer_holder_;
  size_t nitems_;
  size_t minimum_nitems_to_load_;
  std::vector<T> buffer;
};  // class MMapStorage



}

#endif //KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_STORAGE_H_
