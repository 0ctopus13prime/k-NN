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
#include "partial_loading_thread_locals.h"
#include "partial_loading_macros.h"

namespace knn_jni::partial_loading {

template<typename T>
class VectorStorage {
 public:
  void LoadBlock(faiss::IOReader *io_reader, size_t nitems) {
#ifdef PARTIAL_LOADING_COUT
    std::cout << "VectorStorage::LoadBlock, nitems=" << nitems
              << ", bytes=" << (sizeof(T) * nitems)
              << ", io_reader=" << io_reader << std::endl;
#endif
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
        minimum_nitems_to_load_(1) {
  }

  void LoadBlock(faiss::IOReader *io_reader, size_t _nitems) {
    auto index_input_mediator = knn_jni::util::ParameterCheck::require_non_null(
        dynamic_cast<knn_jni::stream::FaissOpenSearchIOReader *>(io_reader),
        "dynamic_cast<FaissOpenSearchIOReader*>(io_reader)")->mediator;
    const auto offset = index_input_mediator->getOffset();
#ifdef PARTIAL_LOADING_COUT
    std::cout << "------------ MMapStorage offset=" << offset << std::endl;
#endif
    const auto num_bytes = sizeof(T) * _nitems;
#ifdef PARTIAL_LOADING_COUT
    std::cout << "------------ MMapStorage num_bytes=" << num_bytes << std::endl;
#endif
    mapped_pointer_holder_ = mmaper_->fileMapping(offset, num_bytes);
#ifdef PARTIAL_LOADING_COUT
    std::cout << "------------ MMapStorage fileMapping done. pointer="
              << ((size_t) mapped_pointer_holder_.mapped_pointer_) << std::endl;
#endif
    nitems_ = _nitems;
    index_input_mediator->seek(offset + num_bytes);
  }

  const T &operator[](const size_t index) const {
#ifdef PARTIAL_LOADING_COUT
    std::cout << "------------ MMapStorage index=" << index
              << ", nitems=" << nitems_
              << ", sizeof(T)=" << sizeof(T)
              << ", pointer=" << ((size_t) mapped_pointer_holder_.calibrated_mapped_pointer_)
              << ", minimum_nitems_to_load_=" << minimum_nitems_to_load_
              << ", __partial_loading_buffer.size()=" << __partial_loading_buffer.size()
              << std::endl;
#endif
    if (__partial_loading_buffer.size() >= (16 * 1024)) {
      __partial_loading_buffer.clear();
    }

    const auto old_size = __partial_loading_buffer.size();
    __partial_loading_buffer.resize(__partial_loading_buffer.size() + sizeof(T) * minimum_nitems_to_load_);
    auto ret = &__partial_loading_buffer[old_size];

    std::memcpy(ret,
                mapped_pointer_holder_.calibrated_mapped_pointer_ + sizeof(T) * index,
                sizeof(T) * minimum_nitems_to_load_);
    return *reinterpret_cast<const T *>(ret);
  }

  void setMinimumItemsToLoad(int32_t nitems) {
    minimum_nitems_to_load_ = nitems;
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
};  // class MMapStorage



}

#endif //KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_STORAGE_H_
