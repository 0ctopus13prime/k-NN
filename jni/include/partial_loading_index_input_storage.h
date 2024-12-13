//
// Created by Kim, Dooyong on 11/26/24.
//

#ifndef KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_INDEX_INPUT_STORAGE_H_
#define KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_INDEX_INPUT_STORAGE_H_

#include <optional>

#include <jni.h>
#include <faiss/impl/io.h>
#include "faiss_stream_support.h"
#include "partial_loading_thread_locals.h"
#include "partial_loading_context.h"
#include "partial_loading_macros.h"
#include "parameter_utils.h"
#include "jni_util.h"
#include "memory_util.h"

namespace knn_jni::partial_loading {

struct FaissIndexInputStorageBase {
  size_t nitems_;
  size_t base_offset_;
  size_t minimum_nitems_to_load_;

  FaissIndexInputStorageBase()
      : nitems_(),
        base_offset_(),
        minimum_nitems_to_load_(1) {
  }
};

template<typename T>
struct FaissIndexInputStorage final : FaissIndexInputStorageBase {
  FaissIndexInputStorage()
      : FaissIndexInputStorageBase() {
  }

  void LoadBlock(faiss::IOReader *io_reader, size_t _nitems) {
    auto index_input_mediator = knn_jni::util::ParameterCheck::require_non_null(
        dynamic_cast<knn_jni::stream::FaissOpenSearchIOReader *>(io_reader),
        "dynamic_cast<FaissOpenSearchIOReader*>(io_reader)")->mediator;
    base_offset_ = index_input_mediator->getOffset();
#ifdef PARTIAL_LOADING_COUT
    std::cout << "=============== FaissIndexInputStorage::base_offset -> "
              << base_offset_ << std::endl;
#endif
    nitems_ = _nitems;

#ifdef PARTIAL_LOADING_COUT
    std::cout << "=============== FaissIndexInputStorage::nitems -> "
              << nitems_ << std::endl;
#endif
    index_input_mediator->seek(base_offset_ + sizeof(T) * _nitems);

#ifdef PARTIAL_LOADING_COUT
    std::cout << "================ FaissIndexInputStorage::LoadBlock" << std::endl;
#endif
  }

  const T &operator[](const size_t index) const {
#ifdef PARTIAL_LOADING_COUT
    std::cout << "================ FaissIndexInputStorage::operator[](" << index << ")"
              << ", base_offset=" << base_offset_
              << ", minimum_nitems_to_load=" << minimum_nitems_to_load_
              << ", __partial_loading_buffer.size()=" << __partial_loading_buffer.size()
              << ", sizeof(T)=" << sizeof(T)
              << std::endl;
#endif

    return *const_cast<FaissIndexInputStorage *>(this)->reloadBytes(index);
  }

  T *reloadBytes(const size_t index) {
    if (__partial_loading_buffer.size() >= (16 * 1024)) {
      __partial_loading_buffer.clear();
    }

    const auto old_size = __partial_loading_buffer.size();
    __partial_loading_buffer.resize(
        __partial_loading_buffer.size() + sizeof(T) * minimum_nitems_to_load_);
    auto ret = &__partial_loading_buffer[old_size];

    auto *mediator =
        knn_jni::partial_loading::PartialLoadingContext::getIndexInputWithBufferFromThreadLocal();

#ifdef PARTIAL_LOADING_COUT
    std::cout << "========== buffer.size=" << __partial_loading_buffer.size()
              << ", index=" << index
              << ", offset=" << (base_offset_ + (sizeof(T) * index))
              << ", region_size=" << (sizeof(T) * nitems_)
              << std::endl;
#endif

    mediator->copyBytesWithOffset(
        base_offset_ + (sizeof(T) * index),
        sizeof(T) * minimum_nitems_to_load_,
        ret);

    return (T *) ret;
  }

  void setMinimumItemsToLoad(int32_t nitems) {
    minimum_nitems_to_load_ = nitems;
  }

  size_t size() const {
    return sizeof(T) * nitems_;
  }
};  // FaissIndexInputStorage



}

#endif //KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_INDEX_INPUT_STORAGE_H_
