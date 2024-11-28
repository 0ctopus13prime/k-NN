//
// Created by Kim, Dooyong on 11/26/24.
//

#ifndef KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_INDEX_INPUT_STORAGE_H_
#define KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_INDEX_INPUT_STORAGE_H_

#include <optional>

#include <jni.h>
#include <faiss/impl/io.h>
#include "faiss_stream_support.h"
#include "partial_loading_context.h"
#include "parameter_utils.h"
#include "jni_util.h"
#include "memory_util.h"

namespace knn_jni::partial_loading {

struct FaissIndexInputStorageBase {
  size_t nitems_;
  size_t base_offset_;
  size_t minimum_nitems_to_load_;
  int32_t previous_index_;

  FaissIndexInputStorageBase()
      : nitems_(),
        base_offset_(),
        minimum_nitems_to_load_(1),
        previous_index_(-1) {
  }
};



template<typename T>
struct FaissIndexInputStorage final : FaissIndexInputStorageBase {
  std::vector<T> buffer;

  FaissIndexInputStorage()
      : FaissIndexInputStorageBase(),
        buffer(FaissIndexInputStorageBase::minimum_nitems_to_load_) {
  }

  void LoadBlock(faiss::IOReader *io_reader, size_t _nitems) {
    auto index_input_mediator = knn_jni::util::ParameterCheck::require_non_null(
        dynamic_cast<knn_jni::stream::FaissOpenSearchIOReader *>(io_reader),
        "dynamic_cast<FaissOpenSearchIOReader*>(io_reader)")->mediator;
    base_offset_ = index_input_mediator->getOffset();
//    std::cout << "=============== FaissIndexInputStorage::base_offset -> "
//              << base_offset_ << std::endl;
    nitems_ = _nitems;
//    std::cout << "=============== FaissIndexInputStorage::nitems -> "
//              << nitems_ << std::endl;
    index_input_mediator->seek(base_offset_ + sizeof(T) * _nitems);

//    std::cout << "================ FaissIndexInputStorage::LoadBlock" << std::endl;
  }

  const T &operator[](const int32_t index) const {
//    std::cout << "================ FaissIndexInputStorage::operator[](" << index << ")"
//              << ", base_offset=" << base_offset_
//              << ", previous_index=" << previous_index_
//              << ", minimum_nitems_to_load=" << minimum_nitems_to_load_
//              << ", buffer.size()=" << buffer.size()
//              << ", sizeof(T)=" << sizeof(T)
//              << std::endl;
    if (index < previous_index_ || index >= (previous_index_ + minimum_nitems_to_load_)) {
      // index is not in [previous_index, previous_index + minimum_nitems_to_load)
      // We reload bytes from IndexInput.
      const_cast<FaissIndexInputStorage *>(this)->reloadBytes(index);
    }

    return buffer[index - previous_index_];
  }

  void reloadBytes(const size_t index) {
    auto* mediator =
        knn_jni::partial_loading::PartialLoadingContext::getIndexInputWithBufferFromThreadLocal();
    mediator->copyBytesWithOffset(
        base_offset_ + (sizeof(T) * index), sizeof(T) * minimum_nitems_to_load_, (uint8_t *) buffer.data());
    previous_index_ = index;
  }

  void setMinimumItemsToLoad(int32_t nitems) {
    minimum_nitems_to_load_ = nitems;
    buffer.resize(minimum_nitems_to_load_);
  }

  size_t size() const {
    return sizeof(T) * nitems_;
  }
};  // FaissIndexInputStorage



}

#endif //KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_INDEX_INPUT_STORAGE_H_
