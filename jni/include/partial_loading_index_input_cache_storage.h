#ifndef KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_INDEX_INPUT_CACHE_STORAGE_H_
#define KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_INDEX_INPUT_CACHE_STORAGE_H_

#include <optional>

#include <jni.h>
#include <faiss/impl/io.h>
#include "faiss_stream_support.h"
#include "partial_loading_context.h"
#include "parameter_utils.h"
#include "jni_util.h"
#include "memory_util.h"
#include "partial_loading_page_cache.h"

namespace knn_jni::partial_loading {

struct FaissIndexInputCacheStorageBase {
  size_t nitems_;
  size_t base_offset_;
  size_t minimum_nitems_to_load_;
  size_t region_size_bytes_;
  size_t page_size_bytes_;
  size_t num_items_in_page_;
  size_t num_pages_;
  std::unique_ptr<PageHeader[]> pages_;
  std::shared_ptr<MemoryManagementUnitV1> memory_management_unit_;

  FaissIndexInputCacheStorageBase()
      : nitems_(),
        base_offset_(),
        minimum_nitems_to_load_(1) {
  }
};

template<typename T>
struct FaissIndexInputCacheStorage final : FaissIndexInputCacheStorageBase {
  void LoadBlock(faiss::IOReader *io_reader, size_t _nitems) {
    auto index_input_mediator = knn_jni::util::ParameterCheck::require_non_null(
        dynamic_cast<knn_jni::stream::FaissOpenSearchIOReader *>(io_reader),
        "dynamic_cast<FaissOpenSearchIOReader*>(io_reader)")->mediator;
    base_offset_ = index_input_mediator->getOffset();
    region_size_bytes_ = sizeof(T) * _nitems;
    num_pages_ = (region_size_bytes_ + page_size_bytes_ - 1) / page_size_bytes_;
    pages_ = std::make_unique<PageHeader[]>(num_pages_);

    std::cout << "=============== FaissIndexInputCacheStorage::base_offset -> "
              << base_offset_ << std::endl;
    nitems_ = _nitems;
    std::cout << "=============== FaissIndexInputCacheStorage::nitems -> "
              << nitems_ << std::endl;
    index_input_mediator->seek(base_offset_ + sizeof(T) * _nitems);

    std::cout << "================ FaissIndexInpFaissIndexInputCacheStorageutStorage::LoadBlock" << std::endl;
  }

  const T &operator[](const int32_t index) const {
    std::cout << "================ FaissIndexInputCacheStorage::operator[](" << index << ")"
              << ", base_offset=" << base_offset_
              << ", minimum_nitems_to_load=" << minimum_nitems_to_load_
              << ", num_items_in_page_=" << num_items_in_page_
              << ", page_size_bytes_=" << page_size_bytes_
              << ", sizeof(T)=" << sizeof(T)
              << std::endl;

    const auto page_index = index / num_items_in_page_;

    auto loader = [this, page_index](uint8_t *dest) {
      const auto page_offset = page_index * page_size_bytes_;

      // Load bytes from IndexInput
      auto *mediator =
          knn_jni::partial_loading::PartialLoadingContext::getIndexInputWithBufferFromThreadLocal();

      auto start_offset = base_offset_ + page_offset;
      auto end_offset = std::min(start_offset + page_size_bytes_,
                                 mediator->getFileLength());
      auto actual_read_bytes = end_offset - start_offset;

      std::cout << "================ FaissIndexInputCacheStorage::loader"
                << ", page_index=" << page_index
                << ", page_offset=" << page_offset
                << ", page_size_bytes_=" << page_size_bytes_
                << ", base_offset_=" << base_offset_
                << ", start_offset=" << start_offset
                << ", end_offset=" << end_offset
                << ", actual_read_bytes=" << actual_read_bytes
                << ", dest=" << dest
                << std::endl;

      mediator->copyBytesWithOffset(
          base_offset_ + page_offset, actual_read_bytes, dest);
    };

    T *page = pages_[page_index].template loadPage<T>(
        memory_management_unit_.get(),
        page_size_bytes_,
        std::move(loader));
    const auto relative_index = index % num_items_in_page_;
    return page[relative_index];
  }

  void setMinimumItemsToLoad(int32_t minimum_nitems_to_load) {
    minimum_nitems_to_load_ = minimum_nitems_to_load;

    // Determine the page size
    page_size_bytes_ = 16 * 1024;
    const auto atomic_unit_bytes = sizeof(T) * minimum_nitems_to_load_;
    while (page_size_bytes_ < atomic_unit_bytes) {
      page_size_bytes_ *= 2;
    }
    // Ex: atomic_unit_bytes = 1777, page_size_bytes_ = 4096
    // then page_size_bytes_ would shrink down to Ceil(4096 / 1777) * 1777 = 3554
    num_items_in_page_ = page_size_bytes_ / atomic_unit_bytes;
    page_size_bytes_ = num_items_in_page_ * atomic_unit_bytes;

    std::cout << "================ FaissIndexInputCacheStorage::setMinimumItemsToLoad"
              << ", minimum_nitems_to_load_=" << minimum_nitems_to_load_
              << ", page_size_bytes_=" << page_size_bytes_
              << ", atomic_unit_bytes=" << atomic_unit_bytes
              << ", num_items_in_page_=" << num_items_in_page_
              << ", page_size_bytes_=" << page_size_bytes_
              << std::endl;
  }

  size_t size() const {
    return sizeof(T) * nitems_;
  }
};  // FaissIndexInputStorage

}

#endif //KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_INDEX_INPUT_CACHE_STORAGE_H_
