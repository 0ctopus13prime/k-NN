//
// Created by Kim, Dooyong on 12/5/24.
//

#ifndef KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_PAGE_CACHE_H_
#define KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_PAGE_CACHE_H_

#include <cstdint>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>
#include <atomic>
#include <functional>
#include <mutex>
#include <set>
#include <thread>
#include "partial_loading_macros.h"

namespace knn_jni::partial_loading {

std::atomic<uint64_t> __storage_version = 0;
std::mutex __storage_version_set_lock;
std::set<uint64_t> __storage_version_set;

struct PageHeader;

thread_local uint64_t __current_storage_version = 0;

struct StoragePageCacheCleaner {
  uint64_t version_;

  explicit StoragePageCacheCleaner(uint64_t version)
      : version_(version) {
  }

  ~StoragePageCacheCleaner() {
    std::lock_guard<decltype(__storage_version_set_lock)> lock(__storage_version_set_lock);
    __storage_version_set.erase(version_);
  }

  static StoragePageCacheCleaner updateStorageVersionAndGet() noexcept {
    __current_storage_version = __storage_version.fetch_add(1, std::memory_order_acq_rel);
    {
      std::lock_guard<decltype(__storage_version_set_lock)> lock(__storage_version_set_lock);
      __storage_version_set.insert(__current_storage_version);
    }
    return StoragePageCacheCleaner{__current_storage_version};
  }

  static uint64_t getMinimumStorageVersion() noexcept {
    std::lock_guard<decltype(__storage_version_set_lock)> lock(__storage_version_set_lock);
    if (__storage_version_set.size()) {
      return *__storage_version_set.begin();
    }
    return std::numeric_limits<uint64_t>::max();
  }
};

struct PageList {
  std::atomic<PageList *> next_;
  std::atomic<uint64_t> *version_;
  std::atomic<PageList *> *page_reference_;
  std::vector<uint8_t> page_;

  PageList(std::atomic<uint64_t> *version,
           std::atomic<PageList *> *page_reference,
           const uint64_t page_size)
      : next_(),
        version_(version),
        page_reference_(page_reference),
        page_(page_size) {
  }
};

struct MemoryManagementUnitV1;

struct PageHeader {
  std::atomic<uint64_t> version_;
  std::atomic<PageList *> page_;
  std::mutex update_lock_;

  PageHeader()
      : version_(),
        page_(),
        update_lock_() {
  }

  template<typename T>
  T *loadPage(MemoryManagementUnitV1 *memory_management_unit_v1,
              uint64_t page_size,
              std::function<void(uint8_t *)> loader);
};

struct MemoryManagementUnitV1 {
  uint64_t max_memory_bytes_;
  std::atomic<uint64_t> memory_used_bytes_;
  PageList page_list_head_;
  PageList sheol_list_head_;
  std::atomic<uint32_t> gc_thread_semaphore_;

  explicit MemoryManagementUnitV1(const uint64_t max_memory_bytes)
      : max_memory_bytes_(max_memory_bytes),
        memory_used_bytes_(),
        page_list_head_(nullptr, nullptr, 0),
        sheol_list_head_(nullptr, nullptr, 0),
        gc_thread_semaphore_() {
  }

  void *getPage(PageHeader *page_header,
                const uint64_t page_size,
                std::function<void(uint8_t *)> loader) {
    auto acquired_page = page_header->page_.load(std::memory_order_acquire);
    if (acquired_page) {
      return acquired_page->page_.data();
    } else {
      // Page fault
      return loadNewPage(page_header, page_size, std::move(loader));
    }  // End if
  }

  uint64_t tryCleanUpPage(PageList *page, const uint64_t min_version) {
    if (page->version_->load(std::memory_order_acquire) < min_version) {
      const auto reclaimed = page->page_.size();
      memory_used_bytes_.fetch_sub(reclaimed, std::memory_order_acq_rel);
      delete page;
      return reclaimed;
    } else {
      updateNewPage(&sheol_list_head_, page);
      return 0;
    }
  }

  void detachPage(PageList *prev, PageList *current) {
    prev->next_.store(current->next_);
    current->page_reference_->store(nullptr, std::memory_order_release);
  }

  uint64_t garbageCollectList(PageList *base_node, const uint64_t min_version) {
    if (base_node == nullptr) {
      return 0;
    }

    uint64_t reclaimed = 0;
    auto current = base_node->next_.load(std::memory_order_acquire);
    PageList *prev = base_node;
    while (current) {
      auto next = current->next_.load(std::memory_order_acquire);
      if (current->version_->load(std::memory_order_acquire) < min_version) {
        detachPage(prev, current);
        reclaimed += tryCleanUpPage(current, min_version);
        current = next;
        continue;
      }
      prev = current;
      current = next;
    }

    return reclaimed;
  }

  void signalGarbageCollect() {
    const auto num = gc_thread_semaphore_.fetch_add(1, std::memory_order_acq_rel);
    if (num == 0) {
      std::thread gc_thread([this]() {
        auto start = std::chrono::high_resolution_clock::now();

        uint64_t reclaimed = 0;
        int32_t max_tries = 3;
        while ((max_tries--) > 0 && memory_used_bytes_.load(std::memory_order_acquire) > max_memory_bytes_) {
          const auto min_version = StoragePageCacheCleaner::getMinimumStorageVersion();
          reclaimed += garbageCollectList(&sheol_list_head_, min_version);
          reclaimed += garbageCollectList(
              page_list_head_.next_.load(std::memory_order_acquire), min_version);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "[GC] reclaimed: " << reclaimed
                  << ", took: " << duration.count()
                  << ", curr mem: " << memory_used_bytes_.load(std::memory_order_acquire)
                  << std::endl;

        // Clean up active list
        gc_thread_semaphore_.fetch_sub(1);
      });

      gc_thread.detach();
      return;
    }

    gc_thread_semaphore_.fetch_sub(1);
  }

  void updateNewPage(PageList *head, PageList *new_node) {
    PageList *old_header;
    do {
      old_header = head->next_.load(std::memory_order_acquire);
      new_node->next_.store(old_header, std::memory_order_release);
    } while (!head->next_.compare_exchange_weak(old_header, new_node, std::memory_order_acq_rel));
  }

  void *loadNewPage(PageHeader *page_header,
                    const uint64_t page_size,
                    std::function<void(uint8_t *)> loader) {
#ifdef PARTIAL_LOADING_COUT
    std::cout << "*************** MemoryManagementUnitV1::loadNewPage"
              << ", page_size=" << page_size
              << ", memory_bytes_using=" << memory_used_bytes_.fetch_add(page_size, std::memory_order_acquire)
              << ", free_bytes="
              << (((int64_t) max_memory_bytes_) - (int64_t) memory_used_bytes_.fetch_add(page_size, std::memory_order_acquire))
              << std::endl;
#endif

    PageList *list;
    {
      std::lock_guard<decltype(page_header->update_lock_)> guard{page_header->update_lock_};
      if (auto existing_page = page_header->page_.load(std::memory_order_acquire)) {
        // We have it already.
        return existing_page->page_.data();
      }

      // Increase memory gauge
      const auto memory_bytes_using = memory_used_bytes_.fetch_add(page_size, std::memory_order_acq_rel) + page_size;
      if (memory_bytes_using > max_memory_bytes_) {
        signalGarbageCollect();
      }

      list = new PageList(&(page_header->version_), &(page_header->page_), page_size);
      loader(list->page_.data());
      page_header->page_.store(list, std::memory_order_release);
      updateNewPage(&page_list_head_, list);
    }

    return list->page_.data();
  }
};

template<typename T>
T *PageHeader::loadPage(MemoryManagementUnitV1 *memory_management_unit_v1,
                        const uint64_t page_size,
                        std::function<void(uint8_t *)> loader) {
#ifdef PARTIAL_LOADING_COUT
  std::cout << "*************** PageHeader::loadPage, this="
            << ((uint64_t) this)
            << ", page_size=" << page_size
            << ", memory_management_unit_v1=" << memory_management_unit_v1
            << std::endl;
#endif

  // Update version
  uint64_t expected = version_.load(std::memory_order_acquire);
  while (__current_storage_version > expected
      && !version_.compare_exchange_weak(expected, __current_storage_version));

  // Try to get a page
  void *acquired_page = memory_management_unit_v1->getPage(this, page_size, std::move(loader));

#ifdef PARTIAL_LOADING_COUT
  std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
#endif
  return (T *) acquired_page;
}

}

#endif //KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_PAGE_CACHE_H_
