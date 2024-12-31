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
  std::atomic<uint8_t *> *page_;
  std::vector<uint8_t> page_holder_;
  uint64_t version_snapshot_;

  PageList(std::atomic<uint64_t> *version,
           std::atomic<uint8_t *> *page_reference,
           const uint64_t page_size)
      : next_(),
        version_(version),
        page_(page_reference),
        page_holder_(page_size) {
    page_->store(page_holder_.data());
  }

  PageList() = default;
};

struct MemoryManagementUnitV1;

struct PageHeader {
  std::atomic<uint64_t> version_;
  std::atomic<uint8_t *> page_;
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
        page_list_head_(),
        sheol_list_head_(),
        gc_thread_semaphore_() {
  }

  void *getPage(PageHeader *page_header,
                const uint64_t page_size,
                std::function<void(uint8_t *)> loader) {
    auto *acquired_page = page_header->page_.load(std::memory_order_acquire);
    if (acquired_page) {
      return acquired_page;
    }  // End if

    // Page fault
    return loadNewPage(page_header, page_size, std::move(loader));
  }

  static void appendTo(PageList *target, PageList *to_append) {
    auto *tmp = target->next_.load(std::memory_order_acquire);
    target->next_.store(to_append, std::memory_order_release);
    to_append->next_.store(tmp, std::memory_order_release);
  }

  static void collectCandidate(PageList &candidate,
                               int32_t &num_candidate,
                               PageList *base_node,
                               const uint64_t min_version,
                               PageList **last_one) {
    if (last_one) {
      *last_one = base_node;
    }

    if (base_node == nullptr) {
      return;
    }

    auto *current = base_node->next_.load(std::memory_order_acquire);
    PageList *prev = base_node;
    while (current) {
      // Remember next first
      auto *next = current->next_.load(std::memory_order_acquire);

      const auto current_version = current->version_->load(std::memory_order_acquire);
      if (current_version < min_version) {
        // Found a target whose version < min_version

        // Detach the target from the list.
        prev->next_.store(next, std::memory_order_release);

        // Add the detached one to the candidate list.
        current->version_snapshot_ = current_version;
        ++num_candidate;
        appendTo(&candidate, current);

        current = next;
        continue;
      }

      if (last_one) {
        *last_one = current;
      }

      prev = current;
      current = next;
    }  // End while
  }

  void finalizeTargets(PageList &candidate, PageList *last_one) {
    // Let's make it 50% of MAX_CAP.
    constexpr double kTarget = 0.50;
    const auto gc_target_bytes =
        (uint64_t) (((double) memory_used_bytes_.load(std::memory_order_acquire))
            - (kTarget * ((double) max_memory_bytes_)));

    struct MaxPageVersionComparator {
      bool operator()(const PageList *lhs, const PageList *rhs) const noexcept {
        return lhs->version_snapshot_ < rhs->version_snapshot_;
      }
    };

    // Max version heap
    std::priority_queue<PageList *, std::vector<PageList *>, MaxPageVersionComparator> pq;

    uint64_t estimated_reclaim = 0;
    for (PageList *target = candidate.next_.load(std::memory_order_acquire); target != nullptr;) {
      auto *next = target->next_.load(std::memory_order_acquire);

      if ((estimated_reclaim + target->page_holder_.size()) < gc_target_bytes) {
        pq.push(target);
        estimated_reclaim += target->page_holder_.size();
      } else if (target->version_snapshot_ < pq.top()->version_snapshot_) {
        pq.push(target);
        estimated_reclaim += target->page_holder_.size();

        do {
          const auto after_removal_reclaim =
              estimated_reclaim - pq.top()->page_holder_.size();
          if (after_removal_reclaim <= gc_target_bytes) {
            break;
          }

          // Give a second chance and put it back to list.
          auto *evicted = pq.top();
          appendTo(last_one, evicted);

          // Do the eviction.
          estimated_reclaim -= evicted->page_holder_.size();
          pq.pop();
        } while (!pq.empty());  // End while
      } else {
        // Append it back to the list.
        appendTo(last_one, target);
      }  // End if

      target = next;
    }  // End for

    // Append final targets to candidate
    candidate.next_.store(nullptr);
    while (!pq.empty()) {
      appendTo(&candidate, pq.top());
      pq.pop();
    }
  }

  uint64_t cleanUpTargets(PageList *target) {
    uint64_t reclaimed = 0;

    while (target) {
      // Remember next
      auto *next = target->next_.load(std::memory_order_acquire);

      // Try to reclaim
      const auto version = target->version_->load(std::memory_order_acquire);
      target->page_->store(nullptr, std::memory_order_release);
      if (target->version_->load(std::memory_order_acquire) != version) {
        // It was accessed during the detaching. Put it in sheol list.
        appendTo(&sheol_list_head_, target);
      } else {
        // It's safe to reclaim
        reclaimed += target->page_holder_.size();
        delete target;
      }

      target = next;
    }  // End for

    memory_used_bytes_.fetch_sub(reclaimed);
    return reclaimed;
  }

  void signalGarbageCollect() {
    const auto num = gc_thread_semaphore_.fetch_add(1, std::memory_order_acq_rel);
    if (num == 0) {
      std::thread gc_thread([this]() {
        auto start = std::chrono::high_resolution_clock::now();

        // 1. Collect a candidate.
        PageList candidate{};
        int32_t num_candidate = 0;
        const auto min_version = StoragePageCacheCleaner::getMinimumStorageVersion();
        collectCandidate(candidate, num_candidate, &sheol_list_head_, min_version, nullptr);

        PageList *last_one = nullptr;
        collectCandidate(candidate,
                         num_candidate,
                         page_list_head_.next_.load(std::memory_order_acquire),
                         min_version,
                         &last_one);

        // 2. Finalize the targets.
        finalizeTargets(candidate, last_one);

        // 3. Try to clean up targets.
        const uint64_t reclaimed = cleanUpTargets(candidate.next_.load(std::memory_order_acquire));

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "[GC] reclaimed: " << reclaimed
                  << ", took: " << duration.count()
                  << ", num_candidate: " << num_candidate
                  << ", curr mem: " << memory_used_bytes_.load(std::memory_order_acquire)
                  << std::endl;

        // Clean up active list
        gc_thread_semaphore_.fetch_sub(1);
      });

      gc_thread.detach();
      return;
    }

    gc_thread_semaphore_.fetch_sub(1, std::memory_order_acq_rel);
  }

  static void updateNewPage(PageList *head, PageList *new_node) {
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
      if (auto *existing_page = page_header->page_.load(std::memory_order_acquire)) {
        // We have it already.
        return existing_page;
      }

      // Increase memory gauge
      const auto memory_bytes_using =
          memory_used_bytes_.fetch_add(page_size, std::memory_order_acq_rel) + page_size;
      if (memory_bytes_using > max_memory_bytes_) {
        signalGarbageCollect();
      }

      list = new PageList(&(page_header->version_), &(page_header->page_), page_size);
      loader(list->page_holder_.data());
      updateNewPage(&page_list_head_, list);
    }

    return list->page_holder_.data();
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
