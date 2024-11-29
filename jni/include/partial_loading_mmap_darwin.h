//
// Created by Kim, Dooyong on 11/29/24.
//

#ifndef KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_MMAP_DARWIN_H_
#define KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_MMAP_DARWIN_H_

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace knn_jni::partial_loading {

struct MMaper {
  struct MappedPointerHolder {
    MappedPointerHolder()
        : mapped_pointer_(),
          calibrated_mapped_pointer_(),
          map_size_() {
    }

    MappedPointerHolder(void *mapped_pointer, size_t calibrated, size_t map_size)
        : mapped_pointer_(reinterpret_cast<uint8_t *>(mapped_pointer)),
          calibrated_mapped_pointer_(reinterpret_cast<uint8_t *>(mapped_pointer) + calibrated),
          map_size_(map_size) {
    }

    MappedPointerHolder(MappedPointerHolder &&other) noexcept
        : mapped_pointer_(other.mapped_pointer_),
          calibrated_mapped_pointer_(other.calibrated_mapped_pointer_),
          map_size_(other.map_size_) {
      other.reset();
    }

    MappedPointerHolder &operator=(MappedPointerHolder &&other) noexcept {
      if (this != &other) {
        mapped_pointer_ = other.mapped_pointer_;
        calibrated_mapped_pointer_ = other.calibrated_mapped_pointer_;
        map_size_ = other.map_size_;
        other.reset();
      }

      return *this;
    }

    ~MappedPointerHolder() {
      if (mapped_pointer_) {
        std::cout << "--------------------- ~MappedPointerHolder() 111 "
                  << " mapped_pointer_=" << ((size_t) mapped_pointer_)
                  << std::endl;
        munmap(mapped_pointer_, map_size_);
        mapped_pointer_ = nullptr;
        calibrated_mapped_pointer_ = nullptr;
        map_size_ = 0;
      }

      std::cout << "--------------------- ~MappedPointerHolder() 222" << std::endl;
    }

    void reset() {
      mapped_pointer_ = nullptr;
      calibrated_mapped_pointer_ = nullptr;
      map_size_ = 0;
    }

    uint8_t *mapped_pointer_;
    uint8_t *calibrated_mapped_pointer_;
    size_t map_size_;
  };

  explicit MMaper(std::string file_path)
      : file_path_(std::move(file_path)) {
  }

  MappedPointerHolder fileMapping(size_t offset, int64_t map_size) {
    static const auto PAGE_SIZE = sysconf(_SC_PAGESIZE);

    if (map_size == 0) {
      throw std::runtime_error("Cannot do mapping with zero map_size.");
    }

    // Open the file
    int fd = open(file_path_.c_str(), O_RDONLY); // Read-write access
    if (fd == -1) {
      throw std::runtime_error("Failed to open [" + file_path_ + "]. Error=[" + strerror(errno) + ']');
    }

    const size_t file_size = getFileSize(fd); // Size of the file or region to map
    size_t actual_map_size = map_size > 0 ? map_size : file_size;
    if ((offset + actual_map_size) > file_size) {
      throw std::runtime_error("Mapping size would beyond the range of file size. "
                               "Offset=" + std::to_string(file_size)
                                   + ", map_size=" + std::to_string(actual_map_size));
    }
    const auto normalized_offset = (offset / PAGE_SIZE) * PAGE_SIZE;
    const auto calibrated = offset - normalized_offset;

    // Map the file into memory
    void *mapped = mmap(nullptr, actual_map_size + calibrated, PROT_READ, MAP_PRIVATE, fd, normalized_offset);
    if (mapped == MAP_FAILED) {
      close(fd);
      throw std::runtime_error("Failed to call mmap with [" + file_path_
                                   + "]. Error=[" + strerror(errno) + ']');
    }

    // We can safely close the file after mmap.
    close(fd);

    return MappedPointerHolder(mapped, calibrated, actual_map_size);
  }

  size_t getFileSize(int file_descriptor) {
    struct stat file_stat;

    // Get file status
    if (fstat(file_descriptor, &file_stat) != -1) {
      return static_cast<size_t>(file_stat.st_size);
    }

    throw std::runtime_error("Failed fstat with [" + file_path_
                                 + "]. Error=[" + strerror(errno) + ']');
  }

  std::string file_path_;
};

}

#endif //KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_MMAP_DARWIN_H_
