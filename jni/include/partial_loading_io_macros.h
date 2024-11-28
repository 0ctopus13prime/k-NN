//
// Created by Kim, Dooyong on 11/14/24.
//

#ifndef KNNPLUGIN_JNI_INCLUDE_FAISS_IO_MACROS_H_
#define KNNPLUGIN_JNI_INCLUDE_FAISS_IO_MACROS_H_

#include <faiss/impl/io_macros.h>


#define OS_READXBVECTOR(bytes_accessor)                               \
    {                                                                 \
        size_t size;                                                  \
        READANDCHECK(&size, 1);                                       \
        FAISS_THROW_IF_NOT(size >= 0 && size < (uint64_t{1} << 40));  \
        (bytes_accessor).LoadBlock(f, 4 * size);                      \
    }

#define OS_READVECTOR(bytes_accessor)                                \
    {                                                                \
        size_t size;                                                 \
        READANDCHECK(&size, 1);                                      \
        FAISS_THROW_IF_NOT(size >= 0 && size < (1ULL << 40U));       \
        (bytes_accessor).LoadBlock(f, size);                         \
    }

#endif //KNNPLUGIN_JNI_INCLUDE_FAISS_IO_MACROS_H_
