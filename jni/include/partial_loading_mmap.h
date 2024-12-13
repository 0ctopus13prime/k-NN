//
// Created by Kim, Dooyong on 11/29/24.
//

#ifndef KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_MMAP_H_
#define KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_MMAP_H_

#if defined(_WIN32)
  #error "Window platform is not yet supported."
#elif defined(__APPLE__) && defined(__MACH__)
  #include "partial_loading_mmap_darwin.h"
#elif defined(__linux__)
  #include "partial_loading_mmap_linux.h"
#else
  #error "Unrecognizable operating system. Currently we support Darwin, Linux and Windows."
#endif

#endif //KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_MMAP_H_
