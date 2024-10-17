/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

#ifndef OPENSEARCH_KNN_JNI_NMSLIB_STREAM_SUPPORT_H
#define OPENSEARCH_KNN_JNI_NMSLIB_STREAM_SUPPORT_H

#include "native_engines_stream_support.h"

namespace knn_jni {
namespace stream {

/**
 * NmslibIOReader implementation delegating NativeEngineIndexInputMediator to read bytes.
 */
class NmslibOpenSearchIOReader final : public similarity::NmslibIOReader {
 public:
  explicit NmslibOpenSearchIOReader(NativeEngineIndexInputMediator *_mediator)
      : similarity::NmslibIOReader(),
        mediator(_mediator) {
  }

  void read(char *bytes, size_t len) final {
    if (len > 0) {
      // Mediator calls IndexInput, then copy read bytes to `ptr`.
      mediator->copyBytes(len, (uint8_t *) bytes);
    }
  }

  size_t remainingBytes() final {
    return mediator->remainingBytes();
  }

 private:
  NativeEngineIndexInputMediator *mediator;
};  // class NmslibOpenSearchIOReader


class NmslibOpenSearchIOWriter final : public similarity::NmslibIOWriter {
 public:
  explicit NmslibOpenSearchIOWriter(NativeEngineIndexOutputMediator *_mediator)
      : similarity::NmslibIOWriter(),
        mediator(_mediator) {
  }

  void write(char *bytes, size_t len) final {
    if (len > 0) {
      mediator->writeBytes((uint8_t *) bytes, len);
    }
  }

 private:
  NativeEngineIndexOutputMediator *mediator;
};  // class NmslibOpenSearchIOWriter


}
}

#endif //OPENSEARCH_KNN_JNI_NMSLIB_STREAM_SUPPORT_H
