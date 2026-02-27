// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

#include "faiss_methods.h"
#include "faiss/index_factory.h"
#include <iostream>

namespace knn_jni {
namespace faiss_wrapper {

faiss::Index* FaissMethods::indexFactory(int d, const char* description, faiss::MetricType metric) {
    return faiss::index_factory(d, description, metric);
}

faiss::IndexBinary* FaissMethods::indexBinaryFactory(int d, const char* description) {
    return faiss::index_binary_factory(d, description);
}

faiss::IndexIDMapTemplate<faiss::Index>* FaissMethods::indexIdMap(faiss::Index* index) {
    return new faiss::IndexIDMap(index);
}

faiss::IndexIDMapTemplate<faiss::IndexBinary>* FaissMethods::indexBinaryIdMap(faiss::IndexBinary* index) {
    return new faiss::IndexBinaryIDMap(index);
}

void FaissMethods::writeIndex(const faiss::Index* idx, faiss::IOWriter* writer, bool skipFlat) {
    constexpr int IO_FLAG_SKIP_STORAGE = 1;
    std::cout << "______________ FaissMethods::writeIndex, skip flag="
              << (skipFlat ? IO_FLAG_SKIP_STORAGE : 0)
              << std::endl;
    faiss::write_index(idx, writer, skipFlat ? IO_FLAG_SKIP_STORAGE : 0);
}

void FaissMethods::writeIndexBinary(const faiss::IndexBinary* idx, faiss::IOWriter* writer) {
    faiss::write_index_binary(idx, writer);
}

} // namespace faiss_wrapper
} // namesapce knn_jni
