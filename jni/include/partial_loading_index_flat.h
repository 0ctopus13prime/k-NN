//
// Created by Kim, Dooyong on 11/14/24.
//

#ifndef KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_INDEX_FLAT_H_
#define KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_INDEX_FLAT_H_

#include "partial_loading_index_flat_codes.h"
#include <faiss/Index.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/utils/prefetch.h>
#include <vector>
#include "partial_loading_macros.h"

namespace faiss {

template<template<class> typename Storage>
struct OpenSearchFlatL2Dis : DistanceComputer {
  const Storage<uint8_t> *codes_data;
  size_t code_size;
  size_t d;
  idx_t nb;
  const float *queryVector;
  size_t ndis;

  OpenSearchFlatL2Dis(const Storage<uint8_t> *_codes_data,
                      size_t _code_size,
                      size_t _d,
                      idx_t _ntotal)
      : DistanceComputer(),
        codes_data(_codes_data),
        code_size(_code_size),
        d(_d),
        nb(_ntotal),
        queryVector(),
        ndis() {
  }

  float operator()(idx_t i) final {
    ndis++;
#ifdef PARTIAL_LOADING_COUT
    std::cout << "====================== OpenSearchFlatL2Dis "
              << ", i=" << i
              << ", type(Storage)" << typeid(Storage<uint8_t>).name()
              << ", codes_data=" << ((uint64_t) codes_data)
              << ", code_size=" << code_size
              << ", d=" << d
              << std::endl;
#endif
    auto &vec = (*codes_data)[i * code_size];
    const float *vec_pointer = (float *) &vec;
    return fvec_L2sqr(queryVector,
                      vec_pointer,
                      d);
  }

  void set_query(const float *_queryVector) final {
    queryVector = _queryVector;
  }

  float symmetric_dis(idx_t i, idx_t j) final {
  }

  // compute four distances
  void distances_batch_4(
      const idx_t idx0,
      const idx_t idx1,
      const idx_t idx2,
      const idx_t idx3,
      float &dis0,
      float &dis1,
      float &dis2,
      float &dis3) final {
#ifdef PARTIAL_LOADING_COUT
    std::cout << "====================== OpenSearchFlatL2Dis"
              << ", type(Storage)" << typeid(Storage<uint8_t>).name()
              << ", idx0=" << idx0
              << ", idx1=" << idx1
              << ", idx2=" << idx2
              << ", idx3=" << idx3
              << ", codes_data=" << ((uint64_t) codes_data)
              << ", code_size=" << code_size
              << ", d=" << d
              << std::endl;
#endif

    ndis += 4;

    // compute first, assign next
    const auto *__restrict y0 =
        reinterpret_cast<const float *>(&(*codes_data)[idx0 * code_size]);
    const auto *__restrict y1 =
        reinterpret_cast<const float *>(&(*codes_data)[idx1 * code_size]);
    const auto *__restrict y2 =
        reinterpret_cast<const float *>(&(*codes_data)[idx2 * code_size]);
    const auto *__restrict y3 =
        reinterpret_cast<const float *>(&(*codes_data)[idx3 * code_size]);

    float dp0 = 0;
    float dp1 = 0;
    float dp2 = 0;
    float dp3 = 0;
    fvec_L2sqr_batch_4(queryVector,
                       y0, y1, y2, y3,
                       d,
                       dp0, dp1, dp2, dp3);
    dis0 = dp0;
    dis1 = dp1;
    dis2 = dp2;
    dis3 = dp3;
  }
};



//////////////////////////////
// OpenSearchIndexFlat
//////////////////////////////

template<template<class> typename Storage>
struct OpenSearchIndexFlat : Index {
  size_t code_size;
  Storage<uint8_t> codes;

  DistanceComputer *get_distance_computer() const override {
    return new OpenSearchFlatL2Dis<Storage>(
        &codes,
        code_size,
        Index::d,
        Index::ntotal);
  }

  void search(
      idx_t n,
      const float *x,
      idx_t k,
      float *distances,
      idx_t *labels,
      const SearchParameters *params = nullptr) const override {
  }

  void range_search(
      idx_t n,
      const float *x,
      float radius,
      RangeSearchResult *result,
      const SearchParameters *params = nullptr) const override {
  }

  void reconstruct(idx_t key, float *recons) const override {
  }

  void add(idx_t n, const float *x) final {
  }

  void reset() final {
  }

  OpenSearchIndexFlat() = default;
};

template<template<class> typename Storage>
struct OpenSearchIndexFlatL2 : OpenSearchIndexFlat<Storage> {
  // Special cache for L2 norms.
  // If this cache is set, then get_distance_computer() returns
  // a special version that computes the distance using dot products
  // and l2 norms.
  std::vector<float> cached_l2norms;

  OpenSearchIndexFlatL2() = default;
};

}

#endif //KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_INDEX_FLAT_H_
