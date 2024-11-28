//
// Created by Kim, Dooyong on 11/14/24.
//

#ifndef KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_INDEX_FLAT_CODES_H_
#define KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_INDEX_FLAT_CODES_H_

#include <vector>

#include <faiss/Index.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/IndexFlatCodes.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/CodePacker.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/extra_distances.h>

namespace faiss {

struct CodePacker;

template<template<class> typename Storage>
struct OpenSearchIndexFlatCodes : Index {
  size_t code_size;
  Storage<uint8_t> codes;

  OpenSearchIndexFlatCodes()
      : Index(), code_size(), codes() {
  }

  void add(idx_t n, const float *x) override {
  }

  void reset() override {
  }

  void reconstruct_n(idx_t i0, idx_t ni, float *recons) const override {
  }

  void reconstruct(idx_t key, float *recons) const override {
  }

  size_t sa_code_size() const override {
    return code_size;
  }

  /** remove some ids. NB that because of the structure of the
   * index, the semantics of this operation are
   * different from the usual ones: the new ids are shifted */
  size_t remove_ids(const IDSelector &sel) override {
  }

  /** a FlatCodesDistanceComputer offers a distance_to_code method
   *
   * The default implementation explicitly decodes the vector with sa_decode.
   */
  virtual FlatCodesDistanceComputer *get_FlatCodesDistanceComputer() const;

  DistanceComputer *get_distance_computer() const override {
    return get_FlatCodesDistanceComputer();
  }

  /** Search implemented by decoding */
  void search(
      idx_t n,
      const float *x,
      idx_t k,
      float *distances,
      idx_t *labels,
      const SearchParameters *params = nullptr) const override;

  void range_search(
      idx_t n,
      const float *x,
      float radius,
      RangeSearchResult *result,
      const SearchParameters *params = nullptr) const override;

  void check_compatible_for_merge(const Index &otherIndex) const override {
  }

  void merge_from(Index &otherIndex, idx_t add_id = 0) override {
  }
};

template<template<class> typename Storage>
void OpenSearchIndexFlatCodes<Storage>::search(
    idx_t n,
    const float *x,
    idx_t k,
    float *distances,
    idx_t *labels,
    const SearchParameters *params/* = nullptr*/) const {
}

template<template<class> typename Storage>
FlatCodesDistanceComputer *OpenSearchIndexFlatCodes<Storage>::get_FlatCodesDistanceComputer() const {
}

template<template<class> typename Storage>
void OpenSearchIndexFlatCodes<Storage>::range_search(
    idx_t n,
    const float *x,
    float radius,
    RangeSearchResult *result,
    const SearchParameters *params/* = nullptr*/) const {
}

}

#endif //KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_INDEX_FLAT_CODES_H_
