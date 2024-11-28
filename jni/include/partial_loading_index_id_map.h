//
// Created by Kim, Dooyong on 11/14/24.
//

#ifndef KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_INDEX_ID_MAP_H_
#define KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_INDEX_ID_MAP_H_

#include <faiss/Index.h>
#include <faiss/IndexBinary.h>
#include <faiss/impl/IDGrouper.h>
#include <faiss/impl/IDSelector.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <limits>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/WorkerThread.h>

#include <unordered_map>
#include <vector>

namespace faiss {


// IDSelector that translates the ids using an IDMap
template<template <class> typename Storage>
struct OpenSearchIDSelectorTranslated : IDSelector {
  const Storage<int64_t>& id_map;
  const IDSelector* sel;

  OpenSearchIDSelectorTranslated(
      const Storage<int64_t>& _id_map,
      const IDSelector* _sel)
      : id_map(_id_map), sel(_sel) {}

  bool is_member(idx_t id) const override {
    return sel->is_member(id_map[id]);
  }
};

// IDGrouper that translates the ids using an IDMap
template<template <class> typename Storage>
struct OpenSearchIDGrouperTranslated : IDGrouper {
  const Storage<int64_t>& id_map;
  const IDGrouper* grp;

  OpenSearchIDGrouperTranslated(
      const Storage<int64_t>& _id_map,
      const IDGrouper* _grp)
      : id_map(_id_map), grp(_grp) {}

  idx_t get_group(idx_t id) const override {
    return grp->get_group(id_map[id]);
  }
};


/** Index that translates search results to ids */
template <typename IndexT, template <class> typename Storage>
struct OpenSearchIndexIDMapTemplate : IndexT {
  using component_t = typename IndexT::component_t;
  using distance_t = typename IndexT::distance_t;

  IndexT* index = nullptr; ///! the sub-index
  bool own_fields = false; ///! whether pointers are deleted in destructo
  Storage<idx_t> id_map;

  explicit OpenSearchIndexIDMapTemplate(IndexT* index);

  /// @param xids if non-null, ids to store for the vectors (size n)
  void add_with_ids(idx_t n, const component_t* x, const idx_t* xids)
  override;

  /// this will fail. Use add_with_ids
  void add(idx_t n, const component_t* x) override;

  void search(
      idx_t n,
      const component_t* x,
      idx_t k,
      distance_t* distances,
      idx_t* labels,
      const SearchParameters* params = nullptr) const override;

  void train(idx_t n, const component_t* x) override;

  void reset() override;

  /// remove ids adapted to IndexFlat
  size_t remove_ids(const IDSelector& sel) override;

  void range_search(
      idx_t n,
      const component_t* x,
      distance_t radius,
      RangeSearchResult* result,
      const SearchParameters* params = nullptr) const override;

  void merge_from(IndexT& otherIndex, idx_t add_id = 0) override;
  void check_compatible_for_merge(const IndexT& otherIndex) const override;

  ~OpenSearchIndexIDMapTemplate() override;

  OpenSearchIndexIDMapTemplate() {
    own_fields = false;
    index = nullptr;
  }
};


/*****************************************************
 * IndexIDMap implementation
 *******************************************************/

template <typename IndexT, template <class> typename Storage>
OpenSearchIndexIDMapTemplate<IndexT, Storage>::OpenSearchIndexIDMapTemplate(
    IndexT* index) : IndexT(), index(index) {
  FAISS_THROW_IF_NOT_MSG(index->ntotal == 0, "index must be empty on input");
  this->is_trained = index->is_trained;
  this->metric_type = index->metric_type;
  this->verbose = index->verbose;
  this->d = index->d;
  sync_d(this);
}

template <typename IndexT, template <class> typename Storage>
void OpenSearchIndexIDMapTemplate<IndexT, Storage>::add(
    idx_t,
    const typename IndexT::component_t*) {
  FAISS_THROW_MSG(
      "add does not make sense with IndexIDMap, "
      "use add_with_ids");
}

template <typename IndexT, template <class> typename Storage>
void OpenSearchIndexIDMapTemplate<IndexT, Storage>::train(
    idx_t n,
    const typename IndexT::component_t* x) {
  index->train(n, x);
  this->is_trained = index->is_trained;
}

template <typename IndexT, template <class> typename Storage>
void OpenSearchIndexIDMapTemplate<IndexT, Storage>::reset() {
//  index->reset();
//  id_map.clear();
//  this->ntotal = 0;
  std::cout << "NOOOOOOOOOOOOOO OpenSearchIndexIDMapTemplate<IndexT, Storage>::reset()" << std::endl;
  assert(false);
}

template <typename IndexT, template <class> typename Storage>
void OpenSearchIndexIDMapTemplate<IndexT, Storage>::add_with_ids(
    idx_t n,
    const typename IndexT::component_t* x,
    const idx_t* xids) {
  // We don't need this.
  //  index->add(n, x);
  //  for (idx_t i = 0; i < n; i++)
  //    id_map.push_back(xids[i]);
  //  this->ntotal = index->ntotal;
  std::cout << "NOOOOOOOOOOOOOOOO OpenSearchIndexIDMapTemplate<IndexT, Storage>::add_with_ids" << std::endl;
  assert(false);
}

namespace {

/// RAII object to reset the IDSelector in the params object
struct ScopedSelChange {
  SearchParameters* params = nullptr;
  IDSelector* old_sel = nullptr;

  void set(SearchParameters* params_2, IDSelector* new_sel) {
    this->params = params_2;
    old_sel = params_2->sel;
    params_2->sel = new_sel;
  }
  ~ScopedSelChange() {
    if (params) {
      params->sel = old_sel;
    }
  }
};

/// RAII object to reset the IDGrouper in the params object
struct ScopedGrpChange {
  SearchParameters* params = nullptr;
  IDGrouper* old_grp = nullptr;

  void set(SearchParameters* params_2, IDGrouper* new_grp) {
    this->params = params_2;
    old_grp = params_2->grp;
    params_2->grp = new_grp;
  }
  ~ScopedGrpChange() {
    if (params) {
      params->grp = old_grp;
    }
  }
};

} // namespace



template <typename IndexT, template <class> typename Storage>
void OpenSearchIndexIDMapTemplate<IndexT, Storage>::search(
    idx_t numQueries,
    const typename IndexT::component_t* queryVectors,
    idx_t k,
    typename IndexT::distance_t* distances,
    idx_t* ids,
    const SearchParameters* params) const {
  if (params && params->sel) {
    throw std::runtime_error("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO params && params->sel");
  }

  if (params && params->grp) {
    throw std::runtime_error("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO params && params->grp");
  }

  index->search(numQueries, queryVectors, k, distances, ids, params);

  for (idx_t i = 0; i < numQueries * k; i++) {
    ids[i] = ids[i] < 0 ? ids[i] : id_map[ids[i]];
  }
}

template <typename IndexT, template <class> typename Storage>
void OpenSearchIndexIDMapTemplate<IndexT, Storage>::range_search(
    idx_t n,
    const typename IndexT::component_t* x,
    typename IndexT::distance_t radius,
    RangeSearchResult* result,
    const SearchParameters* params) const {
  if (params) {
    OpenSearchIDSelectorTranslated<Storage> this_idtrans(this->id_map, nullptr);
    ScopedSelChange sel_change;
    OpenSearchIDGrouperTranslated<Storage> this_idgrptrans(this->id_map, nullptr);
    ScopedGrpChange grp_change;

    if (params->sel) {
      auto idtrans = dynamic_cast<const IDSelectorTranslated*>(params->sel);

      if (!idtrans) {
        auto params_non_const = const_cast<SearchParameters*>(params);
        this_idtrans.sel = params->sel;
        sel_change.set(params_non_const, &this_idtrans);
      }
    }

    if (params->grp) {
      auto idtrans = dynamic_cast<const IDGrouperTranslated*>(params->grp);

      if (!idtrans) {
        auto params_non_const = const_cast<SearchParameters*>(params);
        this_idgrptrans.grp = params->grp;
        grp_change.set(params_non_const, &this_idgrptrans);
      }
    }
    index->range_search(n, x, radius, result, params);
  } else {
    index->range_search(n, x, radius, result);
  }

#pragma omp parallel for
  for (idx_t i = 0; i < result->lims[result->nq]; i++) {
    result->labels[i] = result->labels[i] < 0 ? result->labels[i]
                                              : id_map[result->labels[i]];
  }
}

template <typename IndexT, template <class> typename Storage>
size_t OpenSearchIndexIDMapTemplate<IndexT, Storage>::remove_ids(const IDSelector& sel) {
//  // remove in sub-index first
//  OpenSearchIDSelectorTranslated<Storage> sel2(id_map, &sel);
//  size_t nremove = index->remove_ids(sel2);
//
//  int64_t j = 0;
//  for (idx_t i = 0; i < this->ntotal; i++) {
//    if (sel.is_member(id_map[i])) {
//      // remove
//    } else {
//      id_map[j] = id_map[i];
//      j++;
//    }
//  }
//  FAISS_ASSERT(j == index->ntotal);
//  this->ntotal = j;
//  id_map.resize(this->ntotal);
//  return nremove;
  std::cout << "NOOOOOOOO size_t OpenSearchIndexIDMapTemplate<IndexT, Storage>::remove_ids" << std::endl;
  assert(false);
  return 0;
}

template <typename IndexT, template <class> typename Storage>
void OpenSearchIndexIDMapTemplate<IndexT, Storage>::check_compatible_for_merge(
    const IndexT& otherIndex) const {
  auto other = dynamic_cast<const OpenSearchIndexIDMapTemplate<IndexT, Storage>*>(&otherIndex);
  FAISS_THROW_IF_NOT(other);
  index->check_compatible_for_merge(*other->index);
}

template <typename IndexT, template <class> typename Storage>
void OpenSearchIndexIDMapTemplate<IndexT, Storage>::merge_from(IndexT& otherIndex, idx_t add_id) {
  std::cout << "NOOOOOOOOOOO OpenSearchIndexIDMapTemplate<IndexT, Storage>::merge_from" << std::endl;
  assert(false);
//  check_compatible_for_merge(otherIndex);
//  auto other = static_cast<OpenSearchIndexIDMapTemplate<IndexT, Storage>*>(&otherIndex);
//  index->merge_from(*other->index);
//  for (size_t i = 0; i < other->id_map.size(); i++) {
//    id_map.push_back(other->id_map[i] + add_id);
//  }
//  other->id_map.resize(0);
//  this->ntotal = index->ntotal;
//  other->ntotal = 0;
}

template <typename IndexT, template <class> typename Storage>
OpenSearchIndexIDMapTemplate<IndexT, Storage>::~OpenSearchIndexIDMapTemplate() {
  if (own_fields)
    delete index;
}



}

#endif //KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_INDEX_ID_MAP_H_
