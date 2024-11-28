//
// Created by Kim, Dooyong on 11/14/24.
//

#ifndef KNNPLUGIN_JNI_INCLUDE_OS_INDEX_HNSW_H_
#define KNNPLUGIN_JNI_INCLUDE_OS_INDEX_HNSW_H_

#include <vector>
#include "partial_loading_hnsw.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/HNSW.h>
#include <faiss/utils/utils.h>

#include <omp.h>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <unordered_set>

#include <sys/stat.h>
#include <sys/types.h>
#include <cstdint>

#include <faiss/Index2Layer.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/sorting.h>

namespace faiss {

template <class BlockResultHandler>
void _hnsw_search(
    const IndexHNSW* index,
    idx_t n,
    const float* x,
    BlockResultHandler& bres,
    const SearchParameters* params_in);

template <template<class> typename Storage>
struct OpenSearchIndexHNSW : Index {
  using H = OpenSearchHNSW<Storage>;
  using storage_idx_t = typename H::storage_idx_t;

  // The graph index
  OpenSearchHNSW<Storage> hnsw;

  // the sequential storage
  bool own_storage = false;
  Index* storage = nullptr;

  explicit OpenSearchIndexHNSW(int d = 0, int M = 32, MetricType metric = METRIC_L2)
      : Index(d, metric), hnsw(M) {}

  ~OpenSearchIndexHNSW() override {
    if (own_storage) {
      delete storage;
    }
  }

  void search(
      idx_t numQueries,
      const float* queryVectors,
      idx_t k,
      float* distances,
      idx_t* ids,
      const SearchParameters* params_in = nullptr) const override {
    HeapBlockResultHandler<HNSW::C> heap_block_result_handler(
        numQueries, distances, ids, k);

    _hnsw_search(this, numQueries, queryVectors,
                 heap_block_result_handler, params_in);
  }

  void add(idx_t n, const float* x) override {
  }

  void train(idx_t n, const float* x) override {
  }

  void range_search(
      idx_t n,
      const float* x,
      float radius,
      RangeSearchResult* result,
      const SearchParameters* params = nullptr) const override {
  }

  void reconstruct(idx_t key, float* recons) const override {
  }

  void reset() override {
  }

  DistanceComputer* get_distance_computer() const override {
    return storage->get_distance_computer();
  }
};

/** Flat index topped with with a HNSW structure to access elements
 *  more efficiently.
 */

template <template<class> typename Storage>
struct OpenSearchIndexHNSWFlat : OpenSearchIndexHNSW<Storage> {
  OpenSearchIndexHNSWFlat() : OpenSearchIndexHNSW<Storage>() {
    Index::is_trained = true;
  }
};


template <typename I, class BlockResultHandler>
void _hnsw_search(
    const I* index,
    idx_t numQueries,
    const float* queryVectors,
    BlockResultHandler& blockResultHandler,
    const SearchParameters* paramsIn) {
  const SearchParametersHNSW* params = nullptr;
  const auto& hnsw = index->hnsw;

  int efSearch = hnsw.efSearch;
  if (paramsIn) {
    params = dynamic_cast<const SearchParametersHNSW*>(paramsIn);
    FAISS_THROW_IF_NOT_MSG(params, "params type invalid");
    efSearch = params->efSearch;
  }
  size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;

  const idx_t check_period = InterruptCallback::get_period_hint(
      hnsw.max_level * index->d * efSearch);

  for (idx_t i0 = 0; i0 < numQueries; i0 += check_period) {
    const idx_t i1 = std::min(i0 + check_period, numQueries);

    VisitedTable vt (index->ntotal);
    typename BlockResultHandler::SingleResultHandler res(blockResultHandler);

    std::unique_ptr<DistanceComputer> distanceComputer(
        index->storage->get_distance_computer());

    for (idx_t i = i0; i < i1; i++) {
      res.begin(i);
      distanceComputer->set_query(&queryVectors[i * index->d]);

      HNSWStats stats = hnsw.search(*distanceComputer, res, vt, params);
      n1 += stats.n1;
      n2 += stats.n2;
      ndis += stats.ndis;
      nhops += stats.nhops;
      res.end();
    }

    InterruptCallback::check();
  }

  hnsw_stats.combine({n1, n2, ndis, nhops});
}

}  // namespace faiss

#endif //KNNPLUGIN_JNI_INCLUDE_OS_INDEX_HNSW_H_
