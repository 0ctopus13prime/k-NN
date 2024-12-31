//
// Created by Kim, Dooyong on 11/14/24.
//

#ifndef KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_HNSW_H_
#define KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_HNSW_H_

#include <faiss/impl/HNSW.h>

#include <queue>
#include <unordered_set>
#include <vector>

#include <cstddef>
#include <string>
#include <omp.h>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>
#include <faiss/impl/AuxIndexStructures.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/prefetch.h>

#ifdef __AVX2__
#include <immintrin.h>

#include <limits>
#include <type_traits>
#endif

namespace faiss {

template <template <class> typename Storage>
struct OpenSearchHNSW {
  /// internal storage of vectors (32 bits: this is expensive)
  using storage_idx_t = int32_t;

  // for now, we do only these distances
  using C = CMax<float, int64_t>;

  typedef std::pair<float, storage_idx_t> Node;

  /** Heap structure that allows fast
   */
  struct MinimaxHeap {
    int n;
    int k;
    int nvalid;

    std::vector<storage_idx_t> ids;
    std::vector<float> dis;
    typedef faiss::CMax<float, storage_idx_t> HC;

    explicit MinimaxHeap(int n) : n(n), k(0), nvalid(0), ids(n), dis(n) {}

    void push(storage_idx_t i, float v) {
      if (k == n) {
        if (v >= dis[0])
          return;
        if (ids[0] != -1) {
          --nvalid;
        }
        faiss::heap_pop<HC>(k--, dis.data(), ids.data());
      }
      faiss::heap_push<HC>(++k, dis.data(), ids.data(), v, i);
      ++nvalid;
    }

    float max() const {
      return dis[0];
    }

    int size() const {
      return nvalid;
    }

    void clear() {
      nvalid = k = 0;
    }

#ifdef __AVX512F__

    int pop_min(float* vmin_out) {
    assert(k > 0);
    static_assert(
            std::is_same<storage_idx_t, int32_t>::value,
            "This code expects storage_idx_t to be int32_t");

    int32_t min_idx = -1;
    float min_dis = std::numeric_limits<float>::infinity();

    __m512i min_indices = _mm512_set1_epi32(-1);
    __m512 min_distances =
            _mm512_set1_ps(std::numeric_limits<float>::infinity());
    __m512i current_indices = _mm512_setr_epi32(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i offset = _mm512_set1_epi32(16);

    // The following loop tracks the rightmost index with the min distance.
    // -1 index values are ignored.
    const int k16 = (k / 16) * 16;
    for (size_t iii = 0; iii < k16; iii += 16) {
        __m512i indices =
                _mm512_loadu_si512((const __m512i*)(ids.data() + iii));
        __m512 distances = _mm512_loadu_ps(dis.data() + iii);

        // This mask filters out -1 values among indices.
        __mmask16 m1mask =
                _mm512_cmpgt_epi32_mask(_mm512_setzero_si512(), indices);

        __mmask16 dmask =
                _mm512_cmp_ps_mask(min_distances, distances, _CMP_LT_OS);
        __mmask16 finalmask = m1mask | dmask;

        const __m512i min_indices_new = _mm512_mask_blend_epi32(
                finalmask, current_indices, min_indices);
        const __m512 min_distances_new =
                _mm512_mask_blend_ps(finalmask, distances, min_distances);

        min_indices = min_indices_new;
        min_distances = min_distances_new;

        current_indices = _mm512_add_epi32(current_indices, offset);
    }

    // leftovers
    if (k16 != k) {
        const __mmask16 kmask = (1 << (k - k16)) - 1;

        __m512i indices = _mm512_mask_loadu_epi32(
                _mm512_set1_epi32(-1), kmask, ids.data() + k16);
        __m512 distances = _mm512_maskz_loadu_ps(kmask, dis.data() + k16);

        // This mask filters out -1 values among indices.
        __mmask16 m1mask =
                _mm512_cmpgt_epi32_mask(_mm512_setzero_si512(), indices);

        __mmask16 dmask =
                _mm512_cmp_ps_mask(min_distances, distances, _CMP_LT_OS);
        __mmask16 finalmask = m1mask | dmask;

        const __m512i min_indices_new = _mm512_mask_blend_epi32(
                finalmask, current_indices, min_indices);
        const __m512 min_distances_new =
                _mm512_mask_blend_ps(finalmask, distances, min_distances);

        min_indices = min_indices_new;
        min_distances = min_distances_new;
    }

    // grab min distance
    min_dis = _mm512_reduce_min_ps(min_distances);
    // blend
    __mmask16 mindmask =
            _mm512_cmpeq_ps_mask(min_distances, _mm512_set1_ps(min_dis));
    // pick the max one
    min_idx = _mm512_mask_reduce_max_epi32(mindmask, min_indices);

    if (min_idx == -1) {
        return -1;
    }

    if (vmin_out) {
        *vmin_out = min_dis;
    }
    int ret = ids[min_idx];
    ids[min_idx] = -1;
    --nvalid;
    return ret;
}

#elif __AVX2__

    int pop_min(float* vmin_out) {
    assert(k > 0);
    static_assert(
            std::is_same<storage_idx_t, int32_t>::value,
            "This code expects storage_idx_t to be int32_t");

    int32_t min_idx = -1;
    float min_dis = std::numeric_limits<float>::infinity();

    size_t iii = 0;

    __m256i min_indices = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, -1);
    __m256 min_distances =
            _mm256_set1_ps(std::numeric_limits<float>::infinity());
    __m256i current_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i offset = _mm256_set1_epi32(8);

    // The baseline version is available in non-AVX2 branch.

    // The following loop tracks the rightmost index with the min distance.
    // -1 index values are ignored.
    const int k8 = (k / 8) * 8;
    for (; iii < k8; iii += 8) {
        __m256i indices =
                _mm256_loadu_si256((const __m256i*)(ids.data() + iii));
        __m256 distances = _mm256_loadu_ps(dis.data() + iii);

        // This mask filters out -1 values among indices.
        __m256i m1mask = _mm256_cmpgt_epi32(_mm256_setzero_si256(), indices);

        __m256i dmask = _mm256_castps_si256(
                _mm256_cmp_ps(min_distances, distances, _CMP_LT_OS));
        __m256 finalmask = _mm256_castsi256_ps(_mm256_or_si256(m1mask, dmask));

        const __m256i min_indices_new = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(current_indices),
                _mm256_castsi256_ps(min_indices),
                finalmask));

        const __m256 min_distances_new =
                _mm256_blendv_ps(distances, min_distances, finalmask);

        min_indices = min_indices_new;
        min_distances = min_distances_new;

        current_indices = _mm256_add_epi32(current_indices, offset);
    }

    // Vectorizing is doable, but is not practical
    int32_t vidx8[8];
    float vdis8[8];
    _mm256_storeu_ps(vdis8, min_distances);
    _mm256_storeu_si256((__m256i*)vidx8, min_indices);

    for (size_t j = 0; j < 8; j++) {
        if (min_dis > vdis8[j] || (min_dis == vdis8[j] && min_idx < vidx8[j])) {
            min_idx = vidx8[j];
            min_dis = vdis8[j];
        }
    }

    // process last values. Vectorizing is doable, but is not practical
    for (; iii < k; iii++) {
        if (ids[iii] != -1 && dis[iii] <= min_dis) {
            min_dis = dis[iii];
            min_idx = iii;
        }
    }

    if (min_idx == -1) {
        return -1;
    }

    if (vmin_out) {
        *vmin_out = min_dis;
    }
    int ret = ids[min_idx];
    ids[min_idx] = -1;
    --nvalid;
    return ret;
}

#else

// baseline non-vectorized version
    int pop_min(float* vmin_out) {
      assert(k > 0);
      // returns min. This is an O(n) operation
      int i = k - 1;
      while (i >= 0) {
        if (ids[i] != -1) {
          break;
        }
        i--;
      }
      if (i == -1) {
        return -1;
      }
      int imin = i;
      float vmin = dis[i];
      i--;
      while (i >= 0) {
        if (ids[i] != -1 && dis[i] < vmin) {
          vmin = dis[i];
          imin = i;
        }
        i--;
      }
      if (vmin_out) {
        *vmin_out = vmin;
      }
      int ret = ids[imin];
      ids[imin] = -1;
      --nvalid;

      return ret;
    }
#endif

    int count_below(float thresh) {
      int n_below = 0;
      for (int i = 0; i < k; i++) {
        if (dis[i] < thresh) {
          n_below++;
        }
      }

      return n_below;
    }
  };

  /// to sort pairs of (id, distance) from nearest to fathest or the reverse
  struct NodeDistCloser {
    float d;
    int id;
    NodeDistCloser(float d, int id) : d(d), id(id) {}
    bool operator<(const NodeDistCloser& obj1) const {
      return d < obj1.d;
    }
  };

  struct NodeDistFarther {
    float d;
    int id;
    NodeDistFarther(float d, int id) : d(d), id(id) {}
    bool operator<(const NodeDistFarther& obj1) const {
      return d > obj1.d;
    }
  };

  /// assignment probability to each layer (sum=1)
  std::vector<double> assign_probas;

  /// number of neighbors stored per layer (cumulative), should not
  /// be changed after first add
  std::vector<int> cum_nneighbor_per_level;

  /// level of each vector (base level = 1), size = ntotal
  Storage<int> levels;

  /// offsets[i] is the offset in the neighbors array where vector i is stored
  /// size ntotal + 1
  std::vector<size_t> offsets;

  /// neighbors[offsets[i]:offsets[i+1]] is the list of neighbors of vector i
  /// for all levels. this is where all storage goes.
  Storage<storage_idx_t> neighbors;

  /// entry point in the search structure (one of the points with maximum
  /// level
  storage_idx_t entry_point = -1;

  faiss::RandomGenerator rng;

  /// maximum level
  int max_level = -1;

  /// expansion factor at construction time
  int efConstruction = 40;

  /// expansion factor at search time
  int efSearch = 16;

  /// during search: do we check whether the next best distance is good
  /// enough?
  bool check_relative_distance = true;

  /// use bounded queue during exploration
  bool search_bounded_queue = true;

  // methods that initialize the tree sizes

  std::priority_queue<HNSW::Node> search_from_candidate_unbounded(
      const Node& node,
      DistanceComputer& qdis,
      int ef,
      VisitedTable* vt,
      HNSWStats& stats) const {
    int ndis = 0;
    std::priority_queue<Node> top_candidates;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;

    top_candidates.push(node);
    candidates.push(node);

    vt->set(node.second);

    while (!candidates.empty()) {
      float d0;
      storage_idx_t v0;
      std::tie(d0, v0) = candidates.top();

      if (d0 > top_candidates.top().first) {
        break;
      }

      candidates.pop();

      size_t begin, end;
      neighbor_range(v0, 0, &begin, &end);

      // a faster version: reference version in unit test test_hnsw.cpp
      // the following version processes 4 neighbors at a time
      size_t jmax = begin;
      for (size_t j = begin; j < end; j++) {
        int v1 = neighbors[j];
        if (v1 < 0)
          break;

        prefetch_L2(vt->visited.data() + v1);
        jmax += 1;
      }

      int counter = 0;
      size_t saved_j[4];

      auto add_to_heap = [&](const size_t idx, const float dis) {
        if (top_candidates.top().first > dis ||
            top_candidates.size() < ef) {
          candidates.emplace(dis, idx);
          top_candidates.emplace(dis, idx);

          if (top_candidates.size() > ef) {
            top_candidates.pop();
          }
        }
      };

      for (size_t j = begin; j < jmax; j++) {
        int v1 = neighbors[j];

        bool vget = vt->get(v1);
        vt->set(v1);
        saved_j[counter] = v1;
        counter += vget ? 0 : 1;

        if (counter == 4) {
          float dis[4];
          qdis.distances_batch_4(
              saved_j[0],
              saved_j[1],
              saved_j[2],
              saved_j[3],
              dis[0],
              dis[1],
              dis[2],
              dis[3]);

          for (size_t id4 = 0; id4 < 4; id4++) {
            add_to_heap(saved_j[id4], dis[id4]);
          }

          ndis += 4;

          counter = 0;
        }
      }

      for (size_t icnt = 0; icnt < counter; icnt++) {
        float dis = qdis(saved_j[icnt]);
        add_to_heap(saved_j[icnt], dis);

        ndis += 1;
      }

      stats.nhops += 1;
    }

    ++stats.n1;
    if (candidates.size() == 0) {
      ++stats.n2;
    }
    stats.ndis += ndis;

    return top_candidates;
  }

  HNSWStats greedy_update_nearest(
      DistanceComputer& qdis,
      int level,
      storage_idx_t& nearest,
      float& d_nearest) const {
    HNSWStats stats;

    for (;;) {
      storage_idx_t prev_nearest = nearest;

      size_t begin, end;
      neighbor_range(nearest, level, &begin, &end);

      size_t ndis = 0;

      // a faster version: reference version in unit test test_hnsw.cpp
      // the following version processes 4 neighbors at a time
      auto update_with_candidate = [&](const storage_idx_t idx,
                                       const float dis) {
        if (dis < d_nearest) {
          nearest = idx;
          d_nearest = dis;
        }
      };

      int n_buffered = 0;
      storage_idx_t buffered_ids[4];

      for (size_t j = begin; j < end; j++) {
        storage_idx_t v = neighbors[j];
        if (v < 0)
          break;
        ndis += 1;

        buffered_ids[n_buffered] = v;
        n_buffered += 1;

        if (n_buffered == 4) {
          float dis[4];
          qdis.distances_batch_4(
              buffered_ids[0],
              buffered_ids[1],
              buffered_ids[2],
              buffered_ids[3],
              dis[0],
              dis[1],
              dis[2],
              dis[3]);

          for (size_t id4 = 0; id4 < 4; id4++) {
            update_with_candidate(buffered_ids[id4], dis[id4]);
          }

          n_buffered = 0;
        }
      }

      // process leftovers
      for (size_t icnt = 0; icnt < n_buffered; icnt++) {
        float dis = qdis(buffered_ids[icnt]);
        update_with_candidate(buffered_ids[icnt], dis);
      }

      // update stats
      stats.ndis += ndis;
      stats.nhops += 1;

      if (nearest == prev_nearest) {
        return stats;
      }
    }
  }

  int search_from_candidates(
      DistanceComputer& qdis,
      ResultHandler<C>& res,
      MinimaxHeap& candidates,
      VisitedTable& vt,
      HNSWStats& stats,
      int level,
      int nres_in,
      const SearchParametersHNSW* params) const {
    int nres = nres_in;
    int ndis = 0;

    // can be overridden by search params
    const bool do_dis_check = params ? params->check_relative_distance
                               : check_relative_distance;
    int efSearch = params ? params->efSearch : efSearch;
    const IDSelector* sel = params ? params->sel : nullptr;

//    std::cout << "---------------------- 1" << std::endl;

    C::T threshold = res.threshold;
    for (int i = 0; i < candidates.size(); i++) {
      idx_t v1 = candidates.ids[i];
      float d = candidates.dis[i];
      FAISS_ASSERT(v1 >= 0);
      if (!sel || sel->is_member(v1)) {
        if (d < threshold) {
          if (res.add_result(d, v1)) {
            threshold = res.threshold;
          }
        }
      }
      vt.set(v1);
    }

//    std::cout << "---------------------- 2, do_dis_check=" << do_dis_check << std::endl;

    int nstep = 0;

    while (candidates.size() > 0) {
//      std::cout << "---------------------- 3: " << candidates.size() << std::endl;

      float d0 = 0;
      int v0 = candidates.pop_min(&d0);

      if (do_dis_check) {
        // tricky stopping condition: there are more that ef
        // distances that are processed already that are smaller
        // than d0

        int n_dis_below = candidates.count_below(d0);
        if (n_dis_below >= efSearch) {
          break;
        }
      }

      size_t begin, end;
      neighbor_range(v0, level, &begin, &end);
//      std::cout << "------------------- begin=" << begin << ", end=" << end << std::endl;

      // a faster version: reference version in unit test test_hnsw.cpp
      // the following version processes 4 neighbors at a time
      size_t jmax = begin;
      for (size_t j = begin; j < end; j++) {
        int v1 = neighbors[j];
//        std::cout << "---------------- neighbors[" << j << "] -> " << v1 << std::endl;
        if (v1 < 0)
          break;

        prefetch_L2(vt.visited.data() + v1);
        jmax += 1;
      }

      int counter = 0;
      size_t saved_j[4];

      threshold = res.threshold;

      auto add_to_heap = [&](const size_t idx, const float dis) {
        if (!sel || sel->is_member(idx)) {
          if (dis < threshold) {
            if (res.add_result(dis, idx)) {
              threshold = res.threshold;
              nres += 1;
            }
          }
        }
        candidates.push(idx, dis);
      };

//      std::cout << "----------------- jmax=" << jmax << std::endl;
      for (size_t j = begin; j < jmax; j++) {
        int v1 = neighbors[j];
//        std::cout << "---------------- neighbors[" << j << "] -> " << v1 << std::endl;

        bool vget = vt.get(v1);
        vt.set(v1);
        saved_j[counter] = v1;
        counter += vget ? 0 : 1;

        if (counter == 4) {
          float dis[4];
          qdis.distances_batch_4(
              saved_j[0],
              saved_j[1],
              saved_j[2],
              saved_j[3],
              dis[0],
              dis[1],
              dis[2],
              dis[3]);

          for (size_t id4 = 0; id4 < 4; id4++) {
            add_to_heap(saved_j[id4], dis[id4]);
          }

          ndis += 4;

          counter = 0;
        }
      }

      for (size_t icnt = 0; icnt < counter; icnt++) {
        float dis = qdis(saved_j[icnt]);
        add_to_heap(saved_j[icnt], dis);

        ndis += 1;
      }

      nstep++;
      if (!do_dis_check && nstep > efSearch) {
        break;
      }
    }

    if (level == 0) {
      stats.n1++;
      if (candidates.size() == 0) {
        stats.n2++;
      }
      stats.ndis += ndis;
      stats.nhops += nstep;
    }

    return nres;
  }

  /// initialize the assign_probas and cum_nneighbor_per_level to
  /// have 2*M links on level 0 and M links on levels > 0
  void set_default_probas(int M, float levelMult)  {
    int nn = 0;
    cum_nneighbor_per_level.push_back(0);
    for (int level = 0;; level++) {
      float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
      if (proba < 1e-9)
        break;
      assign_probas.push_back(proba);
      nn += level == 0 ? M * 2 : M;
      cum_nneighbor_per_level.push_back(nn);
    }
  }

  /// set nb of neighbors for this level (before adding anything)
  void set_nb_neighbors(int level_no, int n)  {
    FAISS_THROW_IF_NOT(levels.size() == 0);
    int cur_n = nb_neighbors(level_no);
    for (int i = level_no + 1; i < cum_nneighbor_per_level.size(); i++) {
      cum_nneighbor_per_level[i] += n - cur_n;
    }
  }

  // methods that access the tree sizes

  /// nb of neighbors for this level
  int nb_neighbors(int layer_no) const {
    return cum_nneighbor_per_level[layer_no + 1] -
        cum_nneighbor_per_level[layer_no];
  }

  /// cumumlative nb up to (and excluding) this level
  int cum_nb_neighbors(int layer_no) const {
    return cum_nneighbor_per_level[layer_no];
  }

  /// range of entries in the neighbors table of vertex no at layer_no
  void neighbor_range(idx_t no, int layer_no, size_t* begin, size_t* end)
  const {
    size_t o = offsets[no];
    *begin = o + cum_nb_neighbors(layer_no);
    *end = o + cum_nb_neighbors(layer_no + 1);
  }

  /// only mandatory parameter: nb of neighbors
  explicit OpenSearchHNSW(int M = 32): rng(12345) {
    set_default_probas(M, 1.0 / log(M));
    // offsets.push_back(0);
  }

  /// pick a random level for a new point
  int random_level()  {
    double f = rng.rand_float();
    // could be a bit faster with bissection
    for (int level = 0; level < assign_probas.size(); level++) {
      if (f < assign_probas[level]) {
        return level;
      }
      f -= assign_probas[level];
    }
    // happens with exponentially low probability
    return assign_probas.size() - 1;
  }

  /// add n random levels to table (for debugging...)
  void fill_with_random_links(size_t n)  {
    std::cout << "NOOOOOOOOOOOOOOOO fill_with_random_links" << std::endl;
    assert(false);

    int max_level = prepare_level_tab(n);
    RandomGenerator rng2(456);

    for (int level = max_level - 1; level >= 0; --level) {
      std::vector<int> elts;
      for (int i = 0; i < n; i++) {
        if (levels[i] > level) {
          elts.push_back(i);
        }
      }
      printf("linking %zd elements in level %d\n", elts.size(), level);

      if (elts.size() == 1)
        continue;

      for (int ii = 0; ii < elts.size(); ii++) {
        int i = elts[ii];
        size_t begin, end;
        neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end; j++) {
          int other = 0;
          do {
            other = elts[rng2.rand_int(elts.size())];
          } while (other == i);

          neighbors[j] = other;
        }
      }
    }
  }

  void add_links_starting_from(
      DistanceComputer& ptdis,
      storage_idx_t pt_id,
      storage_idx_t nearest,
      float d_nearest,
      int level,
      omp_lock_t* locks,
      VisitedTable& vt,
      bool keep_max_size_level0 = false)  {
    std::cout << "NOOOOOOOOOOOOOOOOOOOOOO add_links_starting_from" << std::endl;
    assert(false);

    std::priority_queue<NodeDistCloser> link_targets;

    search_neighbors_to_add(
        *this, ptdis, link_targets, nearest, d_nearest, level, vt);

    // but we can afford only this many neighbors
    int M = nb_neighbors(level);

    do_shrink_neighbor_list(ptdis, link_targets, M, keep_max_size_level0);

    std::vector<storage_idx_t> neighbors;
    neighbors.reserve(link_targets.size());
    while (!link_targets.empty()) {
      storage_idx_t other_id = link_targets.top().id;
      add_link(*this, ptdis, pt_id, other_id, level, keep_max_size_level0);
      neighbors.push_back(other_id);
      link_targets.pop();
    }

    omp_unset_lock(&locks[pt_id]);
    for (storage_idx_t other_id : neighbors) {
      omp_set_lock(&locks[other_id]);
      add_link(*this, ptdis, other_id, pt_id, level, keep_max_size_level0);
      omp_unset_lock(&locks[other_id]);
    }
    omp_set_lock(&locks[pt_id]);
  }

  /** add point pt_id on all levels <= pt_level and build the link
   * structure for them. */
  void add_with_locks(
      DistanceComputer& ptdis,
      int pt_level,
      int pt_id,
      std::vector<omp_lock_t>& locks,
      VisitedTable& vt,
      bool keep_max_size_level0 = false) {
    //  greedy search on upper levels
    storage_idx_t nearest;
#pragma omp critical
    {
      nearest = entry_point;

      if (nearest == -1) {
        max_level = pt_level;
        entry_point = pt_id;
      }
    }

    if (nearest < 0) {
      return;
    }

    omp_set_lock(&locks[pt_id]);

    int level = max_level; // level at which we start adding neighbors
    float d_nearest = ptdis(nearest);

    for (; level > pt_level; level--) {
      greedy_update_nearest(ptdis, level, nearest, d_nearest);
    }

    for (; level >= 0; level--) {
      add_links_starting_from(
          ptdis,
          pt_id,
          nearest,
          d_nearest,
          level,
          locks.data(),
          vt,
          keep_max_size_level0);
    }

    omp_unset_lock(&locks[pt_id]);

    if (pt_level > max_level) {
      max_level = pt_level;
      entry_point = pt_id;
    }
  }

  /// search interface for 1 point, single thread
  HNSWStats search(
      DistanceComputer& distanceComputer,
      ResultHandler<C>& resultHandler,
      VisitedTable& visitedTable,
      const SearchParametersHNSW* params = nullptr) const  {
    HNSWStats stats;
    if (entry_point == -1) {
      return stats;
    }
    const int k = extract_k_from_ResultHandler(resultHandler);

    const bool bounded_queue =
        params ? params->bounded_queue : this->search_bounded_queue;

    // greedy search on upper levels
    storage_idx_t nearest = entry_point;
    float d_nearest = distanceComputer(nearest);

    for (int level = max_level; level >= 1; level--) {
      HNSWStats local_stats =
          greedy_update_nearest(distanceComputer, level, nearest, d_nearest);
      stats.combine(local_stats);
    }

    int ef = std::max(params ? params->efSearch : efSearch, k);
    if (bounded_queue) { // this is the most common branch
      MinimaxHeap candidates(ef);

      candidates.push(nearest, d_nearest);

      search_from_candidates(
          distanceComputer, resultHandler, candidates, visitedTable, stats, 0, 0, params);
    } else {
      std::priority_queue<Node> top_candidates =
          search_from_candidate_unbounded(
              Node(d_nearest, nearest), distanceComputer, ef, &visitedTable, stats);

      while (top_candidates.size() > k) {
        top_candidates.pop();
      }

      while (!top_candidates.empty()) {
        float d;
        storage_idx_t label;
        std::tie(d, label) = top_candidates.top();
        resultHandler.add_result(d, label);
        top_candidates.pop();
      }
    }

    visitedTable.advance();

    return stats;
  }

  void reset() {
//    max_level = -1;
//    entry_point = -1;
//    offsets.clear();
//    offsets.push_back(0);
//    levels.clear();
//    neighbors.clear();
    std::cout << "NOOOOOOOOOOOOOOOOOOOO reset()" << std::endl;
    assert(false);
  }

  void clear_neighbor_tables(int level) {
    std::cout << "NOOOOOOOOOOOOOOOOOOOOOOO clear_neighbor_tables" << std::endl;
    assert(false);

    for (int i = 0; i < levels.size(); i++) {
      size_t begin, end;
      neighbor_range(i, level, &begin, &end);
      for (size_t j = begin; j < end; j++) {
        neighbors[j] = -1;
      }
    }
  }

  void print_neighbor_stats(int level) const {
    FAISS_THROW_IF_NOT(level < cum_nneighbor_per_level.size());
    printf("stats on level %d, max %d neighbors per vertex:\n",
           level,
           nb_neighbors(level));
    size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
#pragma omp parallel for reduction(+ : tot_neigh) reduction(+ : tot_common) \
        reduction(+ : tot_reciprocal) reduction(+ : n_node)
    for (int i = 0; i < levels.size(); i++) {
      if (levels[i] > level) {
        n_node++;
        size_t begin, end;
        neighbor_range(i, level, &begin, &end);
        std::unordered_set<int> neighset;
        for (size_t j = begin; j < end; j++) {
          if (neighbors[j] < 0)
            break;
          neighset.insert(neighbors[j]);
        }
        int n_neigh = neighset.size();
        int n_common = 0;
        int n_reciprocal = 0;
        for (size_t j = begin; j < end; j++) {
          storage_idx_t i2 = neighbors[j];
          if (i2 < 0)
            break;
          FAISS_ASSERT(i2 != i);
          size_t begin2, end2;
          neighbor_range(i2, level, &begin2, &end2);
          for (size_t j2 = begin2; j2 < end2; j2++) {
            storage_idx_t i3 = neighbors[j2];
            if (i3 < 0)
              break;
            if (i3 == i) {
              n_reciprocal++;
              continue;
            }
            if (neighset.count(i3)) {
              neighset.erase(i3);
              n_common++;
            }
          }
        }
        tot_neigh += n_neigh;
        tot_common += n_common;
        tot_reciprocal += n_reciprocal;
      }
    }
    float normalizer = n_node;
    printf("   nb of nodes at that level %zd\n", n_node);
    printf("   neighbors per node: %.2f (%zd)\n",
           tot_neigh / normalizer,
           tot_neigh);
    printf("   nb of reciprocal neighbors: %.2f\n",
           tot_reciprocal / normalizer);
    printf("   nb of neighbors that are also neighbor-of-neighbors: %.2f (%zd)\n",
           tot_common / normalizer,
           tot_common);
  }

  int prepare_level_tab(size_t n, bool preset_levels = false) {
    std::cout << "NOOOOOOOOOOOOOOOOOOOOOO prepare_level_tab" << std::endl;
    assert(false);

    size_t n0 = offsets.size() - 1;

    if (preset_levels) {
      FAISS_ASSERT(n0 + n == levels.size());
    } else {
      FAISS_ASSERT(n0 == levels.size());
      for (int i = 0; i < n; i++) {
        int pt_level = random_level();
        levels.push_back(pt_level + 1);
      }
    }

    int max_level = 0;
    for (int i = 0; i < n; i++) {
      int pt_level = levels[i + n0] - 1;
      if (pt_level > max_level)
        max_level = pt_level;
      offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
    }
    neighbors.resize(offsets.back(), -1);

    return max_level;
  }

  static void shrink_neighbor_list(
      DistanceComputer& qdis,
      std::priority_queue<NodeDistFarther>& input,
      std::vector<NodeDistFarther>& output,
      int max_size,
      bool keep_max_size_level0 = false)  {
    // This prevents number of neighbors at
    // level 0 from being shrunk to less than 2 * M.
    // This is essential in making sure
    // `faiss::gpu::GpuIndexCagra::copyFrom(IndexHNSWCagra*)` is functional
    std::vector<NodeDistFarther> outsiders;

    while (input.size() > 0) {
      NodeDistFarther v1 = input.top();
      input.pop();
      float dist_v1_q = v1.d;

      bool good = true;
      for (NodeDistFarther v2 : output) {
        float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);

        if (dist_v1_v2 < dist_v1_q) {
          good = false;
          break;
        }
      }

      if (good) {
        output.push_back(v1);
        if (output.size() >= max_size) {
          return;
        }
      } else if (keep_max_size_level0) {
        outsiders.push_back(v1);
      }
    }
    size_t idx = 0;
    while (keep_max_size_level0 && (output.size() < max_size) &&
        (idx < outsiders.size())) {
      output.push_back(outsiders[idx++]);
    }
  }

  void permute_entries(const idx_t* map) {
    std::cout << "NOOOOOOOOOOOOOOOOOOO permute_entries" << std::endl;
    assert(false);

    // remap levels
    storage_idx_t ntotal = levels.size();
    std::vector<storage_idx_t> imap(ntotal); // inverse mapping
    // map: new index -> old index
    // imap: old index -> new index
    for (int i = 0; i < ntotal; i++) {
      assert(map[i] >= 0 && map[i] < ntotal);
      imap[map[i]] = i;
    }
    if (entry_point != -1) {
      entry_point = imap[entry_point];
    }
    std::vector<int> new_levels(ntotal);
    std::vector<size_t> new_offsets(ntotal + 1);
    std::vector<storage_idx_t> new_neighbors(neighbors.size());
    size_t no = 0;
    for (int i = 0; i < ntotal; i++) {
      storage_idx_t o = map[i]; // corresponding "old" index
      new_levels[i] = levels[o];
      for (size_t j = offsets[o]; j < offsets[o + 1]; j++) {
        storage_idx_t neigh = neighbors[j];
        new_neighbors[no++] = neigh >= 0 ? imap[neigh] : neigh;
      }
      new_offsets[i + 1] = no;
    }
    assert(new_offsets[ntotal] == offsets[ntotal]);
    // swap everyone
    std::swap(levels, new_levels);
    std::swap(offsets, new_offsets);
    std::swap(neighbors, new_neighbors);
  }

  /// remove neighbors from the list to make it smaller than max_size
  static void do_shrink_neighbor_list(
      DistanceComputer& qdis,
      std::priority_queue<NodeDistCloser>& resultSet1,
      int max_size,
      bool keep_max_size_level0 = false) {
    if (resultSet1.size() < max_size) {
      return;
    }
    std::priority_queue<NodeDistFarther> resultSet;
    std::vector<NodeDistFarther> returnlist;

    while (resultSet1.size() > 0) {
      resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
      resultSet1.pop();
    }

    HNSW::shrink_neighbor_list(
        qdis, resultSet, returnlist, max_size, keep_max_size_level0);

    for (NodeDistFarther curen2 : returnlist) {
      resultSet1.emplace(curen2.d, curen2.id);
    }
  }

  static int extract_k_from_ResultHandler(ResultHandler<C>& res) {
    using RH = HeapBlockResultHandler<C>;
    if (auto hres = dynamic_cast<RH::SingleResultHandler*>(&res)) {
      return hres->k;
    }

    if (auto hres = dynamic_cast<
        GroupedHeapBlockResultHandler<C>::SingleResultHandler*>(&res)) {
      return hres->k;
    }

    return 1;
  }
};  // class OpenSearchHNSW


}


#endif //KNNPLUGIN_JNI_INCLUDE_PARTIAL_LOADING_HNSW_H_
