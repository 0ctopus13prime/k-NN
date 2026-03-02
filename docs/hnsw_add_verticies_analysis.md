# hnsw_add_vertices — Deep Dive Analysis

## What is this function?

`hnsw_add_vertices` is the core function that builds the HNSW (Hierarchical Navigable Small World) graph.
When you add vectors to an HNSW index, this is the function that decides:
- Which layer each vector lives on
- Who each vector's neighbors are
- How the graph is connected

Think of HNSW as a multi-layer skip-list, but for vectors. The top layers are sparse (few vectors),
the bottom layer (level 0) has every vector. During search, you start at the top and greedily descend.

---

## Function Signature

```cpp
void hnsw_add_vertices(
    IndexHNSW& index_hnsw,  // The HNSW index we're building
    size_t n0,               // Number of vectors already in the index
    size_t n,                // Number of NEW vectors being added
    const float* x,          // Pointer to the new vector data
    bool verbose,            // Print progress?
    bool preset_levels       // Are levels pre-assigned? (usually false)
)
```

- `n0`: If you already have 5000 vectors and are adding 5000 more, `n0 = 5000`.
- `n`: The count of new vectors being added in this batch.
- `x`: Raw vector data. For standard flat storage, this is a contiguous array of floats.
  **For BBQ**, this is the quantized vector data (binary codes + correction factors).
- `preset_levels`: Usually `false`. If `true`, the HNSW levels were already assigned externally.

---

## Step-by-Step Breakdown

### Step 1: Assign Levels to Each Vector

```cpp
int max_level = hnsw.prepare_level_tab(n, preset_levels);
```

Every vector in HNSW gets assigned a "level". This is done randomly using an exponential distribution:
- Most vectors (the vast majority) get level 0 (bottom layer only)
- A few get level 1 (appear in bottom + one layer above)
- Even fewer get level 2, and so on

This is what makes HNSW hierarchical. The higher layers act like "express lanes" for search.

**What `prepare_level_tab` does internally:**
1. For each new vector, calls `random_level()` which returns a random level
   (exponentially distributed, controlled by parameter `M`)
2. Stores the level in `hnsw.levels[pt_id]`
3. Allocates space in the `neighbors` array for each vector's neighbor lists
   (each level has its own neighbor list)
4. Returns the maximum level assigned to any new vector

**Key detail:** The neighbor list capacity per level is:
- Level 0: `2 * M` neighbors (more connections at the bottom for better recall)
- Level 1+: `M` neighbors

---

### Step 2: Initialize Locks

```cpp
std::vector<omp_lock_t> locks(ntotal);
for (int i = 0; i < ntotal; i++) {
    omp_init_lock(&locks[i]);
}
```

HNSW construction is parallelized with OpenMP. Each vector gets its own lock because
when we add links between vectors, we need to modify both vectors' neighbor lists.
The lock prevents two threads from modifying the same vector's neighbor list simultaneously.

---

### Step 3: Sort Vectors by Level (Bucket Sort)

```cpp
// build histogram
for (int i = 0; i < n; i++) {
    storage_idx_t pt_id = i + n0;
    int pt_level = hnsw.levels[pt_id] - 1;
    hist[pt_level]++;
}
// ... bucket sort into `order` array
```

This sorts vectors so that **higher-level vectors come last** in the `order` array.
Why? Because we process from highest level to lowest. Vectors at higher levels need
to be inserted first — they form the "skeleton" of the graph that lower-level vectors
will use to find their neighbors.

**Example:** If we have 10000 vectors:
- ~9900 at level 0
- ~90 at level 1
- ~9 at level 2
- ~1 at level 3

The `order` array will be: `[level-0 vectors..., level-1 vectors..., level-2 vectors..., level-3 vectors]`

We process right-to-left: level 3 first, then level 2, then level 1, then level 0.

---

### Step 4: Process Each Level (Highest to Lowest)

```cpp
for (int pt_level = hist.size() - 1;
     pt_level >= int(!index_hnsw.init_level0);
     pt_level--) {
    int i0 = i1 - hist[pt_level];
    // ... process vectors at this level
}
```

For each level (starting from the highest):
1. Identify which vectors belong to this level (`i0` to `i1` in the `order` array)
2. Randomly shuffle them (to avoid dataset ordering bias)
3. Insert each vector into the graph

**The `init_level0` flag:** If `true`, we process all levels including level 0.
If `false`, we skip level 0 (used when level 0 is initialized separately).

---

### Step 5: Random Shuffle Within Each Level

```cpp
for (int j = i0; j < i1; j++) {
    std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);
}
```

Fisher-Yates shuffle. This is important because if your dataset has some ordering
(e.g., similar vectors are adjacent), inserting them in order would create a biased graph.
Shuffling ensures the graph structure doesn't depend on input order.

---

### Step 6: Create Distance Computer (THE CRITICAL PART FOR BBQ)

```cpp
std::unique_ptr<DistanceComputer> dis(
    storage_distance_computer(index_hnsw.storage));
```

This creates the distance computer that will be used to measure distances between vectors.

**What `storage_distance_computer` does:**
```cpp
DistanceComputer* storage_distance_computer(const Index* storage) {
    if (is_similarity_metric(storage->metric_type)) {
        return new NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}
```

For `FaissBBQFlat` (which uses `METRIC_INNER_PRODUCT`):
1. `storage->get_distance_computer()` returns a `BBQDistanceComputer`
2. Since inner product is a similarity metric (bigger = better), it wraps it in `NegativeDistanceComputer`
3. `NegativeDistanceComputer` negates all distances: `return -basedis->operator()(i)`

**Why negate?** HNSW internally uses a min-heap (keeps smallest values).
For inner product, we want the LARGEST similarity. By negating:
- A similarity of 146 becomes -146
- A similarity of 140 becomes -140
- The min-heap keeps -146 (which is the best match)

This is a standard trick to use a min-heap as a max-heap.

---

### Step 7: Set Query and Insert Each Vector

```cpp
for (int i = i0; i < i1; i++) {
    storage_idx_t pt_id = order[i];
    dis->set_query((const float*) (((const uint8_t*) x) + (pt_id - n0) * 112));

    hnsw.add_with_locks(
        *dis, pt_level, pt_id, locks, vt,
        index_hnsw.keep_max_size_level0 && (pt_level == 0));
}
```

For each vector being inserted:

#### 7a. `set_query` — Tell the distance computer "this is the vector I'm inserting"

**IMPORTANT NOTE:** The original Faiss code uses:
```cpp
dis->set_query(x + (pt_id - n0) * d);
```
where `d` is the vector dimension. This assumes `x` is a flat float array with stride = `d * sizeof(float)`.

But in this BBQ-modified version, it's been changed to:
```cpp
dis->set_query((const float*) (((const uint8_t*) x) + (pt_id - n0) * 112));
```
The stride is hardcoded to **112 bytes**. This is the `oneElementSize` of the BBQ quantized data:
`quantizedVectorBytes + 3 * sizeof(float) + sizeof(int32_t)`.

**This is a hardcoded value.** If the dimension changes, 112 might not be correct.
This could be a source of bugs if the quantized vector size doesn't match 112.

#### 7b. `add_with_locks` — Actually insert the vector into the graph

This is the core insertion algorithm. Here's what happens:

---

### Step 8: `add_with_locks` — The Insertion Algorithm

```cpp
void HNSW::add_with_locks(
    DistanceComputer& ptdis,  // distance computer with query = vector being inserted
    int pt_level,              // level assigned to this vector
    int pt_id,                 // ID of this vector
    std::vector<omp_lock_t>& locks,
    VisitedTable& vt,
    bool keep_max_size_level0)
```

#### Phase A: Find the entry point

```cpp
nearest = entry_point;
if (nearest == -1) {
    max_level = pt_level;
    entry_point = pt_id;
    return;  // First vector, nothing to connect to
}
```

If this is the very first vector, it becomes the entry point and we're done.
Otherwise, we start from the current entry point.

#### Phase B: Greedy descent from top to `pt_level`

```cpp
for (; level > pt_level; level--) {
    greedy_update_nearest(*this, ptdis, level, nearest, d_nearest);
}
```

Starting from the top level, greedily walk toward the nearest neighbor of our new vector.
At each level, we look at the current nearest node's neighbors and move to whichever
is closest to our new vector. We keep descending until we reach the level where our
new vector will be inserted.

**`greedy_update_nearest` in detail:**
```
loop forever:
    look at all neighbors of `nearest` at this level
    compute distance from each neighbor to our new vector
    if any neighbor is closer than `nearest`:
        update `nearest` to that neighbor
    else:
        break (we've found the local minimum at this level)
```

This uses `ptdis(neighbor_id)` which calls `BBQDistanceComputer::operator()` (wrapped in negation).
It processes neighbors in batches of 4 using `distances_batch_4` for efficiency.

#### Phase C: Build links at each level from `pt_level` down to 0

```cpp
for (; level >= 0; level--) {
    add_links_starting_from(
        ptdis, pt_id, nearest, d_nearest, level, locks, vt,
        keep_max_size_level0);
}
```

At each level where our vector exists, we need to:
1. Find the best neighbors for our vector
2. Create bidirectional links

---

### Step 9: `add_links_starting_from` — Finding and Creating Links

```cpp
void HNSW::add_links_starting_from(
    DistanceComputer& ptdis,
    storage_idx_t pt_id,
    storage_idx_t nearest,
    float d_nearest,
    int level, ...)
```

#### 9a. Search for candidate neighbors

```cpp
search_neighbors_to_add(*this, ptdis, link_targets, nearest, d_nearest, level, vt);
```

This does a BFS-like search starting from `nearest`, exploring the graph at this level
to find the `efConstruction` closest vectors to our new vector.

**How `search_neighbors_to_add` works:**
- Maintains two priority queues:
  - `candidates`: min-heap of vectors to explore (nearest first)
  - `results`: max-heap of best neighbors found so far (farthest first, capped at `efConstruction`)
- Starting from the entry point, repeatedly:
  1. Pop the nearest unexplored candidate
  2. If it's farther than the farthest result, stop (no point exploring further)
  3. Look at all its neighbors
  4. For each unvisited neighbor, compute distance using `ptdis(nodeId)`
  5. If it's better than the worst result (or we have fewer than `efConstruction` results), add it

The `efConstruction` parameter controls how thorough this search is.
Higher = better graph quality but slower construction.

#### 9b. Shrink the candidate list

```cpp
int M = nb_neighbors(level);  // M for upper levels, 2*M for level 0
::faiss::shrink_neighbor_list(ptdis, link_targets, M, keep_max_size_level0);
```

We found up to `efConstruction` candidates, but we can only keep `M` (or `2*M`) neighbors.
The shrinking uses a heuristic called "Simple Neighbor Selection" from the HNSW paper:

```
For each candidate (nearest first):
    Check if any already-selected neighbor is closer to this candidate
    than the candidate is to our query vector.
    If yes: skip it (it's "covered" by an existing neighbor)
    If no: keep it
```

**This is where `symmetric_dis` is used:**
```cpp
float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);
```

To check if neighbor v2 "covers" candidate v1, we need the distance between v2 and v1.
This is a stored-vector-to-stored-vector distance, hence `symmetric_dis`.

For BBQ, this calls `BBQDistanceComputer::symmetric_dis` (negated by `NegativeDistanceComputer`).

#### 9c. Create bidirectional links

```cpp
// Add link: pt_id -> other_id
add_link(*this, ptdis, pt_id, other_id, level, keep_max_size_level0);

// Add link: other_id -> pt_id (reverse direction)
omp_set_lock(&locks[other_id]);
add_link(*this, ptdis, other_id, pt_id, level, keep_max_size_level0);
omp_unset_lock(&locks[other_id]);
```

Links are bidirectional. When we add vector A as a neighbor of vector B,
we also add vector B as a neighbor of vector A.

If the neighbor list is already full, `add_link` calls `shrink_neighbor_list`
again to decide which neighbor to evict.

---

### Step 10: Update Entry Point

```cpp
if (pt_level > max_level) {
    max_level = pt_level;
    entry_point = pt_id;
}
```

If this vector was assigned a higher level than any previous vector,
it becomes the new entry point for the graph.

---

## Summary of Distance Computer Usage

During `hnsw_add_vertices`, the distance computer is used in three ways:

| Method | Where Used | What It Does |
|--------|-----------|--------------|
| `operator()(i)` | `greedy_update_nearest`, `search_neighbors_to_add` | Distance from the vector being inserted (set via `set_query`) to stored vector `i` |
| `distances_batch_4(...)` | `greedy_update_nearest` | Same as `operator()` but for 4 vectors at once (optimization) |
| `symmetric_dis(i, j)` | `shrink_neighbor_list` | Distance between two stored vectors `i` and `j` |

All three are wrapped by `NegativeDistanceComputer` for inner product metrics.

---

## BBQ-Specific Concerns

### 1. Hardcoded stride of 112

```cpp
dis->set_query((const float*) (((const uint8_t*) x) + (pt_id - n0) * 112));
```

The original code uses `x + (pt_id - n0) * d` (stride = dimension * sizeof(float)).
The BBQ version hardcodes 112 bytes as the stride. This is `oneElementSize`:
`quantizedVectorBytes + 3*sizeof(float) + sizeof(int32_t)`.

**If the vector dimension changes, this 112 must change too.**
For example, with 768-dim vectors: `768/8 = 96 bytes` for binary + 16 bytes for corrections = 112.
But for 128-dim vectors: `128/8 = 16 bytes` + 16 = 32 bytes. The hardcoded 112 would be wrong.

### 2. `set_query` points into the quantized data array

When `set_query` is called, it sets `query = (uint64_t*) x`, pointing directly into the
quantized vector data. The `operator()` then computes `popcount(query[i] & target[i])`
using this pointer. This means the query IS a quantized vector, not a float vector.

### 3. `FaissBBQFlat::add()` is a no-op

```cpp
void add(faiss::idx_t n, const float* x) final {
    ntotal += n;
}
```

The `add` method only increments `ntotal`. It does NOT copy vector data.
The actual quantized data is stored in `quantizedVectorsAndCorrectionFactors`
which is populated separately via `passBBQVectors` from Java.

This means by the time `hnsw_add_vertices` runs, the quantized data must already
be in `quantizedVectorsAndCorrectionFactors`. The `x` pointer passed to
`hnsw_add_vertices` must point to this data.
