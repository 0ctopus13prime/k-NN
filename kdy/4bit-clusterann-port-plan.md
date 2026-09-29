# Native bulk SIMD dot product for the cluster ANN 4-bit index (POC)

Date: 2026-09-24. Scope: ClusterANN Flow A (per-centroid OSQ), the `.clap` block-columnar ADC path, 4-bit
documents. HNSW is out of scope.

> **Branch note (2026-09-24, later the same day).** The work below was first done on `cluster-ann` (navneet1v's
> older cluster ANN tree). It was then re-ported onto `3-5-cluster-ann`, which is a different implementation
> (`org.opensearch.knn.clusterann.*`, OpenSearch 3.5 / Lucene 10.3.2). See "Port to 3-5-cluster-ann" at the end;
> that section supersedes the writer/reader details in "What changed" for that branch.

## Decision

4-bit documents are now stored as **split-half nibbles**, the Lucene104 `PACKED_NIBBLE` layout
(`byte i = code[i] << 4 | code[i + half]`), and scored by a native multiply-accumulate kernel (`vpdpbusd` on
AVX-512 VNNI, `vpmaddubsw`+`vpmaddwd` on AVX-512BW / AVX2, `udot` on NEON). The query is left unpacked, one 4-bit
code per dimension, exactly as Lucene's own 4-bit scorer does.

Background: the cluster writer used to pack 4-bit docs with `transposeHalfByte`, which is Lucene's *query*-side
function and yields four bit planes; Lucene never writes documents that way. The Lucene git history shows no
change to either function between 10.3 and 10.5 (`transposeHalfByte` since 10.2.0, `packNibbles` since 10.4.0).
A bit-plane kernel was built first (it is still in the file for 1-bit and 2-bit), then replaced for 4-bit because
the nibble kernel needs roughly 8x fewer instructions per vector and search latency is the goal.

Compatibility: existing 4-bit indexes must be rebuilt. For 768 dims the record size is unchanged (384 bytes);
it differs only for dimensions not divisible by 8. 1-bit and 2-bit indexes are untouched.

## What changed

Java (`src/main`)

- `ScalarBitEncoding.FOUR_BIT`: `packDoc` = split-half nibble packer (odd dims pad the last low nibble with 0);
  `docPackedBytes = (dim + 1) / 2`.
- `QuantizedVectorWriter`: 4-bit branch calls `ScalarBitEncoding.FOUR_BIT.packDoc`; `packedBytesPerVector`
  delegates to the enum so the two formulas cannot drift.
- `QuantizedVectorReader`: for 4-bit the query is not transposed; the unpacked codes are copied into a buffer of
  `2 * packedBytes`. `scoreBlock` calls the native kernel with `validOffsets` when `NATIVE_DOT_PRODUCT` (SIMD
  library loads and `-Dclusterann.native.dot` is not `false`), else a plain Java nibble loop. Epilogue untouched.
- `SimdVectorComputeService`: `bulkQuantizedDotProduct(query, docs, offsets, numOffsets, results, bytesPerCode,
  docBits)` and diagnostic `bulkQuantizedDotProductKernel()` (returns e.g. `neon/neon-dotprod`, `avx512/avx512vnni`).

Native (`jni/`)

- `include/simd/clusterann_batch_dot_product.h`: API + layout contract for both encodings.
- `src/simd/similarity_function/clusterann_batch_dot_product.cpp`:
  - 4-bit nibble kernel per ISA, batch 8 (AVX-512, NEON) or 4 (AVX2), masked tail on AVX-512, scalar tail
    elsewhere, L1 prefetch of the next chunk of the query halves and every document before each chunk.
    AVX-512 picks VNNI at runtime via CPUID leaf 7 ECX bit 11; the VNNI and BW bodies come from one macro.
  - 1-bit / 2-bit bit-plane popcount kernel (same design as the existing 1-bit SQ kernels), also prefetching.
  - Driver: 8-block, 4-block, scalar tail; `results[offsets[k]]` so invalid block entries are skipped.
- `src/org_opensearch_knn_jni_SimdVectorComputeService.cpp`: validates docBits, bytesPerCode, query length
  (`2 * bytesPerCode` for 4-bit, `4 * stripe` otherwise), array lengths and every offset before touching data;
  critical array access; releases before any JNI call.

Tests

- `ClusterANNQuantizationTests`: nibble packer tests (incl. odd dimension), updated 4-bit round trip, and three
  native tests (native vs Java over 1/2/4 bits x dims 8..1024 x block sizes 1..33 with shuffled partial offsets;
  all-max overflow guard at dim 4096; bad-offset rejection). They skip without the native library.
- `ClusterANNCohereBenchTests`: opt-in recall/latency harness (see below).

## Verification (2026-09-24, macOS arm64, Apple Silicon, NEON udot)

- Standalone C++ harness (independent unpack-and-multiply reference, 7237 checks over 1/2/4 bits x 20 dims x
  block sizes 1..40 x shuffled offsets + all-max cases): NEON, scalar and AVX2 (under Rosetta) all 0 failures.
  AVX-512 compiled with `-mavx512f -mavx512vl -mavx512bw`; assembly contains `vpdpbusd` (VNNI path) and
  `vpmaddubsw` (fallback). Not executed: no AVX-512 hardware here.
- Native library built locally (`./gradlew buildJniLib -Pknn_libs=opensearchknn_simd`, faiss + nmslib
  submodules initialised). `ClusterANNQuantizationTests`: 26 tests, 0 skipped, 0 failures, i.e. the native
  tests ran against the real library.
- Pre-existing failures unrelated to this change: two `ClusterANN1040KnnVectorsFormatTests` reader tests and
  `ClusterANN1040KnnVectorsWriterTests.testWriteCreatesTwoFiles` (fail on the pristine tree too).

### Cohere 100k sanity (768d, MAXIMUM_INNER_PRODUCT, 4-bit, one segment, k=100, 200 timed queries)

| path | recall@100 mean | recall min | p50 us | p90 us | p99 us | mean us |
|---|---|---|---|---|---|---|
| Java nibble loop (`-Dclusterann.native.dot=false`) | 0.8032 | 0.36 | 4473 | 6530 | 8280 | 4547 |
| native NEON udot | 0.8032 | 0.36 | 2138 | 3186 | 3808 | 2236 |

Recall is bit-identical between the two runs, as expected (same integer dot product, same epilogue). p50 latency
is 2.1x lower, p99 2.2x lower. These are end-to-end `IndexSearcher.search(KnnFloatVectorQuery)` timings on a dev
laptop, single thread, index fully in page cache. Index build (100k) took 13 s; `.clap` is 77 MB.

### Cohere 500k sanity (same setup)

| path | recall@100 mean | recall min | p50 us | p90 us | p99 us | mean us |
|---|---|---|---|---|---|---|
| Java nibble loop | 0.7667 | 0.33 | 8585 | 17482 | 38080 | 10611 |
| native NEON udot | 0.7667 | 0.33 | 5797 | 12783 | 20877 | 7204 |

Again identical recall. Speedup is smaller than at 100k (1.5x p50, 1.8x p99): with more clusters probed the
per-block `readBytes` copy, corrections reads and collector work take a larger share, which is exactly what the
mmap-direct phase 2 targets. Index build (500k) took 102 s; `.clap` is 387 MB. The 1M run needs ~10 GB of scratch
disk and failed with "No space left on device" when only 4 GB were free; the disk has since shown 67 GB free, so it
can be retried with the command below and `count=1000000`, `tests.heap.size=20g`.

## How to run the harness

```
./gradlew :test --tests 'org.opensearch.knn.index.codec.KNN1040Codec.ClusterANNCohereBenchTests' \
  -x buildJniTest -x buildJniLib -x cmakeJniLib -x windowsPatches \
  -Dtests.clusterann.bench.train=$PWD/kdy/cohere-1m-train.json \
  -Dtests.clusterann.bench.count=100000 -Dtests.clusterann.bench.queries=220 -Dtests.clusterann.bench.k=100 \
  -Dtests.clusterann.bench.label=native-100k -Dtests.clusterann.bench.out=$PWD/kdy/bench-100k-native.txt \
  -Dtests.heap.size=10g
# Java fallback for comparison:
#   add -Dtests.jvm.argline=-Dclusterann.native.dot=false
```

Knobs: `count`, `queries` (first 20 are warm-up), `k`, `bits`, `space` (`ip`/`l2`), `out`, `label`.
Use `:test` (root project); a bare `test` fails in the subproject because the filter matches nothing there.

## Not yet verified

- AVX-512 (VNNI and BW fallback) on real hardware: build on c6i/c7i and run `ClusterANNQuantizationTests` and
  the harness there. Graviton for the Linux NEON build.
- Multi-threaded / concurrent-segment latency and memory-bandwidth behaviour.
- 1M-vector run (started; results appended to `kdy/bench-1m-native.txt` when done).

## Follow-ups

- Phase 2: read codes straight from the mmap (`MemorySegmentAddressExtractorUtil` on the postings clone,
  `base + filePointer`) instead of `readBytes` into `flatCodesBuf`; add next-block prefetch in the driver.
- Wire 1-bit and 2-bit through the same native call (kernel already supports them).
- Persist an encoding id / bump `VERSION_CURRENT` so old 4-bit segments are rejected rather than mis-scored.

## Port to `3-5-cluster-ann` (navneet1v/k-NN, head f1ab5ec9)

What differs on that branch, and what the port therefore is:

- **4-bit is already PACKED_NIBBLE there.** `OptimizedScalarQuantizedBlockWriter` packs nibbles, the query stays
  unpacked (`ScalarQuantizedCluster.prepareScan`), and `Int4DotProduct.nibble` is the scalar Java kernel. The block
  layout is `lower | upper | add | sum | codes`, identical to what the kernel expects. No format change needed.
- **4-bit is the default there** (`ClusterANNSQEncoder.DEFAULT_BITS = 4`, `DEFAULT_COMPRESSION = x8`).
- The SIMD JNI library on that branch has no SQ or cluster kernels at all; `platform_defs.h` had been removed.
- The `org.opensearch.knn.clusterann` and `index/codec/clusterann` packages are **synced from the external
  SearchServicesKnnVectorFormats repo** (`scripts/clusterann-sync-from-formats.sh`). A future sync will overwrite the
  hook in `ScalarQuantizedBlockScorer` unless the same change is made in formats; the hook was kept to a few lines and
  the logic lives in a new file (`NativeInt4DotProduct`) for that reason.
- ClusterANN tests run under the `clusterAnnTest` Gradle task (JUnit Platform), not `test`.

Files on `3-5-cluster-ann`:

- Native: `jni/include/platform_defs.h` (restored), `jni/include/simd/clusterann_batch_dot_product.h`,
  `jni/src/simd/similarity_function/clusterann_batch_dot_product.cpp` (unchanged from the first branch), JNI entry
  `bulkQuantizedDotProduct` + `bulkQuantizedDotProductKernel` in `org_opensearch_knn_jni_SimdVectorComputeService.cpp/.h`,
  CMake source list.
- Java: `SimdVectorComputeService` natives; new `clusterann/read/block/scalar/NativeInt4DotProduct` (availability flag,
  `-Dclusterann.native.dot=false` kill switch, `bulk(...)`); `ScalarQuantizedBlockScorer.scoreBlock` now collects the
  wanted positions, fills a per-block `rawDotScratch` (native for PACKED_NIBBLE, `Int4DotProduct` otherwise), then
  runs the unchanged epilogue. `KNNSettings.isFaissAVX512Disabled/isFaissAVX512SPRDisabled` got the same
  null-cluster-service guard `isFaissAVX2Disabled` already had; without it the native library could not load in any
  plain JUnit JVM (NPE before `System.loadLibrary`).
- Build: `clusterAnnTest` now sets `java.library.path` to `jni/build/release`, forwards `-Dclusterann.*`, and honours
  `-Dknn.heap`.
- Tests: `NativeInt4DotProductTests` (JUnit 5, `@EnabledIf` on library availability; 7 cases), harness
  `index/codec/clusterann/ClusterANNCohereBenchTests` (knobs are `-Dknn.clusterann.bench.*`).

Verification on this branch (macOS arm64, NEON udot):

- `clusterAnnTest`: `NativeInt4DotProductTests` 7/7 with the real library; `ScalarQuantizedBlockScorerTests` 22/22,
  `Int4DotProductTests` 45/45, `SQScanContext*Tests` 62/62 unchanged.
- Cohere 100k, 768d, MIP, 4-bit, k=100, 200 timed queries, single segment:

| path | recall@100 mean | recall min | p50 us | p90 us | p99 us | mean us |
|---|---|---|---|---|---|---|
| Java `Int4DotProduct.nibble` | 0.8846 | 0.57 | 6319 | 7214 | 7948 | 6344 |
| native NEON udot | 0.8846 | 0.57 | 2695 | 3026 | 3272 | 2715 |

Identical recall; p50 2.3x lower, p99 2.4x lower. Reports: `kdy/bench-3-5-100k-{native,java}.txt`.

How to run on this branch:

```
git submodule update --init --recursive
./gradlew buildJniLib                       # all variants; add -Dbuild.lib.apply_patches=false on a rebuild
./gradlew clusterAnnTest --tests '*NativeInt4DotProductTests' -x buildJniTest -x buildJniLib -x cmakeJniLib -x windowsPatches
./gradlew clusterAnnTest --tests '*ClusterANNCohereBenchTests' -x buildJniTest -x buildJniLib -x cmakeJniLib -x windowsPatches \
  -Dknn.clusterann.bench.train=$PWD/kdy/cohere-1m-train.json -Dknn.clusterann.bench.count=100000 \
  -Dknn.clusterann.bench.label=native-100k -Dknn.clusterann.bench.out=$PWD/kdy/bench-100k-native.txt -Dknn.heap=10g
#   add -Dclusterann.native.dot=false for the Java baseline
```

## Radial search backport (`3-5-cluster-ann-radial`, 2026-09-29)

Radial (`max_distance` / `min_score`) search did not exist for ClusterANN on `3-5-cluster-ann`: the query builder
rejected it ("Engine [UNDEFINED] does not support radial search"), the cluster query factory read only `k`, and the
reader had no threshold path. The POC (`kdy/cluster-ann-poc`, navneet1v `cluster-ann` @ a422a2ae) supported it via
Lucene's `FloatVectorSimilarityQuery` plus a reader branch that ran the ADC scan into a fixed top-100 collector and
exact-rescored those 100. On Cohere 100k that gave 100% recall for radii of 10 and ~95% for 50, but a hard cap of 100
results per segment (recall 0.33 for a 300-document radius).

Backport design (dedicated query, no collector-type sniffing):

- `KNNQueryBuilder`: cluster ANN fields pass the radial engine check; the radius converts with the Lucene engine's
  formulas (`KNNEngine.LUCENE.distanceToRadialThreshold` / `scoreToRadialThreshold`), because cluster ANN scores are
  Lucene-style similarities (L2 `1/(1+d)`, MIP `dot+1` or `1/(1-dot)`, cosine `(1+cos)/2`), the same scale the block
  scorer and `VectorSimilarityFunction.compare` produce.
- `RNNQueryFactory`: a cluster ANN field gets `ClusterANNRadialQuery(field, vector, resultSimilarity, filter, shardId)`.
- `ClusterANNRadialQuery` / `ClusterANNRadialWeight` (`index/query/clusterann`): per segment, builds `AcceptDocs` from
  the filter (conjoined with `FieldExistsQuery`, as `ClusterANNQuery` does), unwraps the leaf to the per-field
  `KNN1030ClusterANNVectorsReader` and calls `searchRadial`. Memoised per leaf, `KNNScorer` replays the hits, `explain`
  reports the exact score. Nested fields are not supported (same as the Lucene engine's radial search).
- `KNN1030ClusterANNVectorsReader.searchRadial(field, query, resultSimilarity, admissionSlack, maxCandidates, acceptDocs)`:
  phase 1 runs the normal probe plan + `ClusterSearcher.search` into a `RadialCandidateCollector` (new,
  `clusterann/read/orchestration`) that admits every vector whose *quantized* score clears the exact threshold widened
  by a relative slack in the raw metric (10% more distance for L2, 10% less dot product for MIP, 0.1 less cosine;
  `RadialCandidateCollector.admissionThreshold`), bounded by `maxCandidates` (weakest dropped first, `truncated()` reported as
  `TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO`); phase 2 scores the candidates against the stored full-precision vectors
  and keeps those at or above `resultSimilarity`. Precision is therefore exact; recall depends on the estimates and the
  slack. The native nibble kernel runs in phase 1 as for top-k.
- Knobs (JVM properties, read once): `clusterann.radial.admission_slack` (default 0.05, scale-free across metrics) and
  `clusterann.radial.max_candidates` (default 10000 per segment). A first version used a plain fraction of the score
  (0.8x); on MIP scores around 140 that admitted nearly every probed vector and the exact pass ran to the cap every
  query (~10 ms p50 on Cohere 100k), which is why the slack moved into the raw metric.
- Why not Lucene's `FloatVectorSimilarityQuery`: its `VectorSimilarityCollector` is package-private, reports `k()=1`, its
  `minCompetitiveSimilarity()` is `-inf` until the first hit, and a search timeout wraps it in a decorator, so the reader
  could not reliably tell a radial request from a top-k one nor read the threshold.

Tests: `RadialCandidateCollectorTests` (JUnit 5), `ClusterANNRadialQueryTests` (2 segments, 8 Gaussian blobs,
4-bit: exact scores, recall >= 0.9 at radii of 10 and 300, a 400-doc radius returns > 300, cap honoured, filters,
self-only radius, explain, equals), `ClusterANNRadialQueryBuilderTests` (max_distance / min_score on L2 and inner
product build the radial query with the expected thresholds; k still builds the rescore query). The Cohere harness has
an opt-in radial pass: `-Dknn.clusterann.bench.radial=true` (ranks 10 / 100 / 1000, recall, precision, latency).

Cohere 100k (768d, MIP, 4-bit, native NEON kernel, one segment, 50 queries per radius; radii chosen at exact ranks
10 / 100 / 1000). Precision was 1.0 in every run (the membership decision is exact):

| admission slack | recall r=10 | recall r=100 | recall r=1000 | p50 us r=10 | p50 us r=100 | p50 us r=1000 |
|---|---|---|---|---|---|---|
| 0.8x score ratio (first version) | 0.968 | 0.934 | 0.865 | 20647 | 10219 | 10646 |
| 0.10 | 0.968 | 0.934 | 0.865 | 6792 | 9473 | 10215 |
| 0.05 (default) | 0.968 | 0.934 | 0.865 | 3812 | 4950 | 9019 |
| 0.03 | 0.968 | 0.934 | 0.865 | 3328 | 3946 | 6213 |
| 0.00 | 0.870 | 0.847 | 0.805 | 3406 | 3289 | 4367 |

Recall is flat from 0.03 up, so the remaining misses are clusters the planner did not probe, not admission; below
~0.03 the estimates start dropping true neighbours. Default set to 0.05. For reference the top-k scan on the same index
runs at 2.7 ms p50, and the POC's radial path measured 15 to 50 ms per query with a hard 100-per-segment cap.
Reports: `kdy/bench-3-5-100k-radial*.txt`.
