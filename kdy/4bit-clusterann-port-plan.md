# Native bulk SIMD dot product for the cluster ANN 4-bit index (POC)

Date: 2026-09-24. Scope: ClusterANN Flow A (per-centroid OSQ), the `.clap` block-columnar ADC path, 4-bit
documents. HNSW is out of scope.

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
