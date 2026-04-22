# Error Correction Search — Implementation Plan

Reference: [Error Correction Search Design](error_correction_search_design.md)

---

## Task S1: `ErrorResidualRefiner` interface

**New file:** `src/main/java/org/opensearch/knn/index/codec/nativeindex/ErrorResidualRefiner.java`

Simple interface with one method. Eligibility is determined externally by checking compression
level (x32); if the field is x32, the caller assumes this interface is available and calls `refine()`.

```java
public interface ErrorResidualRefiner {
    ScoreDoc[] refine(String field, float[] queryVector, int[] docIds, float[] phase1Scores)
        throws IOException;
}
```

No dependencies. No tests needed — tested through implementors.

---

## Task S2: `ErrorResidualReader`

**New file:** `src/main/java/org/opensearch/knn/index/codec/KNN1040Codec/ErrorResidualReader.java`

Reads the `.ver` file (codec header + per-vector blocks). Loaded eagerly by the KnnVectorsReader.

### Responsibilities

- Open `.ver` file, validate `CodecUtil` index header, parse `dimension` / `numVectors` / `bytesPerBlock`
- Store centroid (passed in from `QuantizedByteVectorValues.getCentroid()`)
- Provide `readBlock(IndexInput clonedInput, int ordinal)` — seeks to
  `dataStartOffset + ordinal * bytesPerBlock` and reads the full block (packed residual + 16B metadata)
- Provide `cloneInput()` for thread-safe concurrent reads (each search thread clones)
- Implement `Closeable` — closes the underlying `IndexInput`

### Key fields

```java
private final int dimension;
private final int numVectors;
private final int bytesPerBlock;       // packedResidualBytes + 16
private final int packedResidualBytes; // (dimension + 1) / 2
private final long dataStartOffset;    // file pointer after codec header + our 3 int fields
private final float[] centroid;
private final IndexInput verInput;
```

### Per-vector block layout (as written by ResidualQuantizer)

```
[packed 4-bit residual]     packedResidualBytes
[lowerInterval]             4B float (LE)  = -delta/2
[upperInterval]             4B float (LE)  = delta/2
[additionalCorrection]      4B float (LE)  = 0.0 (reserved)
[componentSum]              4B int (LE)
```

### Notes

- No `globalResidualLower` / `globalResidualUpper` — bounds are per-vector, stored in each block's metadata
- The `.ver` file uses `CodecUtil.writeIndexHeader` (not a custom magic), matching the fix from the build side

---

## Task S3: Score refinement utility

**File:** `src/main/java/org/opensearch/knn/index/codec/nativeindex/ResidualQuantizer.java` (add methods)

Add static methods for search-time score correction. These are pure functions, easily unit-testable.

```java
/**
 * Precompute q' = query - centroid. Called once per query per segment.
 */
public static float[] computeQPrime(float[] queryVector, float[] centroid);

/**
 * Compute the corrected score for a single candidate using per-vector bounds.
 *
 * Dequantizes each 4-bit nibble using the per-vector [lower, upper] from the block metadata,
 * then computes the dot product <q', Q_4(r)> and adds it to the phase1 score.
 *
 * @param qPrime          precomputed q' = query - centroid
 * @param packedResidual  4-bit packed residual bytes from .ver block
 * @param lower           per-vector lowerInterval (-delta/2)
 * @param upper           per-vector upperInterval (delta/2)
 * @param phase1Score     score from 1st-phase approximate search
 * @param dimension       vector dimensionality
 * @return corrected score = phase1Score + <q', Q_4(r)>
 */
public static float computeCorrectedScore(
    float[] qPrime, byte[] packedResidual,
    float lower, float upper, float phase1Score, int dimension
);
```

### Inner loop

```java
float residualStep = (upper - lower) / 15.0f;
float correction = 0.0f;
for (int d = 0; d < dimension; d++) {
    int nibble = (d % 2 == 0)
        ? (packedResidual[d / 2] & 0x0F)
        : ((packedResidual[d / 2] >>> 4) & 0x0F);
    float r_d = lower + nibble * residualStep;
    correction += qPrime[d] * r_d;
}
return phase1Score + correction;
```

---

## Task S4: `Faiss1040ScalarQuantizedKnnVectorsReader` changes

**File:** `src/main/java/org/opensearch/knn/index/codec/KNN1040Codec/Faiss1040ScalarQuantizedKnnVectorsReader.java`

### Changes

1. Implement `ErrorResidualRefiner` interface
2. Add field: `private final ErrorResidualReader errorResidualReader` (nullable)
3. In constructor: call `tryLoadErrorResidualReader(state, flatVectorsReader)` to eagerly open `.ver`
4. Implement `refine()`:
   - Clone `IndexInput` from `ErrorResidualReader`
   - Precompute `qPrime = ResidualQuantizer.computeQPrime(queryVector, centroid)` — once per query
   - For each candidate: read block, extract packed residual + per-vector lower/upper from metadata,
     call `ResidualQuantizer.computeCorrectedScore()`
   - Return `ScoreDoc[]` with all refined scores (no top-k filtering — caller handles that)
5. In `close()`: add `errorResidualReader` to `IOUtils.close()`

### `tryLoadErrorResidualReader`

```java
private ErrorResidualReader tryLoadErrorResidualReader(SegmentReadState state, FlatVectorsReader reader) {
    // For each SQ field in fieldInfos:
    //   extract centroid via KNN1040ScalarQuantizedUtils
    //   check if .ver file exists in directory
    //   if yes, return new ErrorResidualReader(directory, segmentName, fieldName, centroid)
    // On IOException, log warning, return null (fall back to standard rescore)
}
```

Depends on S1, S2, S3.

---

## Task S5: `NativeEngineKnnVectorQuery` changes

**File:** `src/main/java/org/opensearch/knn/index/query/nativelib/NativeEngineKnnVectorQuery.java`

### Changes

1. Add `isErrorCorrectionEligible(List<LeafReaderContext>, String field)`:
   - Check first segment with the field
   - Return true if `FieldInfoExtractor.isSQField(fieldInfo)` and `extractSQConfig(fieldInfo).getBits() == 1`
   - x32 compression → always use error correction refinement

2. Add `doRescoreWithErrorCorrection(...)`:
   - Parallel per-segment (same pattern as `doRescore`):
     - Get `KnnVectorsReader` via `PerFieldKnnVectorsFormat.FieldsReader.getFieldReader(field)`
     - Cast to `ErrorResidualRefiner`
     - Extract `docIds[]` and `phase1Scores[]` from `PerLeafResult`
     - Call `refiner.refine(field, queryVector, docIds, phase1Scores)`
     - Wrap result as `PerLeafResult` with `SearchMode.EXACT_SEARCH`
   - Submit via `indexSearcher.getTaskExecutor().invokeAll(tasks)`

3. Branch in `createWeight()` at the rescore step:
   ```java
   if (isErrorCorrectionEligible(leafReaderContexts, knnQuery.getField())) {
       perLeafResults = doRescoreWithErrorCorrection(...);
   } else {
       perLeafResults = doRescore(...);
   }
   ```

Depends on S1, S4.

---

## Task S6: Unit tests for `ErrorResidualReader`

**New file:** `src/test/java/org/opensearch/knn/index/codec/KNN1040Codec/ErrorResidualReaderTests.java`

### Test S6.1: `testOpenAndParseHeader`

Write a `.ver` file using `ResidualQuantizer.writeResidualFile()` + `CodecUtil.writeFooter()`,
then open with `ErrorResidualReader`. Verify dimension, numVectors, bytesPerBlock, centroid.

### Test S6.2: `testReadBlock`

Write 3 vectors, read back block for each ordinal. Verify packed residual bytes and per-vector
metadata (lower, upper, componentSum) match expected values.

### Test S6.3: `testCloneInput`

Verify `cloneInput()` returns an independent `IndexInput` that can seek/read without affecting
the original.

### Test S6.4: `testClose`

Verify `close()` releases the underlying `IndexInput` (subsequent reads throw `AlreadyClosedException`).

---

## Task S7: Unit tests for score refinement

**File:** `src/test/java/org/opensearch/knn/index/codec/nativeindex/ResidualQuantizerTests.java` (extend)

### Test S7.1: `testComputeQPrime`

```
query = [1.0, 2.0, 3.0], centroid = [0.5, 0.5, 0.5]
expected q' = [0.5, 1.5, 2.5]
```

### Test S7.2: `testComputeCorrectedScore`

Hand-computed example with per-vector lower/upper:
```
qPrime = [1.0, -1.0, 0.5, -0.5]
packedResidual: nibbles = [0, 15, 8, 7]
lower = -0.5, upper = 0.5, step = 1.0/15
r = [-0.5, 0.5, 0.0333, -0.0333]
correction = 1.0*(-0.5) + (-1.0)*0.5 + 0.5*0.0333 + (-0.5)*(-0.0333)
phase1Score = 5.0
Verify correctedScore ≈ phase1Score + correction
```

### Test S7.3: `testComputeCorrectedScore_zeroDelta`

Per-vector lower == upper (delta=0). Step = 0, all dequantized residuals = lower.
Verify no division by zero.

---

## Task S8: Integration test

**Extend:** `src/test/java/org/opensearch/knn/integ/ErrorCorrectionBuildIT.java`

Add a test method (or verify the existing one covers it) that:
1. Ingests 500 docs into Faiss HNSW SQ 1-bit (x32) index
2. Force merges to 1 segment
3. Runs KNN search with k=10
4. Verifies 10 results returned with non-zero scores
5. This validates the full pipeline: build writes `.ver`, reader loads it, query branches to
   `doRescoreWithErrorCorrection`, `refine()` produces corrected scores

The existing `testErrorCorrectionFileCreatedOnForceMerge` already covers steps 1-4. Once S4/S5 are
wired in, the search path will automatically use error correction refinement for x32, so the
existing IT becomes an end-to-end validation.

---

## Implementation Order

```
S1: ErrorResidualRefiner interface              [~5 min]
  └─ no dependency
S2: ErrorResidualReader                         [~1 hr]
  └─ no dependency (reads .ver format from build side)
S3: Score refinement utility                    [~30 min]
  └─ no dependency
S6: ErrorResidualReader unit tests              [~1 hr]
  └─ depends on S2 (write alongside)
S7: Score refinement unit tests                 [~30 min]
  └─ depends on S3 (write alongside)
S4: Reader changes                              [~1 hr]
  └─ depends on S1, S2, S3
S5: Query changes                               [~1 hr]
  └─ depends on S4
S8: Integration test                            [~30 min]
  └─ depends on S5, run last
```

S1, S2, S3 can be done in parallel. S6/S7 should be written alongside S2/S3.
