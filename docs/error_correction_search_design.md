# Error Correction Search Design

Reference: [Build Design](error_correction_build_design.md) | [Search Path Notes](search-path.md)

---

## Overview

The error correction refinement **replaces** the existing 2nd-phase rescoring (which loads full-precision
vectors) for x32 (SQ 1-bit) fields. Instead of computing `<q, x>` with full-precision vectors, we compute
a score correction `<q', Q_4(r)>` using 4-bit quantized residuals from the `.ver` file, and add it to the
1st-phase score.

For non-x32 fields, the existing `doRescore` path remains unchanged.

---

## Current 2-Phase Search Flow

```
NativeEngineKnnVectorQuery.createWeight()
  1. firstPassK = ceil(finalK * oversampleFactor)         // e.g., 20 for k=10, factor=2
  2. doSearch(firstPassK)                                  // 1st phase: HNSW with 1-bit SQ
     → per-segment approximate search, returns ScoreDoc[] with quantized scores
  3. reduceToTopK(firstPassK)                              // cross-segment top-firstPassK
  4. doRescore(finalK)                                     // 2nd phase: load full-precision vectors
     → per-segment: ExactSearcher with useQuantizedVectorsForSearch=false
     → calls FloatVectorValues.rescorer(target) → full-precision distance
     → returns ScoreDoc[] with exact scores (replaces phase-1 scores)
  5. reduceToTopK(finalK)
  6. TopDocs.merge()
```

### Problem

Step 4 loads full-precision vectors (3072 bytes/vec for dim=768) causing page faults in memory-constrained
environments. Error correction uses `.ver` file (384 bytes/vec for dim=768) — 8x less IO.

---

## Proposed Change

### Branching logic in `createWeight()`

```java
// In NativeEngineKnnVectorQuery.createWeight(), replace line 123:
//   perLeafResults = doRescore(indexSearcher, leafReaderContexts, knnWeight, perLeafResults, finalK);
// with:

if(isErrorCorrectionEligible(leafReaderContexts, knnQuery.getField())){
perLeafResults =

doRescoreWithErrorCorrection(
    indexSearcher, leafReaderContexts, perLeafResults, knnQuery.getField(
),
        knnQuery.

getQueryVector(),finalK
    );
        }else{
perLeafResults =

doRescore(indexSearcher, leafReaderContexts, knnWeight, perLeafResults, finalK);
}
```

### Detection: `isErrorCorrectionEligible`

Check if the reader for the field is an `ErrorResidualRefiner` (i.e., an SQ 1-bit / x32 field with a
`.ver` file available).

```java
private boolean isErrorCorrectionEligible(List<LeafReaderContext> leaves, String field) {
    // Check first segment as representative — all segments for the same field
    // use the same codec format.
    for (LeafReaderContext leaf : leaves) {
        SegmentReader reader = Lucene.segmentReader(leaf.reader());
        KnnVectorsReader vectorsReader =
            ((PerFieldKnnVectorsFormat.FieldsReader) reader.getVectorReader()).getFieldReader(field);
        if (vectorsReader instanceof ErrorResidualRefiner) {
            return true;
        }
    }
    return false;
}
```

---

## New Interfaces and Classes

### 1. `ErrorResidualRefiner` (interface)

```java
package org.opensearch.knn.index.codec.nativeindex;

/**
 * Implemented by KnnVectorsReader subclasses that can refine 1st-phase scores
 * using quantized error residuals.
 */
public interface ErrorResidualRefiner {

    /**
     * Refine scores for the given documents using error correction residuals.
     *
     * @param field          the vector field name
     * @param queryVector    the original query vector (used to compute q' = query - centroid)
     * @param docIds         segment-local document IDs (dense case: docId == ordinal)
     * @param phase1Scores   corresponding 1st-phase scores from approximate search
     * @return ScoreDoc[] with refined scores, sorted by score descending, trimmed to top-k
     */
    ScoreDoc[] refine(String field, float[] queryVector, int[] docIds, float[] phase1Scores, int k) throws IOException;
}
```

### 2. `ErrorResidualReader`

Encapsulates all state needed for residual-based score refinement. Loaded eagerly by the reader.

```java
package org.opensearch.knn.index.codec.KNN1040Codec;

/**
 * Reads and provides access to the .ver (vector error residual) file for a single field.
 * Opened eagerly when the KnnVectorsReader is constructed. Holds the IndexInput slice,
 * parsed header, and the centroid + correction factors from QuantizedByteVectorValues.
 */
public class ErrorResidualReader implements Closeable {

    // From .ver file header
    private final int dimension;
    private final int numVectors;
    private final int bytesPerResidual;
    private final float globalResidualLower;
    private final float globalResidualUpper;
    private final long dataStartOffset;  // offset after header in IndexInput

    // From QuantizedByteVectorValues (via KNN1040ScalarQuantizedUtils)
    private final float[] centroid;

    // File handle (cloned per-thread for concurrent search)
    private final IndexInput verInput;

    /**
     * Open and parse the .ver file for the given field.
     * Called from Faiss1040ScalarQuantizedKnnVectorsReader constructor.
     *
     * @param directory      segment directory
     * @param segmentName    e.g., "_0"
     * @param fieldName      the knn vector field name
     * @param centroid       centroid vector from QuantizedByteVectorValues
     */
    public ErrorResidualReader(Directory directory, String segmentName, String fieldName, float[] centroid)
        throws IOException;

    /** Clone the IndexInput for thread-safe concurrent reads. */
    public IndexInput cloneInput();

    /** Read the packed 4-bit residual bytes for a single vector by ordinal. */
    public byte[] readResidual(IndexInput clonedInput, int ordinal) throws IOException;

    // Getters
    public int getDimension();

    public int getNumVectors();

    public int getBytesPerResidual();

    public float getGlobalResidualLower();

    public float getGlobalResidualUpper();

    public float[] getCentroid();

    @Override
    public void close() throws IOException;
}
```

### 3. Score refinement math (Java, P0)

Per-candidate refinement:

```java
/**
 * Compute the corrected score for a single candidate.
 *
 * @param queryVector     original query float[]
 * @param centroid        centroid float[]
 * @param packedResidual  4-bit packed residual bytes from .ver file
 * @param phase1Score     score from 1st-phase approximate search
 * @param globalMin       globalResidualLower from .ver header
 * @param globalMax       globalResidualUpper from .ver header
 * @param dimension       vector dimensionality
 * @return corrected score = phase1Score + <q', Q_4(r)>
 */
static float computeCorrectedScore(
    float[] queryVector,
    float[] centroid,
    byte[] packedResidual,
    float phase1Score,
    float globalMin,
    float globalMax,
    int dimension
) {
    float residualStep = (globalMax - globalMin) / 15.0f;
    float correction = 0.0f;

    for (int d = 0; d < dimension; d++) {
        // Unpack 4-bit nibble
        int byteIdx = d / 2;
        int nibble = (d % 2 == 0) ? (packedResidual[byteIdx] & 0x0F) : ((packedResidual[byteIdx] >>> 4) & 0x0F);

        // Dequantize residual
        float r_d = globalMin + nibble * residualStep;

        // q'_d = query_d - centroid_d
        float qPrime_d = queryVector[d] - centroid[d];

        correction += qPrime_d * r_d;
    }

    return phase1Score + correction;
}
```

### 4. `doRescoreWithErrorCorrection` in `NativeEngineKnnVectorQuery`

```java
private List<PerLeafResult> doRescoreWithErrorCorrection(
    final IndexSearcher indexSearcher,
    List<LeafReaderContext> leafReaderContexts,
    List<PerLeafResult> perLeafResults,
    String field,
    float[] queryVector,
    int k
) throws IOException {
    List<Callable<PerLeafResult>> tasks = new ArrayList<>(leafReaderContexts.size());

    for (int i = 0; i < perLeafResults.size(); i++) {
        final int idx = i;
        final LeafReaderContext leaf = leafReaderContexts.get(idx);
        tasks.add(() -> {
            PerLeafResult leafResult = perLeafResults.get(idx);
            if (leafResult.getResult().scoreDocs.length == 0) {
                return leafResult;
            }

            // Get the per-field reader
            SegmentReader segReader = Lucene.segmentReader(leaf.reader());
            KnnVectorsReader vectorsReader =
                ((PerFieldKnnVectorsFormat.FieldsReader) segReader.getVectorReader()).getFieldReader(field);
            ErrorResidualRefiner refiner = (ErrorResidualRefiner) vectorsReader;

            // Extract docIds and phase1Scores from approximate results
            ScoreDoc[] scoreDocs = leafResult.getResult().scoreDocs;
            int[] docIds = new int[scoreDocs.length];
            float[] phase1Scores = new float[scoreDocs.length];
            for (int j = 0; j < scoreDocs.length; j++) {
                docIds[j] = scoreDocs[j].doc;
                phase1Scores[j] = scoreDocs[j].score;
            }

            // Refine using error correction
            ScoreDoc[] refined = refiner.refine(field, queryVector, docIds, phase1Scores, k);

            TopDocs refinedTopDocs = new TopDocs(new TotalHits(refined.length, TotalHits.Relation.EQUAL_TO), refined);
            return new PerLeafResult(
                leafResult.getFilterBits(),
                                     leafResult.getFilterBitsCardinality(),
                                     refinedTopDocs,
                                     PerLeafResult.SearchMode.EXACT_SEARCH
            );
        });
    }
    return indexSearcher.getTaskExecutor().invokeAll(tasks);
}
```

---

## Changes to `Faiss1040ScalarQuantizedKnnVectorsReader`

The reader implements `ErrorResidualRefiner` and eagerly loads the `.ver` file.

```java
public class Faiss1040ScalarQuantizedKnnVectorsReader extends AbstractNativeEnginesKnnVectorsReader
    implements ErrorResidualRefiner {

    private final ErrorResidualReader errorResidualReader;  // nullable if .ver not found

    Faiss1040ScalarQuantizedKnnVectorsReader(SegmentReadState state, FlatVectorsReader flatVectorsReader) {
        super(state, flatVectorsReader);
        // Eager load: try to open .ver file
        this.errorResidualReader = tryLoadErrorResidualReader(state, flatVectorsReader);
    }

    private ErrorResidualReader tryLoadErrorResidualReader(
        SegmentReadState state,
        FlatVectorsReader flatVectorsReader
    ) {
        try {
            // Get centroid from QuantizedByteVectorValues
            for (FieldInfo fi : state.fieldInfos) {
                if (FieldInfoExtractor.isSQField(fi)) {
                    QuantizedByteVectorValues qbvv =
                        KNN1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(flatVectorsReader.getFloatVectorValues(
                            fi.getName()));
                    float[] centroid = qbvv.getCentroid();

                    String verFileName = state.segmentInfo.name + "_" + fi.getName() + ".ver";
                    if (Arrays.asList(state.directory.listAll()).contains(verFileName)) {
                        return new ErrorResidualReader(state.directory, state.segmentInfo.name, fi.getName(), centroid);
                    }
                }
            }
        } catch (IOException e) {
            log.warn("Failed to load error residual reader, falling back to standard rescore", e);
        }
        return null;
    }

    @Override
    public ScoreDoc[] refine(String field, float[] queryVector, int[] docIds, float[] phase1Scores, int k)
        throws IOException {
        if (errorResidualReader == null) {
            throw new IllegalStateException("Error residual reader not available for field: " + field);
        }

        final IndexInput clonedInput = errorResidualReader.cloneInput();
        final float[] centroid = errorResidualReader.getCentroid();
        final float globalMin = errorResidualReader.getGlobalResidualLower();
        final float globalMax = errorResidualReader.getGlobalResidualUpper();
        final int dimension = errorResidualReader.getDimension();

        // Score all candidates
        // Use a priority queue to collect top-k
        PriorityQueue<ScoreDoc> pq = new PriorityQueue<>(k, Comparator.comparingDouble(a -> a.score));

        for (int i = 0; i < docIds.length; i++) {
            int ordinal = docIds[i];  // dense case: docId == ordinal
            byte[] packedResidual = errorResidualReader.readResidual(clonedInput, ordinal);

            float correctedScore = computeCorrectedScore(
                queryVector,
                centroid,
                packedResidual,
                phase1Scores[i],
                globalMin,
                globalMax,
                dimension
            );

            if (pq.size() < k) {
                pq.add(new ScoreDoc(docIds[i], correctedScore));
            } else if (correctedScore > pq.peek().score) {
                pq.poll();
                pq.add(new ScoreDoc(docIds[i], correctedScore));
            }
        }

        // Drain PQ into sorted array (descending by score)
        ScoreDoc[] result = new ScoreDoc[pq.size()];
        for (int i = result.length - 1; i >= 0; i--) {
            result[i] = pq.poll();
        }
        return result;
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(errorResidualReader, flatVectorsReader);
        // ... existing close logic ...
    }
}
```

---

## `.ver` File Read Path

### Seek-based random access by ordinal

```java
// In ErrorResidualReader.readResidual():
public byte[] readResidual(IndexInput clonedInput, int ordinal) throws IOException {
    long offset = dataStartOffset + (long) ordinal * bytesPerResidual;
    clonedInput.seek(offset);
    byte[] residual = new byte[bytesPerResidual];
    clonedInput.readBytes(residual, 0, bytesPerResidual);
    return residual;
}
```

For dim=768: `bytesPerResidual = 384`, `dataStartOffset = 28` (header size).
Seek to `28 + ordinal * 384`, read 384 bytes.

### Prefetch optimization (future, after Java baseline works)

```java
// Before scoring loop, prefetch upcoming vectors
PrefetchHelper.prefetch(
    clonedInput, dataStartOffset, bytesPerResidual, docIds,           // ordinals to prefetch (sorted for IO locality)
    docIds.length
    );
```

### C++ SIMD optimization (future P1)

The inner loop of `computeCorrectedScore` is a dot product between:

- `q'` (float[dimension]) — computed once per query
- `Q_4(r)` (4-bit packed, dimension/2 bytes) — per candidate

This can be pushed to C++ via JNI as a new SIMD function:

```cpp
// Future: jni/include/simd/similarity_function/...
float errorCorrectionDotProduct(
    const float* qPrime,           // query - centroid (float[dim])
    const uint8_t* packedResidual, // 4-bit packed residuals
    int dimension,
    float globalMin,
    float residualStep
);
```

NEON/AVX512 can process 16-32 dimensions per cycle by:

1. Loading 8 bytes of packed residual (16 nibbles = 16 dimensions)
2. Unpacking to 16 x int8
3. Converting to float16/float32
4. FMA with q' values

This is a separate task — the Java baseline establishes correctness first.

---

## `isErrorCorrectionEligible` vs reader `instanceof`

The check `vectorsReader instanceof ErrorResidualRefiner` serves dual purpose:

1. Confirms the reader is SQ 1-bit (only `Faiss1040ScalarQuantizedKnnVectorsReader` implements it)
2. Confirms `.ver` file was found (if `errorResidualReader` is null, the `refine` method would fail)

To make this cleaner, add a method to the interface:

```java
public interface ErrorResidualRefiner {
    boolean isErrorCorrectionAvailable(String field);

    ScoreDoc[] refine(String field, float[] queryVector, int[] docIds, float[] phase1Scores, int k) throws IOException;
}
```

Then the eligibility check becomes:

```java
if(vectorsReader instanceof
ErrorResidualRefiner refiner
    &&refiner.

isErrorCorrectionAvailable(field)){
    return true;
    }
```

---

## Dense-case assumption

This design assumes **docId == ordinal** (all documents have the knn_vector field). This means:

- `docIds[i]` from `ScoreDoc` can be used directly as ordinal for `.ver` file seek
- No ordinal-to-docId mapping needed
- This matches the existing SQ 1-bit code path (which also assumes dense layout for the
  `FaissSQFlat` storage in C++)

If sparse support is needed later, the reader would need an ordinal map (similar to Lucene's
`OrdToDocDISIReaderConfiguration`).

---

## Changes Summary

### Files to modify

| File                                            | Change                                                                                             |
|-------------------------------------------------|----------------------------------------------------------------------------------------------------|
| `NativeEngineKnnVectorQuery.java`               | Add `doRescoreWithErrorCorrection()` and `isErrorCorrectionEligible()`; branch in `createWeight()` |
| `Faiss1040ScalarQuantizedKnnVectorsReader.java` | Implement `ErrorResidualRefiner`; eager-load `ErrorResidualReader`; implement `refine()`           |

### Files to add

| File                        | Purpose                                                                                 |
|-----------------------------|-----------------------------------------------------------------------------------------|
| `ErrorResidualRefiner.java` | Interface: `refine()` + `isErrorCorrectionAvailable()`                                  |
| `ErrorResidualReader.java`  | Encapsulates `.ver` file I/O: open, parse header, seek-read by ordinal, centroid access |

### Files NOT modified

| File                                               | Why                                                                |
|----------------------------------------------------|--------------------------------------------------------------------|
| `KNNWeight.java` / `ExactSearcher.java`            | Error correction bypasses the exact search path entirely           |
| `RescoreContext.java`                              | Oversample factor logic unchanged; only the rescore method changes |
| JNI / C++ code                                     | Java-only for P0; SIMD is P1                                       |
| `BuildIndexParams.java` / `NativeIndexWriter.java` | Search-side only                                                   |

---

## Score flow with error correction

```
1st phase: HNSW search with 1-bit SQ
  → ScoreDoc[] with phase1Scores (quantized approximation)
  → firstPassK candidates per segment

2nd phase: error correction refinement (replaces full-precision rescore)
  For each candidate:
    a. Read 384 bytes from .ver file (4-bit packed residual at ordinal)
    b. Compute q' = queryVector - centroid
    c. For each dimension d:
         nibble = unpack 4-bit value from packedResidual
         r_d = globalMin + nibble * residualStep
         correction += q'_d * r_d
    d. correctedScore = phase1Score + correction
  → top-k by correctedScore per segment

Merge: TopDocs.merge() across segments → final top-k
```

---

## Performance considerations

| Metric               | Full-precision rescore         | Error correction rescore         |
|----------------------|--------------------------------|----------------------------------|
| IO per vector        | 3072 B (dim=768)               | 384 B (dim=768)                  |
| Vectors per 4KB page | ~1.3                           | ~10.6                            |
| Compute              | float dot product (dim floats) | 4-bit unpack + float dot product |
| Extra memory         | none                           | `.ver` IndexInput handle         |

The IO reduction is the primary win. Compute cost is similar (both are O(dimension)), with a small
overhead for nibble unpacking in the error correction path.
