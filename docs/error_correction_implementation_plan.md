# Error Correction Build — Implementation Plan

Reference: [Error Correction Build Design](error_correction_build_design.md)

---

## Task 1: Add residual file extension constant

**File:** `src/main/java/org/opensearch/knn/common/KNNConstants.java`

Add:

```java
public static final String RESIDUAL_FILE_EXTENSION = ".ver";
```

No tests needed — consumed by later tasks.

---

## Task 2: Rewrite `ResidualQuantizer` — single-pass with per-vector bounds

**File:** `src/main/java/org/opensearch/knn/index/codec/nativeindex/ResidualQuantizer.java`

Rewrite to use single-pass, per-vector-bounds approach with buffered writing.
The residual range is analytically derived from the 1-bit correction factors (`delta = upper - lower`,
bounds = `[-delta/2, delta/2]`), so no global scanning pass is needed.

### Key changes from current implementation

- **Remove** `computeGlobalResidualRange()` — no longer needed
- **Remove** global min/max parameters from `writeResidualFile()`
- **Rewrite** `writeResidualFile()` as single-pass with per-vector delta and buffered output
- **Add** `quantizeResidualUniform()` — quantizes a residual vector using per-vector bounds
- **Update** `writeHeader()` — header is now 20 bytes (no global min/max fields)
- **Keep** `unpackBit`, `computeResidual`, `packNibbles` — still useful as low-level helpers

### Revised public API

```java
public class ResidualQuantizer {

    public static final int MAGIC = 0x56455231;
    public static final int HEADER_SIZE = 20;        // reduced: no global min/max
    public static final int BITS_PER_DIMENSION = 4;
    public static final int PER_VECTOR_META_BYTES = 16; // lower(4) + upper(4) + correction(4) + sum(4)

    // Low-level helpers (unchanged)
    public static int unpackBit(byte[] binaryCode, int dimensionIndex);
    public static float computeResidual(float fullVecComponent, float centroidComponent,
        byte[] binaryCode, int dimensionIndex, float lowerInterval, float step);
    public static byte packNibbles(int low, int high);

    /**
     * Quantize a residual vector uniformly into [0, 2^bits - 1] using per-vector delta.
     * Maps from [-delta/2, delta/2] → [0, nSteps].
     *
     * @return componentSum — sum of all quantized values (used in ADC scoring)
     */
    public static int quantizeResidualUniform(float[] residual, byte[] scratch,
        int dimension, float delta);

    /**
     * Single-pass: compute residuals, quantize to 4-bit, write .ver file with buffered output.
     * Per-vector bounds derived from 1-bit correction factors (delta = upper - lower).
     * Writes header + N vector blocks. Caller writes CodecUtil footer after this returns.
     *
     * Buffering: accumulates complete vector blocks in a ~64KB buffer and flushes in bulk,
     * following the same pattern as passQuantizedVectorsAndCorrectionFactors.
     */
    public static void writeResidualFile(
        IndexOutput output,
        Supplier<KNNVectorValues<?>> vectorSupplier,
        QuantizedByteVectorValues quantizedValues,
        float[] centroid,
        int dimension,
        int numVectors
    ) throws IOException;

    /**
     * Write the 20-byte .ver file header (no global min/max).
     */
    static void writeHeader(IndexOutput output, int dimension, int numVectors,
        int bytesPerBlock) throws IOException;
}
```

### Per-vector block layout

Each vector block written to the buffer:

```
[packed 4-bit residual]     (dimension + 1) / 2 bytes
[lowerInterval]             4B float = -delta/2 (LE)
[upperInterval]             4B float = delta/2 (LE)
[additionalCorrection]      4B float = 0.0 (reserved)
[componentSum]              4B int   = sum of quantized nibbles (LE)
```

Block size: `oneBlockSize = (dimension + 1) / 2 + 16`

### Buffered write strategy

Follows `passQuantizedVectorsAndCorrectionFactors` pattern:

```java
final int oneBlockSize = packedResidualBytes + PER_VECTOR_META_BYTES;
final int batchSize = Math.max(1, (64 * 1024) / oneBlockSize);
final byte[] buffer = new byte[batchSize * oneBlockSize];

// Scratch arrays reused per vector (no allocation in loop)
final float[] residual = new float[dimension];
final byte[] residualScratch = new byte[dimension];

for (int ord = 0; ord < numVectors; ord++) {
    int batchIdx = ord % batchSize;
    int bufOffset = batchIdx * oneBlockSize;

    // ... compute residual, quantize, pack nibbles into buffer[bufOffset..] ...
    // ... write per-vector metadata at buffer[bufOffset + packedResidualBytes..] ...

    if (batchIdx == batchSize - 1 || ord == numVectors - 1) {
        output.writeBytes(buffer, 0, (batchIdx + 1) * oneBlockSize);
    }
}
```

---

## Task 3: Integrate into `MemOptimizedScalarQuantizedIndexBuildStrategy`

**File:** `src/main/java/org/opensearch/knn/index/codec/nativeindex/MemOptimizedScalarQuantizedIndexBuildStrategy.java`

Add Phase 4 at the end of `buildAndWriteIndex()`, after the existing `writeIndex()` call.
The call is simple — all logic is in `ResidualQuantizer.writeResidualFile()`:

```java
// Phase 4: write error correction residuals (.ver file)
final SegmentWriteState state = indexInfo.getSegmentWriteState();
final String residualFileName = state.segmentInfo.name + "_" + indexInfo.getField()
    + KNNConstants.RESIDUAL_FILE_EXTENSION;

try (IndexOutput residualOutput = state.directory.createOutput(residualFileName, state.context)) {
    ResidualQuantizer.writeResidualFile(
        residualOutput,
        indexInfo.getKnnVectorValuesSupplier(),
        binarizedVectorValues,
        binarizedVectorValues.getCentroid(),
        knnVectorValues.dimension(),
        binarizedVectorValues.size()
    );
    CodecUtil.writeFooter(residualOutput);
}
```

No changes to `BuildIndexParams`, `NativeIndexWriter`, or `Faiss1040ScalarQuantizedKnnVectorsWriter`.

---

## Task 4: Update unit tests for `ResidualQuantizer`

**File:** `src/test/java/org/opensearch/knn/index/codec/nativeindex/ResidualQuantizerTests.java`

Update existing tests and add new ones to cover the single-pass, per-vector-bounds approach.

### Tests to keep (low-level helpers unchanged)

- `testUnpackBit` — unchanged
- `testUnpackBit_multipleBytes` — unchanged
- `testComputeResidual` — unchanged
- `testComputeResidual_bitZero` — unchanged
- `testPackNibbles` — unchanged

### Tests to add

#### Test 4.1: `testQuantizeResidualUniform`

Verify per-vector uniform quantization with known delta:

```
delta = 2.0, residual = [0.0, -1.0, 1.0, 0.5]
halfDelta = 1.0, nSteps = 15
normalized = [(0+1)/2, (-1+1)/2, (1+1)/2, (0.5+1)/2] = [0.5, 0.0, 1.0, 0.75]
q = [round(7.5), 0, 15, round(11.25)] = [8, 0, 15, 11]
componentSum = 8 + 0 + 15 + 11 = 34
```

Also test delta=0 edge case (all residuals zero).

#### Test 4.2: `testWriteResidualFile_headerFormat`

Rewrite: verify 20-byte header (no global min/max). Check magic, dimension, numVectors,
bitsPerDimension, bytesPerBlock.

#### Test 4.3: `testWriteResidualFile_perVectorMetadata`

Write 1 vector, read back the block, verify:
- `lowerInterval = -delta/2`
- `upperInterval = delta/2`
- `additionalCorrection = 0.0`
- `componentSum` matches hand-computed value

#### Test 4.4: `testWriteResidualFile_roundTrip`

Rewrite: dequantize using per-vector lower/upper (not global), verify accuracy
within half a quantization step.

#### Test 4.5: `testWriteResidualFile_oddDimension`

Rewrite: same logic but verify per-vector metadata at the correct offset.

#### Test 4.6: `testWriteResidualFile_totalFileSize`

Verify: `headerSize(20) + numVectors * bytesPerBlock + codecFooter(16)`.

### Tests to remove

- `testQuantize4Bit` — replaced by `testQuantizeResidualUniform`
- `testQuantize4Bit_clamping` — clamping tested within new quantizer
- `testQuantize4Bit_zeroStep` — zero-delta tested within new quantizer
- `testComputeGlobalResidualRange` — method removed
- `testWriteResidualFile_singleVector` — can be folded into round-trip test

---

## Task 5: Unit test for `.ver` file creation in build strategy

**File:**
`src/test/java/org/opensearch/knn/index/codec/nativeindex/MemOptimizedScalarQuantizedIndexBuildStrategyTests.java`

### Test 5.1: `testBuildAndWriteIndex_createsResidualFile`

Follow the existing test pattern in `MemOptimizedScalarQuantizedIndexBuildStrategyTests`:

1. Create a Lucene `Directory` and `SegmentWriteState`
2. Write vectors using Lucene's SQ flat writer (produces `.vec` + `.veb`)
3. Open reader, extract `QuantizedByteVectorValues`
4. Build `BuildIndexParams` (same as existing tests)
5. Call `buildAndWriteIndex()`
6. Assert `.ver` file exists in directory with non-zero size:
   ```java
   final String residualFileName = segmentName + "_" + FIELD_NAME + ".ver";
   assertTrue(Arrays.asList(directory.listAll()).contains(residualFileName));
   assertTrue(directory.fileLength(residualFileName) > 0);
   ```

### Test 5.2: `testBuildAndWriteIndex_residualFileHasCorrectHeader`

Same setup as 5.1 but also read back the `.ver` header (20 bytes):

```java
try (IndexInput input = directory.openInput(residualFileName, IOContext.DEFAULT)) {
    assertEquals(ResidualQuantizer.MAGIC, input.readInt());
    assertEquals(dimension, input.readInt());
    assertEquals(numVectors, input.readInt());
    assertEquals(4, input.readByte());  // bits per dimension
    int bytesPerBlock = input.readInt();
    assertEquals((dimension + 1) / 2 + 16, bytesPerBlock);
}
```

### Test 5.3: `testBuildAndWriteIndex_residualFileHasCorrectSize`

Verify total file size = header (20) + numVectors * bytesPerBlock + CodecUtil footer (16).

---

## Task 6: Integration test

**New file:** `src/test/java/org/opensearch/knn/integ/ErrorCorrectionBuildIT.java`

Single IT class with one test method that validates end-to-end `.ver` file creation.

### Test: `testErrorCorrectionFileCreatedOnForceMerge`

```java
public class ErrorCorrectionBuildIT extends KNNRestTestCase {

    private static final String INDEX_NAME = "error-correction-test";
    private static final String FIELD_NAME = "target_field";
    private static final int DIMENSION = 128;
    private static final int NUM_DOCS = 500;

    public void testErrorCorrectionFileCreatedOnForceMerge() throws Exception {
        // 1. Create index with Faiss HNSW + SQ 1-bit encoder
        String mapping = buildMapping(
            FIELD_NAME,
            DIMENSION,
            "faiss",
            "hnsw",
            SpaceType.INNER_PRODUCT,
            """{"encoder": {"name": "sq", "parameters": {"bits": 1}}}"""
        );
        createKnnIndex(INDEX_NAME, settings(...),mapping);

        // 2. Ingest 500 docs with random vectors
        addKNNDocs(INDEX_NAME, FIELD_NAME, DIMENSION, 0, NUM_DOCS);

        // 3. Force merge to single segment
        forceMergeKnnIndex(INDEX_NAME, 1);

        // 4. Verify .ver file exists
        //    Use cat segments API or stats to confirm segment files
        //    Alternatively, verify via search that the index is functional
        //    (the .ver file should not break existing search)
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, NUM_DOCS, 10);

        // 5. Verify .ver file exists via segment files API
        //    GET /{index}/_segments to get segment names
        //    Then check cluster stats or filesystem for .ver files
        Response segmentsResponse = client().performRequest(new Request("GET", "/" + INDEX_NAME + "/_segments"));
        String responseBody = EntityUtils.toString(segmentsResponse.getEntity());
        // Parse segment names from response
        // Verify at least one .ver file reference exists

        // 6. Verify existing search still works after force merge
        //    (regression check — .ver creation should not break anything)
        float[] queryVector = new float[DIMENSION];
        // fill with random values
        Response searchResponse = searchKNNIndex(INDEX_NAME, FIELD_NAME, queryVector, 10);
        // Assert we get 10 results back
    }
}
```

**What this IT validates:**

- `.ver` file is created during segment merge without errors
- Existing `.faiss` HNSW search still works (no regression from Phase 4)
- The build pipeline handles 500 docs at dim=128 without OOM or other failures

**What this IT does NOT validate (deferred to search-time design):**

- Reading the `.ver` file at search time
- Recall improvement from error correction rescoring

---

## Implementation Order

```
Task 1: KNNConstants                               [done]
  └─ no dependency
Task 2: Rewrite ResidualQuantizer (single-pass)    [~2 hr]
  └─ depends on Task 1
Task 4: Update ResidualQuantizer unit tests        [~1.5 hr]
  └─ depends on Task 2 (write tests alongside implementation)
Task 3: Build strategy integration                  [~30 min]
  └─ depends on Task 2
Task 5: Build strategy unit tests                   [~1 hr]
  └─ depends on Task 3
Task 6: Integration test                            [~1 hr]
  └─ depends on Task 3, run last
```

Tasks 2 and 4 should be done together (write tests as you implement each method).
Task 6 runs last and requires a full build (`./gradlew build`).
