# Error Correction File Build Design

## Background

With 1-bit scalar quantization (SQ), the approximated inner product is:

```
<q, x> ~ <q', Q(x')> + <q, c> + <c, x> - <c, c>
```

where `q' = q - c`, `x' = x - c`, and `Q(x')` is the quantized centered vector.

The residual error vector `r` is defined as:

```
r = x' - Q(x')
```

The true value of `<q, x>` includes an additional term `<q', r>`. If we quantize `r` to 4 bits, we can use it for
2nd-phase rescoring instead of loading full-precision vectors, trading a small accuracy loss for significantly less IO
(384 bytes vs 3072 bytes for dim=768).

This document describes how to build and write the error correction (residual) file during index construction.

---

## Current Build Pipeline

### File outputs today

| File     | Contents                                     | Writer                     |
|----------|----------------------------------------------|----------------------------|
| `.vec`   | Full-precision float32 vectors               | Lucene `FlatVectorsWriter` |
| `.veb`   | 1-bit quantized vectors + correction factors | Lucene `FlatVectorsWriter` |
| `.faiss` | HNSW graph only (no vector storage)          | Native Faiss via JNI       |

### Build flow (existing)

```
Faiss1040ScalarQuantizedKnnVectorsWriter.flush()
  1. flatVectorsWriter.flush() / finish() / close()    --> writes .vec + .veb
  2. Open FlatVectorsReader on .vec/.veb
  3. Extract QuantizedByteVectorValues (reflection)
  4. doFlush() --> NativeIndexWriter.buildAndWriteIndex()
       a. Create IndexOutput for .faiss
       b. Build BuildIndexParams (knnVectorValuesSupplier, quantizedByteVectorValues, ...)
       c. MemOptimizedScalarQuantizedIndexBuildStrategy.buildAndWriteIndex()
            Phase 1: passQuantizedVectorsAndCorrectionFactors()  (quantized → JNI)
            Phase 2: addDocsToSQIndex()                          (docIds → HNSW build)
            Phase 3: writeIndex()                                (HNSW graph → .faiss)
```

### Key data available at build time

Inside `MemOptimizedScalarQuantizedIndexBuildStrategy.buildAndWriteIndex()`:

| Data                   | Source                                              | Access pattern                                                                                       |
|------------------------|-----------------------------------------------------|------------------------------------------------------------------------------------------------------|
| Full-precision vectors | `indexInfo.getKnnVectorValuesSupplier().get()`      | Forward-only iterator; supplier can be called multiple times for fresh iterators                     |
| 1-bit quantized codes  | `QuantizedByteVectorValues.vectorValue(ord)`        | Random access by ordinal                                                                             |
| Correction factors     | `QuantizedByteVectorValues.getCorrectiveTerms(ord)` | Random access; returns `{lowerInterval, upperInterval, additionalCorrection, quantizedComponentSum}` |
| Centroid vector        | `QuantizedByteVectorValues.getCentroid()`           | Single float[] array                                                                                 |
| Dimension              | `knnVectorValues.dimension()`                       | Available after `initializeVectorValues()`                                                           |
| Segment write state    | `indexInfo.getSegmentWriteState()`                  | For creating new IndexOutputs                                                                        |

---

## Proposed Change: Write Error Correction File (`.ver`)

### New file output

| File   | Contents                                                | Size per vector (dim=768)        |
|--------|---------------------------------------------------------|----------------------------------|
| `.ver` | 4-bit quantized residual vectors + per-vector metadata  | 400 bytes (384 packed + 16 meta) |

### Residual computation math

For each vector at ordinal `i`:

```
x       = full-precision vector                    (from knnVectorValuesSupplier)
c       = centroid                                 (from QuantizedByteVectorValues.getCentroid())
x'      = x - c                                   (centered vector)
Q(x')   = dequantize(binaryCode, lowerInterval, upperInterval)  (reconstructed centered vector)
r       = x' - Q(x')                              (residual error)
Q_4(r)  = quantize_4bit(r)                         (4-bit quantized residual)
```

Dequantization for 1-bit SQ:

```
nSteps = 1   (2^1 - 1)
delta  = upperInterval - lowerInterval
Q(x')_d = lowerInterval + bit_d * delta
```

### Residual bounds are analytically known (no scanning pass needed)

For 1-bit SQ with per-vector interval `[lower, upper]`, the quantization step is `delta = upper - lower`.
Each residual component `r_d = x'_d - Q(x')_d` is bounded by `[-delta/2, delta/2]` because the 1-bit
quantizer snaps each component to either `lower` or `upper`, and the maximum error from that snap is
`delta/2`.

This means the 4-bit quantization range is **derived per-vector** from the 1-bit correction factors:
- `residualLower = -delta/2`
- `residualUpper = delta/2`

No global min/max scan is needed. The residual quantization is single-pass.

### Bit unpacking

The 1-bit binary codes in `.veb` are bit-packed (dim 768 = 96 bytes). To compute per-dimension
residuals, we must **unpack** each bit to get the per-dimension quantized value (0 or 1):

```java
int bit = (binaryCode[d / 8] >>> (d % 8)) & 1;
float qVal = lower + bit * delta;
```

### Similarity function scope

This design targets **inner product** (MaxIP) only. For L2, the correction term is
not a simple additive `<q', r>` — it requires additional terms (`-2<Q(x'), r>` and `||r||^2`).
L2 support can be added later as an extension.

### Where to insert the residual computation

The natural insertion point is inside `MemOptimizedScalarQuantizedIndexBuildStrategy.buildAndWriteIndex()`,
**after Phase 3** (writeIndex). At that point:

- The `.faiss` graph is already written
- `QuantizedByteVectorValues` is still available (the reader is closed by the caller, not the strategy)
- We can obtain a fresh `KNNVectorValues` iterator via `indexInfo.getKnnVectorValuesSupplier().get()`

Placing it after Phase 3 keeps the existing build flow unchanged and the residual write is independent of
the HNSW construction.

### Flush and merge coverage

Phase 4 lives inside `MemOptimizedScalarQuantizedIndexBuildStrategy.buildAndWriteIndex()`. Both the flush
path (`NativeIndexWriter.flushIndex()`) and the merge path (`NativeIndexWriter.mergeIndex()`) call
`buildAndWriteIndex()`, so the `.ver` file is created in both cases automatically.

---

## Detailed Build Logic

### Single-pass approach

The residual bounds are analytically known per-vector (see above), so no global scanning pass
is needed. The build iterates all vectors exactly once:

1. Read full-precision vector + 1-bit binary code + correction factors
2. Compute per-dimension residual: `r_d = (x_d - c_d) - (lower + bit_d * delta)`
3. Quantize residual to 4 bits using per-vector bounds `[-delta/2, delta/2]`
4. Pack nibbles + per-vector metadata into a buffer
5. Flush buffer to `IndexOutput` when full

### Step 1: Create `.ver` IndexOutput inside the build strategy

The strategy creates its own `IndexOutput` using segment write state from `BuildIndexParams`.
No changes to `BuildIndexParams` or `NativeIndexWriter` are needed.

File naming convention: `{segmentName}_{fieldName}.ver` (e.g., `_0_target_field.ver`).

```java
// Inside MemOptimizedScalarQuantizedIndexBuildStrategy, after Phase 3
final SegmentWriteState state = indexInfo.getSegmentWriteState();
final String residualFileName = state.segmentInfo.name + "_" + indexInfo.getField() + ".ver";

try (IndexOutput residualOutput = state.directory.createOutput(residualFileName, state.context)) {
    writeResidualFile(residualOutput, indexInfo, binarizedVectorValues, dimension);
    CodecUtil.writeFooter(residualOutput);
}
```

### Step 2: Buffered single-pass write

The write method follows the same batched pattern as `passQuantizedVectorsAndCorrectionFactors`:
accumulate complete vector blocks in a ~64KB byte buffer and flush in bulk.

Each vector block has a fixed size:
```
oneBlockSize = packedResidualBytes + 16 bytes metadata
             = (dimension + 1) / 2 + 16
```

Batch size: `max(1, 65536 / oneBlockSize)` vectors per flush.

```java
private void writeResidualFile(
    final IndexOutput output,
    final BuildIndexParams indexInfo,
    final QuantizedByteVectorValues binarizedVectorValues,
    final int dimension
) throws IOException {
    final float[] centroid = binarizedVectorValues.getCentroid();
    final int numVectors = binarizedVectorValues.size();
    final int packedResidualBytes = (dimension + 1) / 2;
    // Per-vector block: [packed 4-bit residual] [lower(4B)] [upper(4B)] [correction(4B)] [componentSum(4B)]
    final int oneBlockSize = packedResidualBytes + 16;

    // Write file header
    writeHeader(output, dimension, numVectors, oneBlockSize);

    // Allocate batch buffer (~64KB) for buffered writing
    final int batchSize = Math.max(1, (64 * 1024) / oneBlockSize);
    final byte[] buffer = new byte[batchSize * oneBlockSize];

    // Scratch arrays reused per vector (avoid allocation in the loop)
    final float[] residual = new float[dimension];
    final byte[] residualScratch = new byte[dimension];

    final KNNVectorValues<?> vectors = indexInfo.getKnnVectorValuesSupplier().get();
    initializeVectorValues(vectors);

    for (int ord = 0; ord < numVectors; ord++) {
        final int batchIdx = ord % batchSize;
        final int bufOffset = batchIdx * oneBlockSize;

        final float[] fullVec = (float[]) vectors.getVector();
        final byte[] binaryCode = binarizedVectorValues.vectorValue(ord);
        final QuantizationResult terms = binarizedVectorValues.getCorrectiveTerms(ord);
        final float lower = terms.lowerInterval();
        final float upper = terms.upperInterval();
        final float delta = upper - lower;

        // Compute per-dimension residual: r_d = (x_d - c_d) - Q(x')_d
        for (int d = 0; d < dimension; d++) {
            int bit = (binaryCode[d / 8] >>> (d % 8)) & 1;
            float qVal = lower + bit * delta;
            residual[d] = (fullVec[d] - centroid[d]) - qVal;
        }

        // Quantize residual to 4-bit with per-vector bounds [-delta/2, delta/2]
        final float halfDelta = delta / 2.0f;
        int residualComponentSum = quantizeResidualUniform(residual, residualScratch, dimension, delta);

        // Pack 4-bit nibbles: two dimensions per byte, low nibble first
        for (int d = 0; d < dimension; d += 2) {
            int q0 = residualScratch[d] & 0x0F;
            int q1 = (d + 1 < dimension) ? (residualScratch[d + 1] & 0x0F) : 0;
            buffer[bufOffset + d / 2] = (byte) ((q1 << 4) | q0);
        }

        // Write per-vector metadata after the packed residual (little-endian)
        int metaOffset = bufOffset + packedResidualBytes;
        writeFloatLE(buffer, metaOffset, -halfDelta);         // lowerInterval
        writeFloatLE(buffer, metaOffset + 4, halfDelta);      // upperInterval
        writeFloatLE(buffer, metaOffset + 8, 0.0f);           // additionalCorrection (reserved)
        writeIntLE(buffer, metaOffset + 12, residualComponentSum); // componentSum

        // Flush buffer when full or on the last vector
        if (batchIdx == batchSize - 1 || ord == numVectors - 1) {
            int count = batchIdx + 1;
            output.writeBytes(buffer, 0, count * oneBlockSize);
        }

        vectors.nextDoc();
    }
}
```

### Residual quantization helper

Quantizes each residual component uniformly into `[0, 2^bits - 1]` using the per-vector delta:

```java
static int quantizeResidualUniform(float[] residual, byte[] scratch, int dimension, float delta) {
    int nSteps = (1 << ERROR_RESIDUAL_BITS) - 1;  // 15 for 4-bit
    int componentSum = 0;
    for (int d = 0; d < dimension; d++) {
        // Map from [-delta/2, delta/2] → [0, nSteps]
        float normalized = (residual[d] + delta / 2.0f) / delta;
        int q = Math.max(0, Math.min(nSteps, Math.round(normalized * nSteps)));
        scratch[d] = (byte) q;
        componentSum += q;
    }
    return componentSum;
}
```

### Step 3: Integration into `buildAndWriteIndex()`

```java
@Override
public void buildAndWriteIndex(final BuildIndexParams indexInfo) throws IOException, IndexBuildAbortedException {
    // --- existing code: Phase 1-3 (unchanged) ---
    final KNNVectorValues<?> knnVectorValues = indexInfo.getKnnVectorValuesSupplier().get();
    initializeVectorValues(knnVectorValues);
    final QuantizedByteVectorValues binarizedVectorValues = indexInfo.getQuantizedByteVectorValues();
    // ... initFaissSQIndex, doBuildIndex, writeIndex ...

    // --- NEW: Phase 4 — write error correction residuals ---
    final SegmentWriteState state = indexInfo.getSegmentWriteState();
    final String residualFileName = state.segmentInfo.name + "_" + indexInfo.getField() + ".ver";

    try (IndexOutput residualOutput = state.directory.createOutput(residualFileName, state.context)) {
        writeResidualFile(residualOutput, indexInfo, binarizedVectorValues, knnVectorValues.dimension());
        CodecUtil.writeFooter(residualOutput);
    }
}
```

---

## File Format: `.ver` (Vector Error Residuals)

### Header (20 bytes)

```
Offset  Size    Field                       Description
──────  ──────  ──────────────────────────  ─────────────────────────────────────
0       4B      magic                       0x56455231 ("VER1")
4       4B      dimension                   Original vector dimensionality
8       4B      numVectors                  Total vectors in segment
12      1B      bitsPerDimension            Residual quantization bits (4)
13      4B      bytesPerBlock               Total bytes per vector block
17      3B      reserved                    Padding
```

### Per-vector block (repeated numVectors times)

```
Offset  Size                Field               Description
──────  ──────              ──────────────────   ─────────────────────────────────────
0       (dim+1)/2 bytes     packedResidual       4-bit nibbles, low nibble first
P       4B                  lowerInterval        -delta/2 (float, LE)
P+4     4B                  upperInterval        delta/2 (float, LE)
P+8     4B                  additionalCorrection 0.0 (reserved for future use)
P+12    4B                  componentSum         sum of all quantized nibble values
                                                 (P = packedResidualBytes)
```

Block size: `bytesPerBlock = (dimension + 1) / 2 + 16`

This per-vector layout mirrors the `.veb` format (packed quantized data + correction factors),
meaning the existing ADC scoring infrastructure (`FaissSQDistanceComputer`) can potentially be
reused for residual scoring at the SIMD P1 stage.

### 4-bit packing

Two dimensions per byte, low nibble first:

```
byte = (q[d+1] << 4) | q[d]
```

Where `q[d]` is the 4-bit quantized residual for dimension `d`, in `[0, 15]`.

Dequantization at search time:

```
r_d = lowerInterval + q_d * (upperInterval - lowerInterval) / 15
```

---

## Changes Summary

### Files to modify

| File                                                 | Change                                                                                    |
|------------------------------------------------------|-------------------------------------------------------------------------------------------|
| `MemOptimizedScalarQuantizedIndexBuildStrategy.java` | Add `writeResidualFile()` + `quantizeResidualUniform()` methods; call after Phase 3       |
| `ResidualQuantizer.java`                             | Rewrite: remove two-pass logic, add single-pass with per-vector bounds and buffered write |
| `KNNConstants.java`                                  | Add `RESIDUAL_FILE_EXTENSION = ".ver"` (already done)                                     |

### Files to add

None for the build side. The reader side (search-time) is a separate design.

### Files NOT modified

| File                                            | Why                                                                             |
|-------------------------------------------------|---------------------------------------------------------------------------------|
| `BuildIndexParams.java`                         | No changes needed; strategy creates its own IndexOutput                         |
| `NativeIndexWriter.java`                        | No changes needed; strategy handles .ver file creation internally               |
| `Faiss1040ScalarQuantizedKnnVectorsWriter.java` | No change needed; `QuantizedByteVectorValues` lifecycle already spans the build |
| JNI / C++ code                                  | Residual computation and writing is pure Java; no native code needed            |
| Lucene flat vectors format                      | We consume its output, don't modify it                                          |

---

## Search-Time Usage (Brief)

At search time, the 2nd-phase scorer would:

1. Read `.ver` header to get `bytesPerBlock`
2. For each candidate from 1st phase, seek to `headerOffset + ord * bytesPerBlock`
3. Read `bytesPerBlock` bytes (packed residual + per-vector metadata)
4. Extract per-vector `lowerInterval`, `upperInterval`, `componentSum`
5. Compute `<q', Q_4(r)>` by dequantizing each 4-bit nibble:
   ```
   r_d = lowerInterval + q_d * (upperInterval - lowerInterval) / 15
   correction += q'_d * r_d
   ```
6. Add correction to the 1st-phase score: `finalScore = phase1Score + correction`

The per-vector metadata layout mirrors `.veb`, which means at the SIMD P1 stage, the existing
`FaissSQDistanceComputer` ADC formula could potentially be reused for residual scoring.

This is a separate design doc — mentioned here only to confirm the file format supports the read pattern.
