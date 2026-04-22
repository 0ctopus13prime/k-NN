/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.lucene104.QuantizedByteVectorValues;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.AlreadyClosedException;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.ResidualQuantizer;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.function.Supplier;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link ErrorResidualReader}.
 *
 * Each test writes a {@code .ver} file using {@link ResidualQuantizer#writeResidualFile} +
 * {@link CodecUtil#writeFooter}, then opens it with {@link ErrorResidualReader} and validates
 * header fields, block reads, cloning, and resource cleanup.
 */
public class ErrorResidualReaderTests extends OpenSearchTestCase {

    private static final String SEGMENT_NAME = "_0";
    private static final String FIELD_NAME = "test_field";

    // ──────────────────────────────────────────────────────────────────────────
    // Header parsing
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Write a .ver file with 2 vectors at dim=4 using empty segment suffix,
     * then open with ErrorResidualReader. Verify all header fields.
     */
    public void testOpenAndParseHeader() throws IOException {
        List<float[]> vectors = List.of(
            new float[] { 1.0f, 2.0f, 3.0f, 4.0f },
            new float[] { 0.5f, 1.5f, 2.5f, 3.5f }
        );
        float[] centroid = new float[] { 0.5f, 0.5f, 0.5f, 0.5f };

        ByteBuffersDirectory dir = new ByteBuffersDirectory();
        writeVerFile(dir, vectors, centroid, new float[] { 0.0f, 0.0f }, new float[] { 1.0f, 1.0f });

        try (ErrorResidualReader reader = new ErrorResidualReader(dir, SEGMENT_NAME, FIELD_NAME, centroid)) {
            assertEquals(4, reader.getDimension());
            assertEquals(2, reader.getNumVectors());
            assertEquals(18, reader.getBytesPerBlock());  // packedResidual(2) + meta(16)
            assertEquals(2, reader.getPackedResidualBytes());
            assertArrayEquals(centroid, reader.getCentroid(), 0.0f);
        }
    }

    /**
     * P1 regression test: write .ver file with a non-empty segment suffix (e.g., the field name),
     * then verify ErrorResidualReader parses the header correctly.
     * This catches the bug where the reader hardcoded empty suffix when computing header length.
     */
    public void testOpenAndParseHeader_nonEmptySuffix() throws IOException {
        List<float[]> vectors = List.of(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        float[] centroid = new float[] { 0.0f, 0.0f, 0.0f, 0.0f };

        ByteBuffersDirectory dir = new ByteBuffersDirectory();
        // Use the 7-arg SegmentWriteState constructor which sets segmentSuffix = FIELD_NAME
        writeVerFileWithSuffix(dir, vectors, centroid, new float[] { 0.0f }, new float[] { 1.0f }, FIELD_NAME);

        try (ErrorResidualReader reader = new ErrorResidualReader(dir, SEGMENT_NAME, FIELD_NAME, centroid)) {
            assertEquals(4, reader.getDimension());
            assertEquals(1, reader.getNumVectors());
            assertEquals(2 + 16, reader.getBytesPerBlock());
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Block reads: packed residual + per-vector metadata
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Write 2 vectors with different correction factors, read each block via ErrorResidualReader,
     * and verify per-vector metadata (lower, upper) and packed residual nibbles.
     *
     * Vec0: [0.5, 0.5, 0.5, 0.5], centroid=[0,0,0,0]
     *   binaryCode=0x05 (bits [1,0,1,0]), lower=0.0, upper=1.0 → delta=1.0
     *   residuals = [-0.5, 0.5, -0.5, 0.5], halfDelta=0.5
     *   q=[0, 15, 0, 15], stored lower=-0.5, upper=0.5
     *
     * Vec1: [1.0, 1.0, 1.0, 1.0], centroid=[0,0,0,0]
     *   binaryCode=0x0F (bits [1,1,1,1]), lower=0.0, upper=2.0 → delta=2.0
     *   residuals = [-1, -1, -1, -1], halfDelta=1.0
     *   q=[0, 0, 0, 0], stored lower=-1.0, upper=1.0
     */
    public void testReadBlock() throws IOException {
        List<float[]> vectors = List.of(
            new float[] { 0.5f, 0.5f, 0.5f, 0.5f },
            new float[] { 1.0f, 1.0f, 1.0f, 1.0f }
        );
        float[] centroid = new float[] { 0.0f, 0.0f, 0.0f, 0.0f };

        ByteBuffersDirectory dir = new ByteBuffersDirectory();
        writeVerFile(
            dir, vectors, centroid,
            new float[] { 0.0f, 0.0f },
            new float[] { 1.0f, 2.0f },
            new byte[][] { new byte[] { 0x05 }, new byte[] { 0x0F } }
        );

        try (ErrorResidualReader reader = new ErrorResidualReader(dir, SEGMENT_NAME, FIELD_NAME, centroid)) {
            try (IndexInput cloned = reader.cloneInput()) {
                // --- Vec0 block ---
                byte[] block0 = reader.readBlock(cloned, 0);
                assertEquals(reader.getBytesPerBlock(), block0.length);
                assertEquals(-0.5f, reader.extractLower(block0), 1e-6f);
                assertEquals(0.5f, reader.extractUpper(block0), 1e-6f);

                // Packed nibbles: q=[0, 15, 0, 15]
                assertEquals(0, block0[0] & 0x0F);
                assertEquals(15, (block0[0] >>> 4) & 0x0F);
                assertEquals(0, block0[1] & 0x0F);
                assertEquals(15, (block0[1] >>> 4) & 0x0F);

                // --- Vec1 block ---
                byte[] block1 = reader.readBlock(cloned, 1);
                assertEquals(-1.0f, reader.extractLower(block1), 1e-6f);
                assertEquals(1.0f, reader.extractUpper(block1), 1e-6f);

                // All nibbles = 0 (residuals at boundary -delta/2)
                for (int d = 0; d < 4; d++) {
                    int nibble = (d % 2 == 0) ? (block1[d / 2] & 0x0F) : ((block1[d / 2] >>> 4) & 0x0F);
                    assertEquals("vec1 dim=" + d, 0, nibble);
                }
            }
        }
    }

    /**
     * P3: Odd dimension test via ErrorResidualReader. dim=5, packedResidualBytes=3.
     * Verifies header, block size, and that the padding nibble (dimension 5) is zero.
     */
    public void testReadBlock_oddDimension() throws IOException {
        List<float[]> vectors = List.of(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });
        float[] centroid = new float[] { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

        ByteBuffersDirectory dir = new ByteBuffersDirectory();
        writeVerFile(dir, vectors, centroid, new float[] { 0.0f }, new float[] { 1.0f });

        try (ErrorResidualReader reader = new ErrorResidualReader(dir, SEGMENT_NAME, FIELD_NAME, centroid)) {
            assertEquals(5, reader.getDimension());
            assertEquals(3, reader.getPackedResidualBytes());          // ceil(5/2)
            assertEquals(3 + 16, reader.getBytesPerBlock());

            try (IndexInput cloned = reader.cloneInput()) {
                byte[] block = reader.readBlock(cloned, 0);

                // Last byte's high nibble (dimension 5, doesn't exist) should be 0
                assertEquals(0, (block[2] >>> 4) & 0x0F);

                // First 5 nibbles should be valid [0, 15]
                for (int d = 0; d < 5; d++) {
                    int nibble = (d % 2 == 0) ? (block[d / 2] & 0x0F) : ((block[d / 2] >>> 4) & 0x0F);
                    assertTrue("dim " + d, nibble >= 0 && nibble <= 15);
                }
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Clone input: thread-safe independent seek positions
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Two cloned inputs read different ordinals independently and produce
     * different per-vector metadata, confirming independent seek positions.
     */
    public void testCloneInput() throws IOException {
        // Two vectors with different correction factors → different per-vector metadata
        List<float[]> vectors = List.of(
            new float[] { 0.5f, 0.5f, 0.5f, 0.5f },
            new float[] { 1.0f, 1.0f, 1.0f, 1.0f }
        );
        float[] centroid = new float[] { 0.0f, 0.0f, 0.0f, 0.0f };

        ByteBuffersDirectory dir = new ByteBuffersDirectory();
        writeVerFile(
            dir, vectors, centroid,
            new float[] { 0.0f, 0.0f },
            new float[] { 1.0f, 2.0f },  // different upper → different per-vector lower/upper
            new byte[][] { new byte[] { 0x05 }, new byte[] { 0x0F } }
        );

        try (ErrorResidualReader reader = new ErrorResidualReader(dir, SEGMENT_NAME, FIELD_NAME, centroid)) {
            try (IndexInput clone1 = reader.cloneInput(); IndexInput clone2 = reader.cloneInput()) {
                // Read vec1 from clone1, then vec0 from clone2 — interleaved, independent seeks
                byte[] block1 = reader.readBlock(clone1, 1);
                byte[] block0 = reader.readBlock(clone2, 0);

                // Vec0 has delta=1.0 → stored lower=-0.5, Vec1 has delta=2.0 → stored lower=-1.0
                // The metadata must differ, proving the clones read independently
                assertEquals(-0.5f, reader.extractLower(block0), 1e-6f);
                assertEquals(-1.0f, reader.extractLower(block1), 1e-6f);
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Close: resource cleanup
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * After close(), calling cloneInput() should throw AlreadyClosedException.
     */
    public void testClose() throws IOException {
        List<float[]> vectors = List.of(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        float[] centroid = new float[] { 0.0f, 0.0f, 0.0f, 0.0f };

        ByteBuffersDirectory dir = new ByteBuffersDirectory();
        writeVerFile(dir, vectors, centroid, new float[] { 0.0f }, new float[] { 1.0f });

        ErrorResidualReader reader = new ErrorResidualReader(dir, SEGMENT_NAME, FIELD_NAME, centroid);
        reader.close();

        expectThrows(AlreadyClosedException.class, reader::cloneInput);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Round-trip dequantization
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * End-to-end: write .ver, read with ErrorResidualReader, dequantize using per-vector
     * lower/upper, verify against true residuals within half a quantization step.
     *
     * Vec=[0.5, 0.5, 0.5, 0.5], centroid=[0,0,0,0]
     * binaryCode=0x05 (bits [1,0,1,0]), lower=0.0, upper=1.0 → delta=1.0
     * residuals = [-0.5, 0.5, -0.5, 0.5] — exact boundary values
     * q=[0, 15, 0, 15] → dequantized = [-0.5, 0.5, -0.5, 0.5]
     */
    public void testRoundTripDequantization() throws IOException {
        List<float[]> vectors = List.of(new float[] { 0.5f, 0.5f, 0.5f, 0.5f });
        float[] centroid = new float[] { 0.0f, 0.0f, 0.0f, 0.0f };

        ByteBuffersDirectory dir = new ByteBuffersDirectory();
        writeVerFile(
            dir, vectors, centroid,
            new float[] { 0.0f }, new float[] { 1.0f },
            new byte[][] { new byte[] { 0x05 } }
        );

        float[] expectedResiduals = { -0.5f, 0.5f, -0.5f, 0.5f };

        try (ErrorResidualReader reader = new ErrorResidualReader(dir, SEGMENT_NAME, FIELD_NAME, centroid)) {
            try (IndexInput cloned = reader.cloneInput()) {
                byte[] block = reader.readBlock(cloned, 0);

                float lower = reader.extractLower(block);
                float upper = reader.extractUpper(block);
                float residualStep = (upper - lower) / 15.0f;

                for (int d = 0; d < 4; d++) {
                    int nibble = (d % 2 == 0)
                        ? (block[d / 2] & 0x0F)
                        : ((block[d / 2] >>> 4) & 0x0F);
                    float reconstructed = lower + nibble * residualStep;
                    assertEquals("dim=" + d, expectedResiduals[d], reconstructed, residualStep / 2.0f + 1e-6f);
                }
            }
        }
    }

    /**
     * Round-trip with non-zero centroid to verify the centering step x' = x - c
     * works correctly through the write→read pipeline.
     *
     * Vec=[2.0, 3.0, 4.0, 5.0], centroid=[1.5, 1.5, 1.5, 1.5]
     * binaryCode=0x05 (bits [1,0,1,0]), lower=0.0, upper=1.0 → delta=1.0
     * x'=[0.5, 1.5, 2.5, 3.5], Q(x')=[1.0, 0.0, 1.0, 0.0]
     * residuals = [-0.5, 1.5, 1.5, 3.5]
     *   → clamped to [-0.5, 0.5]: [-0.5, 0.5(clamped), 0.5(clamped), 0.5(clamped)]
     *   → q = [0, 15, 15, 15]
     */
    public void testRoundTripDequantization_nonZeroCentroid() throws IOException {
        List<float[]> vectors = List.of(new float[] { 2.0f, 3.0f, 4.0f, 5.0f });
        float[] centroid = new float[] { 1.5f, 1.5f, 1.5f, 1.5f };

        ByteBuffersDirectory dir = new ByteBuffersDirectory();
        writeVerFile(
            dir, vectors, centroid,
            new float[] { 0.0f }, new float[] { 1.0f },
            new byte[][] { new byte[] { 0x05 } }
        );

        try (ErrorResidualReader reader = new ErrorResidualReader(dir, SEGMENT_NAME, FIELD_NAME, centroid)) {
            try (IndexInput cloned = reader.cloneInput()) {
                byte[] block = reader.readBlock(cloned, 0);

                float lower = reader.extractLower(block);
                float upper = reader.extractUpper(block);

                // dim0: residual=-0.5 (at boundary) → q=0
                assertEquals(0, block[0] & 0x0F);
                // dim1: residual=1.5 (exceeds halfDelta=0.5) → clamped to q=15
                assertEquals(15, (block[0] >>> 4) & 0x0F);
                // Verify per-vector bounds
                assertEquals(-0.5f, lower, 1e-6f);
                assertEquals(0.5f, upper, 1e-6f);
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────────────────────────────────

    /** Write a .ver file with default binary codes (all zeros) and empty segment suffix. */
    private void writeVerFile(
        ByteBuffersDirectory dir,
        List<float[]> vectors,
        float[] centroid,
        float[] lowerIntervals,
        float[] upperIntervals
    ) throws IOException {
        byte[][] defaultCodes = new byte[vectors.size()][];
        for (int i = 0; i < vectors.size(); i++) {
            defaultCodes[i] = new byte[(vectors.get(0).length + 7) / 8];
        }
        writeVerFile(dir, vectors, centroid, lowerIntervals, upperIntervals, defaultCodes);
    }

    /** Write a .ver file with custom binary codes and empty segment suffix. */
    private void writeVerFile(
        ByteBuffersDirectory dir,
        List<float[]> vectors,
        float[] centroid,
        float[] lowerIntervals,
        float[] upperIntervals,
        byte[][] binaryCodes
    ) throws IOException {
        writeVerFileWithSuffix(dir, vectors, centroid, lowerIntervals, upperIntervals, binaryCodes, "");
    }

    /** Write a .ver file with custom binary codes and a specified segment suffix. */
    private void writeVerFileWithSuffix(
        ByteBuffersDirectory dir,
        List<float[]> vectors,
        float[] centroid,
        float[] lowerIntervals,
        float[] upperIntervals,
        String suffix
    ) throws IOException {
        byte[][] defaultCodes = new byte[vectors.size()][];
        for (int i = 0; i < vectors.size(); i++) {
            defaultCodes[i] = new byte[(vectors.get(0).length + 7) / 8];
        }
        writeVerFileWithSuffix(dir, vectors, centroid, lowerIntervals, upperIntervals, defaultCodes, suffix);
    }

    /** Core write method: writes .ver file with specified suffix, binary codes, and correction factors. */
    private void writeVerFileWithSuffix(
        ByteBuffersDirectory dir,
        List<float[]> vectors,
        float[] centroid,
        float[] lowerIntervals,
        float[] upperIntervals,
        byte[][] binaryCodes,
        String suffix
    ) throws IOException {
        int dimension = vectors.get(0).length;
        int numVectors = vectors.size();

        QuantizedByteVectorValues mockQuantized = createMockQuantizedValues(binaryCodes, lowerIntervals, upperIntervals);
        Supplier<KNNVectorValues<?>> supplier = () -> KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            new TestVectorValues.PreDefinedFloatVectorValues(vectors)
        );

        SegmentWriteState state = createSegmentWriteState(dir, suffix);
        String fileName = SEGMENT_NAME + "_" + FIELD_NAME + ".ver";
        try (IndexOutput output = dir.createOutput(fileName, IOContext.DEFAULT)) {
            ResidualQuantizer.writeResidualFile(output, state, supplier, mockQuantized, centroid, dimension, numVectors);
            CodecUtil.writeFooter(output);
        }
    }

    private SegmentWriteState createSegmentWriteState(ByteBuffersDirectory dir, String suffix) {
        SegmentInfo segmentInfo = new SegmentInfo(
            dir, Version.LATEST, Version.LATEST, SEGMENT_NAME, 0,
            false, false, null, Collections.emptyMap(),
            StringHelper.randomId(), Collections.emptyMap(), null
        );
        if (suffix.isEmpty()) {
            return new SegmentWriteState(InfoStream.NO_OUTPUT, dir, segmentInfo, FieldInfos.EMPTY, null, IOContext.DEFAULT);
        }
        return new SegmentWriteState(InfoStream.NO_OUTPUT, dir, segmentInfo, FieldInfos.EMPTY, null, IOContext.DEFAULT, suffix);
    }

    private QuantizedByteVectorValues createMockQuantizedValues(
        byte[][] binaryCodes, float[] lowerIntervals, float[] upperIntervals
    ) throws IOException {
        QuantizedByteVectorValues mockVal = mock(QuantizedByteVectorValues.class);
        when(mockVal.size()).thenReturn(binaryCodes.length);
        for (int i = 0; i < binaryCodes.length; i++) {
            when(mockVal.vectorValue(i)).thenReturn(binaryCodes[i]);
            when(mockVal.getCorrectiveTerms(i)).thenReturn(
                new OptimizedScalarQuantizer.QuantizationResult(lowerIntervals[i], upperIntervals[i], 0.0f, 0)
            );
        }
        return mockVal;
    }
}
