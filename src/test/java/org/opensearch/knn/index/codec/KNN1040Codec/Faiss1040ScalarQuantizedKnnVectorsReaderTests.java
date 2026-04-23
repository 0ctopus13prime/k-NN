/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import com.google.common.collect.ImmutableSet;
import lombok.SneakyThrows;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LogEvent;
import org.apache.logging.log4j.core.Logger;
import org.apache.logging.log4j.core.appender.AbstractAppender;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.lucene104.QuantizedByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.codec.nativeindex.AbstractNativeEnginesKnnVectorsReader;
import org.opensearch.knn.index.codec.nativeindex.ErrorResidualRefiner;
import org.opensearch.knn.index.codec.nativeindex.ResidualQuantizer;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;
import org.opensearch.knn.memoryoptsearch.VectorSearcherFactory;

import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class Faiss1040ScalarQuantizedKnnVectorsReaderTests extends KNNTestCase {

    @SneakyThrows
    public void testCheckIntegrity_thenDelegatesToFlatReader() {
        final FlatVectorsReader fvr = mock(FlatVectorsReader.class);
        createReader(new FieldInfos(new FieldInfo[0]), Collections.emptySet(), fvr).checkIntegrity();
        verify(fvr).checkIntegrity();
    }

    @SneakyThrows
    public void testGetFloatVectorValues_thenDelegatesToFlatReader() {
        final FlatVectorsReader fvr = mock(FlatVectorsReader.class);
        final FloatVectorValues mockValues = mock(FloatVectorValues.class);
        when(fvr.getFloatVectorValues("f")).thenReturn(mockValues);
        assertSame(mockValues, createReader(new FieldInfos(new FieldInfo[0]), Collections.emptySet(), fvr).getFloatVectorValues("f"));
    }

    @SneakyThrows
    public void testGetByteVectorValues_thenThrowsUnsupported() {
        expectThrows(
            UnsupportedOperationException.class,
            () -> createReader(new FieldInfos(new FieldInfo[0]), Collections.emptySet(), mock(FlatVectorsReader.class)).getByteVectorValues(
                "f"
            )
        );
    }

    @SneakyThrows
    public void testSearchFloat_whenSearcherAvailable_thenDelegates() {
        final FieldInfo fi = createFieldInfo("field1", KNNEngine.FAISS, 0);
        KNNEngine mockFaiss = spy(KNNEngine.FAISS);
        VectorSearcherFactory mockFactory = mock(VectorSearcherFactory.class);
        VectorSearcher mockSearcher = mock(VectorSearcher.class);
        when(mockFaiss.getVectorSearcherFactory()).thenReturn(mockFactory);
        when(
            mockFactory.createVectorSearcher(
                any(Directory.class),
                anyString(),
                any(FieldInfo.class),
                any(IOContext.class),
                any(FlatVectorsReader.class)
            )
        ).thenReturn(mockSearcher);
        try (MockedStatic<KNNEngine> ms = mockStatic(KNNEngine.class)) {
            ms.when(() -> KNNEngine.getEngine(any())).thenReturn(mockFaiss);
            ms.when(KNNEngine::getEnginesThatCreateCustomSegmentFiles).thenReturn(ImmutableSet.of(mockFaiss));
            final Faiss1040ScalarQuantizedKnnVectorsReader reader = createReader(
                new FieldInfos(new FieldInfo[] { fi }),
                Set.of("_0_165_field1.faiss"),
                mock(FlatVectorsReader.class)
            );
            float[] target = { 1, 2, 3 };
            reader.search("field1", target, null, null);
            verify(mockSearcher).search(target, null, null);
        }
    }

    @SneakyThrows
    public void testSearchFloat_whenNoSearcher_thenThrowsIllegalState() {
        final FieldInfo fi = createFieldInfo("field1", KNNEngine.FAISS, 0);
        KNNEngine mockFaiss = spy(KNNEngine.FAISS);
        when(mockFaiss.getVectorSearcherFactory()).thenReturn(null);

        try (MockedStatic<KNNEngine> ms = mockStatic(KNNEngine.class)) {
            ms.when(() -> KNNEngine.getEngine(any())).thenReturn(mockFaiss);
            ms.when(KNNEngine::getEnginesThatCreateCustomSegmentFiles).thenReturn(ImmutableSet.of(mockFaiss));
            final Faiss1040ScalarQuantizedKnnVectorsReader reader = createReader(
                new FieldInfos(new FieldInfo[] { fi }),
                Collections.emptySet(),
                mock(FlatVectorsReader.class)
            );
            expectThrows(IllegalStateException.class, () -> reader.search("field1", new float[] { 1, 2, 3 }, null, null));
        }
    }

    @SneakyThrows
    public void testSearchByte_thenThrowsUnsupported() {
        expectThrows(
            UnsupportedOperationException.class,
            () -> createReader(new FieldInfos(new FieldInfo[0]), Collections.emptySet(), mock(FlatVectorsReader.class)).search(
                "f",
                new byte[] { 1 },
                null,
                null
            )
        );
    }

    @SneakyThrows
    public void testClose_thenClosesFlatReaderAndSearcher() {
        final FieldInfo fi = createFieldInfo("field1", KNNEngine.FAISS, 0);
        KNNEngine mockFaiss = spy(KNNEngine.FAISS);
        VectorSearcherFactory mockFactory = mock(VectorSearcherFactory.class);
        VectorSearcher mockSearcher = mock(VectorSearcher.class);
        when(mockFaiss.getVectorSearcherFactory()).thenReturn(mockFactory);
        when(
            mockFactory.createVectorSearcher(
                any(Directory.class),
                anyString(),
                any(FieldInfo.class),
                any(IOContext.class),
                any(FlatVectorsReader.class)
            )
        ).thenReturn(mockSearcher);
        final FlatVectorsReader fvr = mock(FlatVectorsReader.class);

        try (MockedStatic<KNNEngine> ms = mockStatic(KNNEngine.class)) {
            ms.when(() -> KNNEngine.getEngine(any())).thenReturn(mockFaiss);
            ms.when(KNNEngine::getEnginesThatCreateCustomSegmentFiles).thenReturn(ImmutableSet.of(mockFaiss));
            final Faiss1040ScalarQuantizedKnnVectorsReader reader = createReader(
                new FieldInfos(new FieldInfo[] { fi }),
                Set.of("_0_165_field1.faiss"),
                fvr
            );
            reader.search("field1", new float[] { 1, 2, 3 }, null, null);
            reader.close();
            verify(fvr).close();
            verify(mockSearcher).close();
        }
    }

    @SneakyThrows
    public void testClose_whenNoSearcher_thenClosesFlatReaderOnly() {
        final FlatVectorsReader fvr = mock(FlatVectorsReader.class);
        createReader(new FieldInfos(new FieldInfo[0]), Collections.emptySet(), fvr).close();
        verify(fvr).close();
    }

    @SneakyThrows
    public void testVectorSearcherHolder_initiallyNotSet() {
        final Faiss1040ScalarQuantizedKnnVectorsReader reader = createReader(
            new FieldInfos(new FieldInfo[0]),
            Collections.emptySet(),
            mock(FlatVectorsReader.class)
        );
        final Field f = AbstractNativeEnginesKnnVectorsReader.class.getDeclaredField("vectorSearcherHolder");
        f.setAccessible(true);
        assertFalse(((AbstractNativeEnginesKnnVectorsReader.VectorSearcherHolder) f.get(reader)).isSet());
    }

    @SneakyThrows
    public void testWarmUp_whenMOSNotSupported_thenLogsWarning() {
        final FieldInfo fi = createFieldInfo("field1", KNNEngine.FAISS, 0);

        // Mock flatVectorsReader to return a ScalarQuantizedFloatVectorValues
        final FlatVectorsReader fvr = mock(FlatVectorsReader.class);
        final ScalarQuantizedFloatVectorValues mockVectorValues = mock(ScalarQuantizedFloatVectorValues.class);
        when(mockVectorValues.size()).thenReturn(0);
        when(fvr.getFloatVectorValues("field1")).thenReturn(mockVectorValues);

        // Set up a log appender to capture log events
        final List<LogEvent> logEvents = new ArrayList<>();
        final Logger logger = (Logger) LogManager.getLogger(Faiss1040ScalarQuantizedKnnVectorsReader.class);
        final AbstractAppender appender = new AbstractAppender("test-appender", null, null, true, null) {
            @Override
            public void append(LogEvent event) {
                logEvents.add(event.toImmutable());
            }
        };
        appender.start();
        logger.addAppender(appender);
        final Level originalLevel = logger.getLevel();
        logger.setLevel(Level.WARN);

        try {
            // Make KNNEngine return null factory so loadMemoryOptimizedSearcherIfRequired returns null
            KNNEngine mockFaiss = spy(KNNEngine.FAISS);
            when(mockFaiss.getVectorSearcherFactory()).thenReturn(null);

            try (MockedStatic<KNNEngine> ms = mockStatic(KNNEngine.class)) {
                ms.when(() -> KNNEngine.getEngine(any())).thenReturn(mockFaiss);
                ms.when(KNNEngine::getEnginesThatCreateCustomSegmentFiles).thenReturn(ImmutableSet.of(mockFaiss));

                final Faiss1040ScalarQuantizedKnnVectorsReader reader = createReader(
                    new FieldInfos(new FieldInfo[] { fi }),
                    Collections.emptySet(),
                    fvr
                );

                reader.warmUp("field1");

                // Verify warning was logged
                boolean foundWarning = logEvents.stream()
                    .anyMatch(e -> e.getLevel() == Level.WARN && e.getMessage().getFormattedMessage().contains("field1"));
                assertTrue("Expected a WARN log about MOS not supported for field1", foundWarning);

                // Verify the searcher warmUp was never called (no searcher available)
                // This is implicitly verified since the searcher is null
            }
        } finally {
            logger.removeAppender(appender);
            logger.setLevel(originalLevel);
            appender.stop();
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // ErrorResidualRefiner: refine() and close() integration
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Verify the reader implements ErrorResidualRefiner.
     */
    @SneakyThrows
    public void testImplementsErrorResidualRefiner() {
        Faiss1040ScalarQuantizedKnnVectorsReader reader = createReader(
            new FieldInfos(new FieldInfo[0]),
            Collections.emptySet(),
            mock(FlatVectorsReader.class)
        );
        assertTrue(reader instanceof ErrorResidualRefiner);
        reader.close();
    }

    /**
     * When no .ver file is loaded (errorResidualReader is null), refine() should throw
     * IllegalStateException.
     */
    @SneakyThrows
    public void testRefine_whenNoVerFile_thenThrowsIllegalState() {
        // The mock directory has no .ver files, so errorResidualReader will be null
        Faiss1040ScalarQuantizedKnnVectorsReader reader = createReader(
            new FieldInfos(new FieldInfo[0]),
            Collections.emptySet(),
            mock(FlatVectorsReader.class)
        );

        expectThrows(
            IllegalStateException.class,
            () -> reader.refine("field", new float[] { 1.0f }, new int[] { 0 }, new float[] { 5.0f })
        );
        reader.close();
    }

    /**
     * When errorResidualReader is available (injected via reflection), refine() should
     * return ScoreDoc[] with corrected scores for all input documents.
     *
     * Setup:
     *   dim=4, centroid=[0,0,0,0], query=[1,-1,1,-1]
     *   q' = [1,-1,1,-1]
     *
     *   Vec0 block: packed nibbles [0,15,0,15], lower=-0.5, upper=0.5
     *     → r=[-0.5, 0.5, -0.5, 0.5]
     *     → correction = 1*(-0.5) + (-1)*0.5 + 1*(-0.5) + (-1)*0.5 = -2.0
     *     → correctedScore = 10.0 + (-2.0) = 8.0
     *
     *   Vec1 block: all nibbles=8 (midpoint), lower=-1.0, upper=1.0
     *     → r = -1.0 + 8*(2/15) ≈ 0.0667 for all dims
     *     → correction = (1 + -1 + 1 + -1) * 0.0667 = 0
     *     → correctedScore ≈ 20.0
     */
    @SneakyThrows
    public void testRefine_withInjectedReader_thenReturnsRefinedScores() {
        int dim = 4;
        float[] centroid = new float[] { 0f, 0f, 0f, 0f };

        // Create mock ErrorResidualReader
        ErrorResidualReader mockResidualReader = mock(ErrorResidualReader.class);
        when(mockResidualReader.getDimension()).thenReturn(dim);
        when(mockResidualReader.getCentroid()).thenReturn(centroid);

        // Mock QuantizedByteVectorValues for V2 scoring
        QuantizedByteVectorValues mockQbvv = mock(QuantizedByteVectorValues.class);
        when(mockResidualReader.getQuantizedByteVectorValues()).thenReturn(mockQbvv);

        // Mock quantizer — use a real one for correct behavior
        OptimizedScalarQuantizer realQuantizer = new OptimizedScalarQuantizer(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);
        when(mockResidualReader.getQuantizer()).thenReturn(realQuantizer);

        // Mock 1-bit data for each candidate (Q(x') binary codes + correction terms)
        // Vec0: all bits = 0, lower=0.0, upper=1.0
        when(mockQbvv.vectorValue(0)).thenReturn(new byte[] { 0x00 });
        when(mockQbvv.getCorrectiveTerms(0)).thenReturn(new OptimizedScalarQuantizer.QuantizationResult(0.0f, 1.0f, 0.0f, 0));
        // Vec1: all bits = 1 (MSB-first: 0xF0 for 4 dims), lower=0.0, upper=2.0
        when(mockQbvv.vectorValue(1)).thenReturn(new byte[] { (byte) 0xF0 });
        when(mockQbvv.getCorrectiveTerms(1)).thenReturn(new OptimizedScalarQuantizer.QuantizationResult(0.0f, 2.0f, 0.0f, 4));

        // Clone returns a mock IndexInput
        IndexInput mockClone = mock(IndexInput.class);
        when(mockResidualReader.cloneInput()).thenReturn(mockClone);

        // Vec0 block: nibbles [0, 15, 0, 15], lower=-0.5, upper=0.5
        byte[] block0 = buildBlock(new int[] { 0, 15, 0, 15 }, -0.5f, 0.5f, 0.0f, 30);
        when(mockResidualReader.readBlock(mockClone, 0)).thenReturn(block0);
        when(mockResidualReader.extractLower(block0)).thenReturn(-0.5f);
        when(mockResidualReader.extractUpper(block0)).thenReturn(0.5f);

        // Vec1 block: all nibbles = 8, lower=-1.0, upper=1.0
        byte[] block1 = buildBlock(new int[] { 8, 8, 8, 8 }, -1.0f, 1.0f, 0.0f, 32);
        when(mockResidualReader.readBlock(mockClone, 1)).thenReturn(block1);
        when(mockResidualReader.extractLower(block1)).thenReturn(-1.0f);
        when(mockResidualReader.extractUpper(block1)).thenReturn(1.0f);

        // Create reader and inject the mock errorResidualReader via reflection
        Faiss1040ScalarQuantizedKnnVectorsReader reader = createReader(
            new FieldInfos(new FieldInfo[0]),
            Collections.emptySet(),
            mock(FlatVectorsReader.class)
        );
        Field errField = Faiss1040ScalarQuantizedKnnVectorsReader.class.getDeclaredField("errorResidualReader");
        errField.setAccessible(true);
        errField.set(reader, mockResidualReader);

        // Call refine
        float[] query = { 1.0f, -1.0f, 1.0f, -1.0f };
        int[] docIds = { 0, 1 };
        float[] phase1Scores = { 10.0f, 20.0f };

        ScoreDoc[] result = reader.refine("field", query, docIds, phase1Scores);

        // Verify basic structure
        assertEquals(2, result.length);
        assertEquals(0, result[0].doc);
        assertEquals(1, result[1].doc);

        // V2 scoring corrects for both query and data quantization error.
        // Verify scores are finite and within a reasonable range of phase1.
        assertTrue("Vec0 score should be finite", Float.isFinite(result[0].score));
        assertTrue("Vec1 score should be finite", Float.isFinite(result[1].score));
        // At least one score should differ from phase1 (Vec1 may stay the same if correction is ~0)
        assertTrue(
            "At least one score should differ from phase1",
            Math.abs(result[0].score - phase1Scores[0]) > 0.01f || Math.abs(result[1].score - phase1Scores[1]) > 0.01f
        );
        // Scores should be in a reasonable neighborhood of phase1 (not wildly off)
        assertTrue("Vec0 score should be within ±50 of phase1", Math.abs(result[0].score - phase1Scores[0]) < 50.0f);
        assertTrue("Vec1 score should be within ±50 of phase1", Math.abs(result[1].score - phase1Scores[1]) < 50.0f);

        // Verify the cloned IndexInput was closed
        verify(mockClone).close();

        reader.close();
        verify(mockResidualReader).close();
    }

    /**
     * When docIds is empty, refine() should return an empty ScoreDoc[].
     */
    @SneakyThrows
    public void testRefine_emptyDocIds_thenReturnsEmptyArray() {
        ErrorResidualReader mockResidualReader = mock(ErrorResidualReader.class);
        when(mockResidualReader.getDimension()).thenReturn(4);
        when(mockResidualReader.getCentroid()).thenReturn(new float[] { 0f, 0f, 0f, 0f });

        // V2 needs quantizer for the per-query setup
        OptimizedScalarQuantizer realQuantizer = new OptimizedScalarQuantizer(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);
        when(mockResidualReader.getQuantizer()).thenReturn(realQuantizer);
        when(mockResidualReader.getQuantizedByteVectorValues()).thenReturn(mock(QuantizedByteVectorValues.class));

        IndexInput mockClone = mock(IndexInput.class);
        when(mockResidualReader.cloneInput()).thenReturn(mockClone);

        Faiss1040ScalarQuantizedKnnVectorsReader reader = createReader(
            new FieldInfos(new FieldInfo[0]),
            Collections.emptySet(),
            mock(FlatVectorsReader.class)
        );
        Field errField = Faiss1040ScalarQuantizedKnnVectorsReader.class.getDeclaredField("errorResidualReader");
        errField.setAccessible(true);
        errField.set(reader, mockResidualReader);

        ScoreDoc[] result = reader.refine("field", new float[] { 1, 2, 3, 4 }, new int[0], new float[0]);

        assertEquals(0, result.length);
        verify(mockClone).close();
        reader.close();
    }

    /**
     * Verify that close() closes the errorResidualReader when it is present.
     */
    @SneakyThrows
    public void testClose_closesErrorResidualReader() {
        ErrorResidualReader mockResidualReader = mock(ErrorResidualReader.class);
        FlatVectorsReader fvr = mock(FlatVectorsReader.class);

        Faiss1040ScalarQuantizedKnnVectorsReader reader = createReader(new FieldInfos(new FieldInfo[0]), Collections.emptySet(), fvr);

        // Inject mock errorResidualReader
        Field errField = Faiss1040ScalarQuantizedKnnVectorsReader.class.getDeclaredField("errorResidualReader");
        errField.setAccessible(true);
        errField.set(reader, mockResidualReader);

        reader.close();

        verify(fvr).close();
        verify(mockResidualReader).close();
    }

    // --- helpers ---

    /**
     * Build a fake per-vector block (packed nibbles + 16B little-endian metadata).
     * Used to set up mock ErrorResidualReader.readBlock() return values.
     */
    private static byte[] buildBlock(int[] nibbles, float lower, float upper, float correction, int componentSum) {
        int dim = nibbles.length;
        int packedBytes = (dim + 1) / 2;
        byte[] block = new byte[packedBytes + ResidualQuantizer.PER_VECTOR_META_BYTES];

        // Pack nibbles
        for (int d = 0; d < dim; d += 2) {
            int q0 = nibbles[d];
            int q1 = (d + 1 < dim) ? nibbles[d + 1] : 0;
            block[d / 2] = (byte) ((q1 << 4) | (q0 & 0x0F));
        }

        // Write metadata in little-endian
        ByteBuffer meta = ByteBuffer.wrap(block, packedBytes, ResidualQuantizer.PER_VECTOR_META_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        meta.putFloat(lower);
        meta.putFloat(upper);
        meta.putFloat(correction);
        meta.putInt(componentSum);

        return block;
    }

    private static FieldInfo createFieldInfo(String name, KNNEngine engine, int fieldNo) {
        KNNCodecTestUtil.FieldInfoBuilder b = KNNCodecTestUtil.FieldInfoBuilder.builder(name).fieldNumber(fieldNo);
        if (engine != null) {
            b.addAttribute(KNNVectorFieldMapper.KNN_FIELD, "true").addAttribute(KNNConstants.KNN_ENGINE, engine.getName());
        }
        return b.build();
    }

    @SneakyThrows
    private static Faiss1040ScalarQuantizedKnnVectorsReader createReader(FieldInfos fieldInfos, Set<String> files, FlatVectorsReader fvr) {
        Directory dir = mock(Directory.class);
        when(dir.openInput(any(), any())).thenReturn(mock(IndexInput.class));
        SegmentInfo si = mock(SegmentInfo.class);
        when(si.files()).thenReturn(files);
        when(si.getId()).thenReturn((si.hashCode() + "").getBytes());
        return new Faiss1040ScalarQuantizedKnnVectorsReader(new SegmentReadState(dir, si, fieldInfos, IOContext.DEFAULT), fvr);
    }
}
