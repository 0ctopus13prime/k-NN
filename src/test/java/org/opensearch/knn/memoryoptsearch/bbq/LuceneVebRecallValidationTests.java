/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.bbq;

import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.index.codec.nativeindex.bbq.BBQReader;
import org.opensearch.knn.index.codec.nativeindex.bbq.BBQWriter;
import org.opensearch.knn.index.codec.nativeindex.bbq.BinarizedByteVectorValues;
import org.opensearch.knn.index.codec.nativeindex.bbq.Lucene102BinaryFlatVectorsScorer;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.lucene.util.quantization.OptimizedScalarQuantizer.discretize;

/**
 * Program 1: Lucene .veb path validation.
 * Writes vectors through BBQWriter, reads back via BBQReader,
 * then does symmetric bit-vector-to-bit-vector full scan for top-k.
 */
public class LuceneVebRecallValidationTests extends OpenSearchTestCase {

    // ===== CONFIGURE THESE =====
    private static final String DATA_PATH = "/Users/kdooyong/workspace/io-opt/tmp/vectors.bin";
    private static final int TOP_K = 100;
    private static final int QUERY_VECTOR_ORDINAL = 0;
    // ===========================

    public void testLuceneVebRecall() throws Exception {
        float[][] vectors = VectorDataLoader.loadVectors(DATA_PATH);
        int dimension = vectors[0].length;
        int numVectors = vectors.length;

        System.out.println("Loaded " + numVectors + " vectors with dimension " + dimension);

        // Create temp directory for Lucene files
        Path tmpDir = Files.createTempDirectory("lucene_veb_validation");
        byte[] segmentId = StringHelper.randomId();

        final FieldInfo fieldInfo = new FieldInfo(
            "test_field",
            0,
            false,
            false,
            false,
            IndexOptions.NONE,
            org.apache.lucene.index.DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,
            new HashMap<>(),
            0,
            0,
            0,
            dimension,
            VectorEncoding.FLOAT32,
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
            false,
            false
        );

        try (Directory directory = new MMapDirectory(tmpDir)) {
            // Write vectors through BBQWriter
            writeVectors(directory, vectors, dimension, segmentId, fieldInfo);

            // Read back and do symmetric scoring
            List<Map.Entry<Integer, Float>> topK =
                readAndScore(directory, dimension, numVectors, segmentId, fieldInfo);

            // Print results
            System.out.println("=== Lucene .veb Symmetric Scoring Top-" + TOP_K + " (query ordinal=" + QUERY_VECTOR_ORDINAL + ") ===");
            for (Map.Entry<Integer, Float> entry : topK) {
                System.out.println(entry.getKey() + "\t" + entry.getValue());
            }
        }
    }

    private void writeVectors(Directory directory, float[][] vectors, int dimension, byte[] segmentId, FieldInfo fieldInfo) throws IOException {
        String segmentName = "test";
        String segmentSuffix = "";

        SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            segmentName,
            vectors.length,
            false,
            false,
            null,
            new HashMap<>(),
            segmentId,
            new HashMap<>(),
            null
        );


        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
        SegmentWriteState writeState = new SegmentWriteState(
            InfoStream.NO_OUTPUT,
            directory,
            segmentInfo,
            fieldInfos,
            null,
            IOContext.DEFAULT
        );

        Lucene102BinaryFlatVectorsScorer scorer = new Lucene102BinaryFlatVectorsScorer();
        try (BBQWriter writer = new BBQWriter(scorer, writeState)) {
            FlatFieldVectorsWriter fieldWriter = writer.addField(fieldInfo);
            for (int i = 0; i < vectors.length; i++) {
                fieldWriter.addValue(i, vectors[i]);
            }
            writer.flush(vectors.length, null);
            writer.finish();
        }
    }

    private List<Map.Entry<Integer, Float>> readAndScore(Directory directory, int dimension, int numVectors, byte[] segmentId,
        FieldInfo fieldInfo
    ) throws IOException {
        String segmentName = "test";
        String segmentSuffix = "";

        SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            segmentName,
            numVectors,
            false,
            false,
            null,
            new HashMap<>(),
            segmentId,
            new HashMap<>(),
            null
        );

        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
        SegmentReadState readState = new SegmentReadState(
            directory,
            segmentInfo,
            fieldInfos,
            IOContext.DEFAULT
        );

        Lucene102BinaryFlatVectorsScorer scorer = new Lucene102BinaryFlatVectorsScorer();
        try (BBQReader reader = new BBQReader(readState, scorer)) {
            return symmetricBitVectorScan(reader, dimension, numVectors);
        }
    }

    /**
     * Symmetric bit-vector-to-bit-vector scoring.
     * Mirrors BBQDistanceComputer::symmetric_dis from the Faiss C++ side.
     *
     * For each pair (queryOrd, targetOrd):
     *   dp = popcount(queryBinary & targetBinary)  -- operating on long[] chunks
     *   score = ax * az * dim + az * lx * x1 + ax * lz * z1 + lx * lz * dp + additionalX + additionalZ - centroidDp
     */
    private List<Map.Entry<Integer, Float>> symmetricBitVectorScan(BBQReader reader, int dimension, int numVectors) throws IOException {
        FloatVectorValues floatVectorValues = reader.getFloatVectorValues("test_field");
        BBQReader.BinarizedVectorValues binarizedValues = (BBQReader.BinarizedVectorValues) floatVectorValues;
        BinarizedByteVectorValues quantizedValues = binarizedValues.quantizedVectorValues;

        float centroidDp = quantizedValues.getCentroidDP();
        int discretizedDims = discretize(dimension, 64);
        int numBytes = discretizedDims / 8;

        // Read query vector binary code + correction factors
        byte[] queryBinary = quantizedValues.vectorValue(QUERY_VECTOR_ORDINAL).clone();
        OptimizedScalarQuantizer.QuantizationResult queryCorrections = quantizedValues.getCorrectiveTerms(QUERY_VECTOR_ORDINAL);
        float ax = queryCorrections.lowerInterval();
        float lx = queryCorrections.upperInterval() - ax;
        float additionalX = queryCorrections.additionalCorrection();
        float x1 = queryCorrections.quantizedComponentSum();

        // Convert query binary to long[] for efficient popcount
        long[] queryLongs = bytesToLongs(queryBinary, numBytes);

        // Full scan
        List<Map.Entry<Integer, Float>> results = new ArrayList<>();
        for (int i = 0; i < numVectors; i++) {
            if (i == QUERY_VECTOR_ORDINAL) {
                continue; // skip self
            }

            // TMP
            if (i == 385) {
                System.out.println();
            }
            // TMP

            byte[] targetBinary = quantizedValues.vectorValue(i).clone();
            var targetCorrections = quantizedValues.getCorrectiveTerms(i);
            float az = targetCorrections.lowerInterval();
            float lz = targetCorrections.upperInterval() - az;
            float additionalZ = targetCorrections.additionalCorrection();
            float z1 = targetCorrections.quantizedComponentSum();

            // Convert target binary to long[] for popcount
            long[] targetLongs = bytesToLongs(targetBinary, numBytes);

            // Compute bit dot product: popcount(query & target)
            int dp = 0;
            for (int w = 0; w < queryLongs.length; w++) {
                dp += Long.bitCount(queryLongs[w] & targetLongs[w]);
            }

            // Symmetric scoring (same as BBQDistanceComputer::symmetric_dis)
            float score = ax * az * dimension
                + az * lx * x1
                + ax * lz * z1
                + lx * lz * dp
                + additionalX
                + additionalZ
                - centroidDp;

            results.add(new AbstractMap.SimpleEntry<>(i, score));
        }

        // Sort by score descending (inner product: higher is better)
        results.sort(Comparator.<Map.Entry<Integer, Float>, Float>comparing(Map.Entry::getValue).reversed());

        // Return top-k
        return results.subList(0, Math.min(TOP_K, results.size()));
    }

    /**
     * Convert byte[] to long[] for efficient bitwise operations.
     * Reads in little-endian order to match the C++ side.
     */
    private static long[] bytesToLongs(byte[] bytes, int numBytes) {
        int numLongs = numBytes / 8;
        long[] longs = new long[numLongs];
        ByteBuffer buffer = ByteBuffer.wrap(bytes, 0, numBytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < numLongs; i++) {
            longs[i] = buffer.getLong();
        }
        return longs;
    }
}
