/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.bbq;

import lombok.extern.log4j.Log4j2;
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
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.nativeindex.bbq.BBQReader;
import org.opensearch.knn.index.codec.nativeindex.bbq.BBQWriter;
import org.opensearch.knn.index.codec.nativeindex.bbq.BinarizedByteVectorValues;
import org.opensearch.knn.index.codec.nativeindex.bbq.Lucene102BinaryFlatVectorsScorer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.jni.FaissService;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Program 2: Faiss BBQDistanceComputer validation.
 * Ingests vectors through BBQWriter -> BBQReader -> FaissBBQFlat via JNI,
 * then calls bbqValidationScan to do symmetric scoring on the C++ side.
 */
@Log4j2
public class FaissBBQIndexBuildTimingTests extends KNNTestCase {

    // ===== CONFIGURE THESE =====
    private static final String DATA_PATH = "/Users/kdooyong/workspace/io-opt/tmp/vectors.bin";
    private static final int TOP_K = 100;
    private static final int QUERY_VECTOR_ORDINAL = 0;
    // ===========================

    public void testFaissBBQRecall() throws Exception {
        float[][] vectors = VectorDataLoader.loadVectors(DATA_PATH);
        int dimension = vectors[0].length;
        int numVectors = vectors.length;

        System.out.println("Loaded " + numVectors + " vectors with dimension " + dimension);

        Path tmpDir = Files.createTempDirectory("faiss_bbq_validation");
        byte[] segmentId = StringHelper.randomId();
        String segmentName = "test";
        try (Directory directory = new MMapDirectory(tmpDir)) {
            final long indexStart = System.nanoTime();

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

            // Step 1: Write vectors through BBQWriter to produce .veb files
            writeVectors(directory, vectors, segmentInfo, fieldInfo);

            // Step 2: Read back via BBQReader and ingest into FaissBBQFlat via JNI
            final long indexMemoryAddress = ingestIntoFaiss(directory, dimension, numVectors, segmentInfo, fieldInfo);

            // Step 3: Write HNSW index to file
            String faissIndexFileName = "faiss-bbq-validation.hnsw";
            try (var indexOutput = directory.createOutput(faissIndexFileName, IOContext.DEFAULT)) {
                IndexOutputWithBuffer outputWithBuffer = new IndexOutputWithBuffer(indexOutput);
                Map<String, Object> writeParams = new HashMap<>();
                writeParams.put("name", "hnsw");
                writeParams.put("data_type", "float");
                writeParams.put("index_description", "BHNSW16,Flat");
                writeParams.put("spaceType", "innerproduct");
                Map<String, Object> subParams = new HashMap<>();
                writeParams.put("parameters", subParams);
                subParams.put("ef_search", 256);
                subParams.put("ef_construction", 256);
                subParams.put("m", 16);
                subParams.put("encoder", Collections.emptyMap());
                subParams.put("indexThreadQty", 1);
                FaissService.writeBBQIndex(outputWithBuffer, indexMemoryAddress, writeParams);
            }

            final long indexingDone = System.nanoTime();
            System.out.println("[INDEXING TIME] " + (indexingDone - indexStart) / 1_000_000 + "ms");
        }
    }

    private void writeVectors(Directory directory, float[][] vectors, SegmentInfo segmentInfo, FieldInfo fieldInfo) throws IOException {
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
        SegmentWriteState writeState =
            new SegmentWriteState(InfoStream.NO_OUTPUT, directory, segmentInfo, fieldInfos, null, IOContext.DEFAULT, "test_field");

        Lucene102BinaryFlatVectorsScorer scorer = new Lucene102BinaryFlatVectorsScorer();
        try (BBQWriter writer = new BBQWriter(scorer, writeState)) {
            FlatFieldVectorsWriter fieldWriter = writer.addField(fieldInfo);
            for (int i = 0; i < vectors.length; i++) {
                fieldWriter.addValue(i, vectors[i].clone());
            }
            writer.flush(vectors.length, null);
            writer.finish();
        }
    }

    private long ingestIntoFaiss(Directory directory, int dimension, int numVectors, SegmentInfo segmentInfo, FieldInfo fieldInfo)
        throws IOException {
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
        SegmentReadState readState = new SegmentReadState(directory, segmentInfo, fieldInfos, IOContext.DEFAULT, "test_field");

        Lucene102BinaryFlatVectorsScorer scorer = new Lucene102BinaryFlatVectorsScorer();
        try (BBQReader reader = new BBQReader(readState, scorer)) {
            FloatVectorValues floatVectorValues = reader.getFloatVectorValues("test_field");
            BBQReader.BinarizedVectorValues binarizedVectorValues = (BBQReader.BinarizedVectorValues) floatVectorValues;
            BinarizedByteVectorValues quantizedVectorValues = binarizedVectorValues.quantizedVectorValues;

            int quantizedVecBytes = quantizedVectorValues.vectorValue(0).length;
            float centroidDp = quantizedVectorValues.getCentroidDP();

            // Initialize Faiss BBQ index
            Map<String, Object> parameters = new HashMap<>();
            parameters.put("name", "hnsw");
            parameters.put("data_type", "float");
            parameters.put("index_description", "BHNSW16,Flat");
            parameters.put("spaceType", "innerproduct");
            Map<String, Object> subParameters = new HashMap<>();
            parameters.put("parameters", subParameters);
            subParameters.put("ef_search", 256);
            subParameters.put("ef_construction", 256);
            subParameters.put("m", 16);
            subParameters.put("encoder", Collections.emptyMap());
            subParameters.put("indexThreadQty", 1);

            long indexMemoryAddress = FaissService.initBBQIndex(numVectors, dimension, parameters, centroidDp, quantizedVecBytes);

            // Pass quantized vectors + correction factors in batches
            passQuantizedVectors(indexMemoryAddress, binarizedVectorValues);

            // Add doc IDs
            int batchSize = 1024;
            int[] docIds = new int[batchSize];
            int numAdded = 0;
            int remaining = numVectors;
            while (remaining > 0) {
                int count = Math.min(batchSize, remaining);
                for (int i = 0; i < count; i++) {
                    docIds[i] = numAdded + i;
                }
                FaissService.addDocsToBBQIndex(indexMemoryAddress, docIds, count, numAdded);
                numAdded += count;
                remaining -= count;
            }

            return indexMemoryAddress;
        }
    }

    private void passQuantizedVectors(final long indexMemoryAddress, final BBQReader.BinarizedVectorValues binarizedVectorValues)
        throws IOException {
        final int batchSize = Math.max(1, (int) (binarizedVectorValues.size() * 0.01));
        byte[] buffer = null;
        for (int i = 0; i < binarizedVectorValues.size(); ) {
            final int loopSize = Math.min(binarizedVectorValues.size() - i, batchSize);
            for (int j = 0, o = 0; j < loopSize; ++j) {
                final byte[] binaryVector = binarizedVectorValues.quantizedVectorValues.vectorValue(i + j);
                if (buffer == null) {
                    // [Quantized Vector | lowerInterval (float) | upperInterval (float) | additionalCorrection (float) |
                    // quantizedComponentSum (int)]
                    buffer = new byte[(binaryVector.length + Integer.BYTES * 4) * batchSize];
                }
                final OptimizedScalarQuantizer.QuantizationResult quantizationResult =
                    binarizedVectorValues.quantizedVectorValues.getCorrectiveTerms(i + j);

                // Copy quantized vector
                System.arraycopy(binaryVector, 0, buffer, o, binaryVector.length);
                o += binaryVector.length;

                // Copy correction factors
                int bits = Float.floatToRawIntBits(quantizationResult.lowerInterval());
                buffer[o++] = (byte) (bits);
                buffer[o++] = (byte) (bits >>> 8);
                buffer[o++] = (byte) (bits >>> 16);
                buffer[o++] = (byte) (bits >>> 24);

                bits = Float.floatToRawIntBits(quantizationResult.upperInterval());
                buffer[o++] = (byte) (bits);
                buffer[o++] = (byte) (bits >>> 8);
                buffer[o++] = (byte) (bits >>> 16);
                buffer[o++] = (byte) (bits >>> 24);

                bits = Float.floatToRawIntBits(quantizationResult.additionalCorrection());
                buffer[o++] = (byte) (bits);
                buffer[o++] = (byte) (bits >>> 8);
                buffer[o++] = (byte) (bits >>> 16);
                buffer[o++] = (byte) (bits >>> 24);

                bits = quantizationResult.quantizedComponentSum();
                buffer[o++] = (byte) (bits);
                buffer[o++] = (byte) (bits >>> 8);
                buffer[o++] = (byte) (bits >>> 16);
                buffer[o++] = (byte) (bits >>> 24);
            }

            FaissService.passBBQVectors(indexMemoryAddress, buffer, loopSize);

            i += loopSize;
        }
    }
}
