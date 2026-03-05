/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.bbq;

import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
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
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.nativeindex.bbq.BBQReader;
import org.opensearch.knn.index.codec.nativeindex.bbq.BBQWriter;
import org.opensearch.knn.index.codec.nativeindex.bbq.Lucene102BinaryFlatVectorsScorer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Random;

/**
 * Validates that the native C++ BBQ scoring produces the same results as the Java baseline.
 * Compares score() and bulkScore() between native and Java implementations.
 */
public class NativeBBQScoringValidationTests extends KNNTestCase {

    private static final String DATA_PATH = "/Users/kdooyong/workspace/io-opt/tmp/vectors.bin";
    private static final int NUM_ITERATIONS = 100;
    private static final int NUM_SAMPLE_VECTORS = 20;
    private static final double EPSILON = 1e-3;

    public void testNativeVsJavaBBQScoring() throws Exception {
        float[][] vectors = VectorDataLoader.loadVectors(DATA_PATH);
        int dimension = vectors[0].length;
        int numVectors = vectors.length;
        System.out.println("Loaded " + numVectors + " vectors with dimension " + dimension);

        Path tmpDir = Files.createTempDirectory("native_bbq_validation");
        byte[] segmentId = StringHelper.randomId();
        String segmentName = "test";

        try (Directory directory = new MMapDirectory(tmpDir)) {
            SegmentInfo segmentInfo = new SegmentInfo(
                directory, Version.LATEST, Version.LATEST, segmentName, numVectors,
                false, false, null, new HashMap<>(), segmentId, new HashMap<>(), null
            );

            FieldInfo fieldInfo = new FieldInfo(
                "test_field", 0, false, false, false,
                IndexOptions.NONE, org.apache.lucene.index.DocValuesType.NONE,
                DocValuesSkipIndexType.NONE, -1, new HashMap<>(), 0, 0, 0,
                dimension, VectorEncoding.FLOAT32,
                VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, false, false
            );

            // Write vectors through BBQWriter
            writeVectors(directory, vectors, segmentInfo, fieldInfo);

            FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
            Random rng = new Random(42);

            for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
                // Pick a random query vector
                int queryOrd = rng.nextInt(numVectors);
                float[] queryVector = vectors[queryOrd];

                // Pick random sample ordinals
                int[] sampleOrdinals = new int[NUM_SAMPLE_VECTORS];
                for (int i = 0; i < NUM_SAMPLE_VECTORS; i++) {
                    sampleOrdinals[i] = rng.nextInt(numVectors);
                    if (sampleOrdinals[i] == queryOrd) {
                        --i;
                    }
                }

                // Get Java baseline scorer (USE_NATIVE = false)
                float[] javaScores = new float[NUM_SAMPLE_VECTORS];
                float[] javaBulkScores = new float[NUM_SAMPLE_VECTORS];
                {
                    Lucene102BinaryFlatVectorsScorer.USE_NATIVE = false;
                    SegmentReadState readState = new SegmentReadState(
                        directory, segmentInfo, fieldInfos, IOContext.DEFAULT, "test_field"
                    );
                    try (BBQReader reader = new BBQReader(readState, new Lucene102BinaryFlatVectorsScorer())) {
                        RandomVectorScorer javaScorer = reader.getRandomVectorScorer("test_field", queryVector);
                        for (int i = 0; i < NUM_SAMPLE_VECTORS; i++) {
                            javaScores[i] = javaScorer.score(sampleOrdinals[i]);
                        }
                        javaScorer.bulkScore(sampleOrdinals, javaBulkScores, NUM_SAMPLE_VECTORS);
                    }
                }

                // Get native C++ scorer (USE_NATIVE = true)
                float[] nativeScores = new float[NUM_SAMPLE_VECTORS];
                float[] nativeBulkScores = new float[NUM_SAMPLE_VECTORS];
                {
                    Lucene102BinaryFlatVectorsScorer.USE_NATIVE = true;
                    SegmentReadState readState = new SegmentReadState(
                        directory, segmentInfo, fieldInfos, IOContext.DEFAULT, "test_field"
                    );
                    try (BBQReader reader = new BBQReader(readState, new Lucene102BinaryFlatVectorsScorer())) {
                        RandomVectorScorer nativeScorer = reader.getRandomVectorScorer("test_field", queryVector);
                        for (int i = 0; i < NUM_SAMPLE_VECTORS; i++) {
                            nativeScores[i] = nativeScorer.score(sampleOrdinals[i]);
                        }
                        nativeScorer.bulkScore(sampleOrdinals, nativeBulkScores, NUM_SAMPLE_VECTORS);
                    }
                }

                // Compare score() results
                for (int i = 0; i < NUM_SAMPLE_VECTORS; i++) {
                    assertEquals(
                        "Iteration " + iter + ", score() mismatch at ordinal " + sampleOrdinals[i]
                            + " (queryOrd=" + queryOrd + ")",
                        javaScores[i], nativeScores[i], EPSILON
                    );
                }

                // Compare bulkScore() results
                for (int i = 0; i < NUM_SAMPLE_VECTORS; i++) {
                    assertEquals(
                        "Iteration " + iter + ", bulkScore() mismatch at ordinal " + sampleOrdinals[i]
                            + " (queryOrd=" + queryOrd + ")",
                        javaBulkScores[i], nativeBulkScores[i], EPSILON
                    );
                }

                // Also verify Java score() == Java bulkScore() (sanity check)
                for (int i = 0; i < NUM_SAMPLE_VECTORS; i++) {
                    assertEquals(
                        "Iteration " + iter + ", Java score() vs bulkScore() mismatch at ordinal " + sampleOrdinals[i],
                        javaScores[i], javaBulkScores[i], EPSILON
                    );
                }

                // Also verify native score() == native bulkScore()
                for (int i = 0; i < NUM_SAMPLE_VECTORS; i++) {
                    assertEquals(
                        "Iteration " + iter + ", native score() vs bulkScore() mismatch at ordinal " + sampleOrdinals[i],
                        nativeScores[i], nativeBulkScores[i], EPSILON
                    );
                }

                if (iter % 10 == 0) {
                    System.out.println("Iteration " + iter + " passed (queryOrd=" + queryOrd + ")");
                }
            }

            System.out.println("All " + NUM_ITERATIONS + " iterations passed.");
        } finally {
            Lucene102BinaryFlatVectorsScorer.USE_NATIVE = true;
        }
    }

    private void writeVectors(Directory directory, float[][] vectors, SegmentInfo segmentInfo, FieldInfo fieldInfo) throws IOException {
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
        SegmentWriteState writeState = new SegmentWriteState(
            InfoStream.NO_OUTPUT, directory, segmentInfo, fieldInfos, null, IOContext.DEFAULT, "test_field"
        );

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
}
