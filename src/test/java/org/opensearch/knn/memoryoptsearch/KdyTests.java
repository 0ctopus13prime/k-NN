/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.backward_codecs.lucene101.Lucene101Codec;
import org.apache.lucene.backward_codecs.lucene101.Lucene101PostingsFormat;
import org.apache.lucene.backward_codecs.lucene912.Lucene912PostingsFormat;
import org.apache.lucene.codecs.FieldsConsumer;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.PostingsWriterBase;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.lucene102.Lucene102BinaryQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene102.Lucene102BinaryQuantizedVectorsWriter;
import org.apache.lucene.codecs.lucene102.Lucene102HnswBinaryQuantizedVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Supplier;

import static org.mockito.Mockito.mock;
import static org.opensearch.knn.DerivedSourceUtils.randomVectorSupplier;
import static org.opensearch.knn.common.KNNConstants.QFRAMEWORK_CONFIG;

public class KdyTests extends KNNTestCase {
    @SneakyThrows
    public void testXXX() {
        final Lucene102BinaryQuantizedVectorsFormat format = new Lucene102BinaryQuantizedVectorsFormat();

        // Params
        final int maxDoc = 1000;
        final int dimension = 128;

        // Directory
        final String dirPath = "/Users/kdooyong/tmp/kdy";
        Path dir = Paths.get(dirPath);
        final MMapDirectory directory = new MMapDirectory(dir);
        if (Files.exists(dir) && Files.isDirectory(dir)) {
            Files.walk(dir).sorted(Comparator.reverseOrder()).filter(path -> !path.equals(dir)).forEach(path -> {
                try {
                    Files.delete(path);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            });
        }

        // Make segment info
        final SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            "_0",
            maxDoc,
            false,
            false,
            null,
            Collections.emptyMap(),
            new byte[StringHelper.ID_LENGTH],
            Collections.emptyMap(),
            null
        );

        // Create write state
        final FieldInfos fieldInfos = mock(FieldInfos.class);
        final InfoStream infoStream = mock(InfoStream.class);
        final SegmentWriteState writeState = new SegmentWriteState(infoStream, directory, segmentInfo, fieldInfos, null, IOContext.DEFAULT);

        final Lucene102BinaryQuantizedVectorsWriter writer = (Lucene102BinaryQuantizedVectorsWriter) format.fieldsWriter(writeState);

        // Add field
        KNNCodecTestUtil.FieldInfoBuilder fieldInfoBuilder = KNNCodecTestUtil.FieldInfoBuilder.builder("target_field")
            .addAttribute(KNNVectorFieldMapper.KNN_FIELD, "true")
            .addAttribute(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
            .vectorDimension(dimension);

        // Add space type from build parameters
        fieldInfoBuilder.addAttribute(KNNConstants.SPACE_TYPE, "");
        fieldInfoBuilder.addAttribute(QFRAMEWORK_CONFIG, "");
        final FieldInfo vectorField = fieldInfoBuilder.build();

        // Start indexing
        final FlatFieldVectorsWriter flatWriter = writer.addField(vectorField);
        final Random random = new Random();
        final Supplier<Object> supplier = randomVectorSupplier(random, dimension, VectorDataType.FLOAT);
        for (int i = 0; i < maxDoc; ++i) {
            final Object vector = supplier.get();
            flatWriter.addValue(i, vector);
        }

        writer.flush(maxDoc, null);
        writer.finish();
        writer.close();
    }

    @SneakyThrows
    public void testYYY() {
        final Lucene102BinaryQuantizedVectorsFormat format = new Lucene102BinaryQuantizedVectorsFormat();

        // Params
        final int maxDoc = 1000;
        final int dimension = 128;

        // Directory
        final String dirPath = "/Users/kdooyong/tmp/kdy";
        Path dir = Paths.get(dirPath);
        final MMapDirectory directory = new MMapDirectory(dir);

        // Make segment info
        final SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            "_0",
            maxDoc,
            false,
            false,
            null,
            Collections.emptyMap(),
            new byte[StringHelper.ID_LENGTH],
            Collections.emptyMap(),
            null
        );

        // Add field
        KNNCodecTestUtil.FieldInfoBuilder fieldInfoBuilder = KNNCodecTestUtil.FieldInfoBuilder.builder("target_field")
            .addAttribute(KNNVectorFieldMapper.KNN_FIELD, "true")
            .addAttribute(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
            .vectorDimension(dimension);

        // Add space type from build parameters
        fieldInfoBuilder.addAttribute(KNNConstants.SPACE_TYPE, "");
        fieldInfoBuilder.addAttribute(QFRAMEWORK_CONFIG, "");
        final FieldInfo vectorField = fieldInfoBuilder.build();
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { vectorField });

        // Create read state
        final SegmentReadState readState = new SegmentReadState(directory, segmentInfo, fieldInfos, IOContext.DEFAULT, "");

        // Get reader
        final FlatVectorsReader reader = format.fieldsReader(readState);
        final Random random = new Random();
        final byte[] binaryVector = new byte[(4 * dimension) / 8];
        random.nextBytes(binaryVector);
        final RandomVectorScorer scorer = reader.getRandomVectorScorer("target_field", binaryVector);
        for (int i = 0; i < maxDoc; ++i) {
            System.out.println("i -> " + scorer.score(i));
        }
    }

    @SneakyThrows
    public void testQQQ() {
        int numVectors = 300;
        int dimensions = 128;
        String fieldName = "target_field";

        // 1. Setup Configuration with our Custom BBQ Codec
        Directory dir = new ByteBuffersDirectory();
        IndexWriterConfig iwc = new IndexWriterConfig();
        iwc.setCodec(new BBQCodec());

        // 2. Indexing 300 Vectors
        try (IndexWriter writer = new IndexWriter(dir, iwc)) {
            Random random = new Random(42);
            for (int i = 0; i < numVectors; i++) {
                float[] vector = new float[dimensions];
                for (int j = 0; j < dimensions; j++) {
                    vector[j] = random.nextFloat() * 2 - 1; // Random values between -1 and 1
                }

                Document doc = new Document();
                // Lucene handles the 32x BBQ quantization internally
                doc.add(new KnnFloatVectorField(fieldName, vector, VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT));
                writer.addDocument(doc);
            }
            writer.commit();
        }

        // 3. Searching the BBQ Index
        try (IndexReader reader = DirectoryReader.open(dir)) {
            IndexSearcher searcher = new IndexSearcher(reader);

            // Create a random query vector
            float[] queryVector = new float[dimensions];
            for (int i = 0; i < dimensions; ++i) {
                queryVector[i] = Math.min(Math.max(ThreadLocalRandom.current().nextFloat(), -5), 5);
            }

            // KNN Search
            KnnFloatVectorQuery query = new KnnFloatVectorQuery(fieldName, queryVector, 5);
            TopDocs topDocs = searcher.search(query, 5);

            System.out.println("Top 5 BBQ results:");
            for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                Document d = searcher.getIndexReader().storedFields().document(scoreDoc.doc);
                System.out.println("ID: " + d.get("id") + " | Score: " + scoreDoc.score);
            }
        }
    }

    public class BBQCodec extends Lucene101Codec {
        private final KnnVectorsFormat bbqFormat = new Lucene102HnswBinaryQuantizedVectorsFormat();

        @Override
        public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
            if ("target_field".equals(field)) {
                return bbqFormat;
            }
            return super.getKnnVectorsFormatForField(field);
        }
    }
}
