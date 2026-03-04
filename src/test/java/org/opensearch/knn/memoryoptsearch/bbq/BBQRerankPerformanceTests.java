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
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
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
import org.opensearch.knn.index.codec.nativeindex.bbq.OffHeapBinarizedVectorValues;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.jni.FaissService;
import org.opensearch.knn.memoryoptsearch.faiss.FaissMemoryOptimizedSearcher;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class BBQRerankPerformanceTests extends KNNTestCase {

    // ===== CONFIGURE THESE =====
    private static final String DATA_PATH = "/Users/kdooyong/workspace/io-opt/tmp/vectors.bin";
    private static final int TOP_K = 100;
    private static final int QUERY_VECTOR_ORDINAL = 0;
    private static final int OVERSAMPLE_FACTOR = 2;
    private static final int WARMUP_ITERATIONS = 1000;
    private static final int BENCH_ITERATIONS = 1000;
    // ===========================

    public void testRerankPerformance() throws Exception {
        float[][] vectors = VectorDataLoader.loadVectors(DATA_PATH);
        int dimension = vectors[0].length;
        int numVectors = vectors.length;
        System.out.println("Loaded " + numVectors + " vectors with dimension " + dimension);

        Path tmpDir = Files.createTempDirectory("bbq_rerank_perf");
        byte[] segmentId = StringHelper.randomId();
        String segmentName = "test";

        try (Directory directory = new MMapDirectory(tmpDir)) {
            SegmentInfo segmentInfo = new SegmentInfo(
                directory, Version.LATEST, Version.LATEST, segmentName, numVectors,
                false, false, null, new HashMap<>(), segmentId, new HashMap<>(), null
            );

            final FieldInfo fieldInfo = new FieldInfo(
                "test_field", 0, false, false, false,
                IndexOptions.NONE, org.apache.lucene.index.DocValuesType.NONE,
                DocValuesSkipIndexType.NONE, -1, new HashMap<>(), 0, 0, 0,
                dimension, VectorEncoding.FLOAT32,
                VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, false, false
            );

            // Step 1: Write BBQ files
            writeVectors(directory, vectors, segmentInfo, fieldInfo);

            // Step 2: Ingest into Faiss
            long indexMemoryAddress = ingestIntoFaiss(directory, dimension, numVectors, segmentInfo, fieldInfo);

            // Step 3: Write HNSW index
            String faissIndexFileName = "bbq-rerank-perf.hnsw";
            try (var indexOutput = directory.createOutput(faissIndexFileName, IOContext.DEFAULT)) {
                IndexOutputWithBuffer outputWithBuffer = new IndexOutputWithBuffer(indexOutput);
                FaissService.writeBBQIndex(outputWithBuffer, indexMemoryAddress, buildFaissParams());
            }

            // Step 4: ANN search + Step 5: Benchmark BBQ rerank (searcher must stay open)
            FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
            SegmentReadState readState = new SegmentReadState(directory, segmentInfo, fieldInfos, IOContext.DEFAULT);
            float[] queryVector = vectors[QUERY_VECTOR_ORDINAL];

            try (var indexInput = directory.openInput(faissIndexFileName, IOContext.DEFAULT)) {
                try (FaissMemoryOptimizedSearcher searcher = new FaissMemoryOptimizedSearcher(indexInput, fieldInfo, readState)) {
                    TopKnnCollector collector = new TopKnnCollector(TOP_K * OVERSAMPLE_FACTOR, Integer.MAX_VALUE);
                    searcher.search(queryVector, collector, AcceptDocs.fromLiveDocs(null, numVectors));
                    TopDocs annResults = collector.topDocs();
                    System.out.println("ANN search returned " + annResults.scoreDocs.length + " candidates\n");

                    // Benchmark BBQ rerank
                    benchmarkBBQRerank(searcher, annResults, queryVector);

                    // Benchmark full precision rerank
                    benchmarkFullPrecisionRerank(annResults, queryVector, vectors);
                }
            }
        }

        System.out.println();
    }

    private void benchmarkBBQRerank(FaissMemoryOptimizedSearcher searcher, TopDocs annResults, float[] queryVector) throws IOException {
        System.out.println("=== BBQ Rerank Benchmark ===");

        // Build doc-to-score map and sorted docId array from ANN results
        Map<Integer, Float> docIdToScore = new HashMap<>();
        int[] sortedDocIds = new int[annResults.scoreDocs.length];
        for (int i = 0; i < annResults.scoreDocs.length; i++) {
            ScoreDoc sd = annResults.scoreDocs[i];
            docIdToScore.put(sd.doc, sd.score);
            sortedDocIds[i] = sd.doc;
        }
        Arrays.sort(sortedDocIds);

        // Warmup
        for (int w = 0; w < WARMUP_ITERATIONS; w++) {
            TopKnnCollector warmupCollector = new TopKnnCollector(TOP_K, Integer.MAX_VALUE);
            DocIdSetIterator disi = new TopDocsDISI(sortedDocIds);
            searcher.rerank(queryVector, disi, docIdToScore, warmupCollector);
            warmupCollector.topDocs();
        }

        // Benchmark
        long[] latencies = new long[BENCH_ITERATIONS];
        for (int iter = 0; iter < BENCH_ITERATIONS; iter++) {
            TopKnnCollector collector = new TopKnnCollector(TOP_K, Integer.MAX_VALUE);
            DocIdSetIterator disi = new TopDocsDISI(sortedDocIds);
            long start = System.nanoTime();
            searcher.rerank(queryVector, disi, docIdToScore, collector);
            collector.topDocs();
            latencies[iter] = System.nanoTime() - start;
        }

        printPercentiles("BBQ Rerank", latencies, annResults.scoreDocs.length);
    }

    /** Simple DocIdSetIterator over a sorted array of doc IDs. */
    private static class TopDocsDISI extends org.apache.lucene.search.DocIdSetIterator {
        private final int[] docIds;
        private int idx = -1;

        TopDocsDISI(int[] sortedDocIds) {
            this.docIds = sortedDocIds;
        }

        @Override
        public int docID() {
            return idx < 0 ? -1 : (idx >= docIds.length ? NO_MORE_DOCS : docIds[idx]);
        }

        @Override
        public int nextDoc() {
            idx++;
            return docID();
        }

        @Override
        public int advance(int target) {
            while (nextDoc() < target) {}
            return docID();
        }

        @Override
        public long cost() {
            return docIds.length;
        }
    }

    private void benchmarkFullPrecisionRerank(TopDocs annResults, float[] queryVector, float[][] vectors) {
        System.out.println("\n=== Full Precision Rerank Benchmark ===");

        int[] docIds = new int[annResults.scoreDocs.length];
        for (int i = 0; i < annResults.scoreDocs.length; i++) {
            docIds[i] = annResults.scoreDocs[i].doc;
        }

        // Warmup
        for (int w = 0; w < WARMUP_ITERATIONS; w++) {
            TopKnnCollector warmupCollector = new TopKnnCollector(TOP_K, Integer.MAX_VALUE);
            for (int docId : docIds) {
                float score = VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT.compare(queryVector, vectors[docId]);
                warmupCollector.collect(docId, score);
            }
            warmupCollector.topDocs();
        }

        // Benchmark
        long[] latencies = new long[BENCH_ITERATIONS];
        for (int iter = 0; iter < BENCH_ITERATIONS; iter++) {
            TopKnnCollector collector = new TopKnnCollector(TOP_K, Integer.MAX_VALUE);
            long start = System.nanoTime();
            for (int docId : docIds) {
                float score = VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT.compare(queryVector, vectors[docId]);
                collector.collect(docId, score);
            }
            collector.topDocs();
            latencies[iter] = System.nanoTime() - start;
        }

        printPercentiles("FP32 Rerank", latencies, annResults.scoreDocs.length);
    }

    private static void printPercentiles(String label, long[] latenciesNanos, int numCandidates) {
        Arrays.sort(latenciesNanos);
        int n = latenciesNanos.length;
        double p50 = latenciesNanos[(int) (n * 0.50)] / 1000.0;
        double p90 = latenciesNanos[(int) (n * 0.90)] / 1000.0;
        double p95 = latenciesNanos[(int) (n * 0.95)] / 1000.0;
        double p99 = latenciesNanos[(int) (n * 0.99)] / 1000.0;
        double p999 = latenciesNanos[Math.min((int) (n * 0.999), n - 1)] / 1000.0;
        System.out.println(label + " (" + numCandidates + " candidates, " + n + " iterations):");
        System.out.printf("  p50  = %10.2f µs%n", p50);
        System.out.printf("  p90  = %10.2f µs%n", p90);
        System.out.printf("  p95  = %10.2f µs%n", p95);
        System.out.printf("  p99  = %10.2f µs%n", p99);
        System.out.printf("  p999 = %10.2f µs%n", p999);
    }

    // ===== Shared setup helpers (reused from FaissBBQRecallValidationTests) =====

    private Map<String, Object> buildFaissParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("name", "hnsw");
        params.put("data_type", "float");
        params.put("index_description", "BHNSW16,Flat");
        params.put("spaceType", "innerproduct");
        Map<String, Object> sub = new HashMap<>();
        params.put("parameters", sub);
        sub.put("ef_search", 256);
        sub.put("ef_construction", 256);
        sub.put("m", 16);
        sub.put("encoder", Collections.emptyMap());
        sub.put("indexThreadQty", 1);
        return params;
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

    private long ingestIntoFaiss(Directory directory, int dimension, int numVectors, SegmentInfo segmentInfo, FieldInfo fieldInfo)
        throws IOException {
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
        SegmentReadState readState = new SegmentReadState(
            directory, segmentInfo, fieldInfos, IOContext.DEFAULT, "test_field"
        );
        Lucene102BinaryFlatVectorsScorer scorer = new Lucene102BinaryFlatVectorsScorer();
        try (BBQReader reader = new BBQReader(readState, scorer)) {
            FloatVectorValues floatValues = reader.getFloatVectorValues("test_field");
            BBQReader.BinarizedVectorValues binarized = (BBQReader.BinarizedVectorValues) floatValues;
            BinarizedByteVectorValues quantized = binarized.quantizedVectorValues;

            int quantizedVecBytes = quantized.vectorValue(0).length;
            float centroidDp = quantized.getCentroidDP();

            long indexAddr = FaissService.initBBQIndex(numVectors, dimension, buildFaissParams(), centroidDp, quantizedVecBytes);
            passQuantizedVectors(indexAddr, binarized);

            int batchSize = 500;
            int[] docIds = new int[batchSize];
            int numAdded = 0;
            int remaining = numVectors;
            while (remaining > 0) {
                int count = Math.min(batchSize, remaining);
                for (int i = 0; i < count; i++) {
                    docIds[i] = numAdded + i;
                }
                FaissService.addDocsToBBQIndex(indexAddr, docIds, count, numAdded);
                numAdded += count;
                remaining -= count;
            }
            return indexAddr;
        }
    }

    private void passQuantizedVectors(long indexAddr, BBQReader.BinarizedVectorValues binarized) throws IOException {
        final int batchSize = 500;
        byte[] buffer = null;
        for (int i = 0; i < binarized.size(); ) {
            int loopSize = Math.min(binarized.size() - i, batchSize);
            for (int j = 0, o = 0; j < loopSize; ++j) {
                byte[] binaryVector = binarized.quantizedVectorValues.vectorValue(i + j);
                if (buffer == null) {
                    buffer = new byte[(binaryVector.length + Integer.BYTES * 4) * batchSize];
                }
                OptimizedScalarQuantizer.QuantizationResult qr = binarized.quantizedVectorValues.getCorrectiveTerms(i + j);

                System.arraycopy(binaryVector, 0, buffer, o, binaryVector.length);
                o += binaryVector.length;

                int bits = Float.floatToRawIntBits(qr.lowerInterval());
                buffer[o++] = (byte) bits; buffer[o++] = (byte) (bits >>> 8);
                buffer[o++] = (byte) (bits >>> 16); buffer[o++] = (byte) (bits >>> 24);

                bits = Float.floatToRawIntBits(qr.upperInterval());
                buffer[o++] = (byte) bits; buffer[o++] = (byte) (bits >>> 8);
                buffer[o++] = (byte) (bits >>> 16); buffer[o++] = (byte) (bits >>> 24);

                bits = Float.floatToRawIntBits(qr.additionalCorrection());
                buffer[o++] = (byte) bits; buffer[o++] = (byte) (bits >>> 8);
                buffer[o++] = (byte) (bits >>> 16); buffer[o++] = (byte) (bits >>> 24);

                bits = qr.quantizedComponentSum();
                buffer[o++] = (byte) bits; buffer[o++] = (byte) (bits >>> 8);
                buffer[o++] = (byte) (bits >>> 16); buffer[o++] = (byte) (bits >>> 24);
            }
            FaissService.passBBQVectors(indexAddr, buffer, loopSize);
            i += loopSize;
        }
    }
}
