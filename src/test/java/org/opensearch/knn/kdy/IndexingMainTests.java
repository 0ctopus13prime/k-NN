/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.kdy;

import io.jhdf.HdfFile;
import io.jhdf.api.Dataset;
import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040ScalarQuantizedKnnVectorsFormat;
import org.opensearch.knn.index.codec.util.UnitTestCodec;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Multi-threaded indexing harness that ingests vectors from an HDF5 file into N Lucene shards,
 * each backed by Faiss HNSW (1-bit scalar-quantized) via {@link Faiss1040ScalarQuantizedKnnVectorsFormat}.
 *
 * <p>Pipeline: HDF5 reader (main thread) → bounded {@link BlockingQueue} → indexer threads →
 * one {@link IndexWriter} per shard.
 *
 * <p>Designed to be invoked as a single JUnit test entry point so {@code ./gradlew test} (or an
 * IDE) can launch it without a custom main method.
 */
@Log4j2
public class IndexingMainTests extends KNNTestCase {

    /* =========================================================================================
     * PARAMETERS — edit and re-run.
     * =========================================================================================
     */
    private static final String ROOT_DIR = "/Users/kdooyong/workspace/radial-test/data/sq-1m";
    private static final String HDF5_PATH =
        "/Users/kdooyong/workspace/radial-test/tmp/java-explore/data/cohere-1m-radial-ground-truth-auto-threshold.hdf5";
    private static final String HDF5_TRAIN_DATASET = "/train";

    private static final int NUM_SHARDS = 3;
    private static final int NUM_CORES = 6;
    private static final int NUM_DOCS = -1;          // -1 means "ingest the whole /train dataset"
    private static final int QUEUE_SIZE = 1000;
    private static final int BULK_SIZE = 500;
    private static final int HDF5_SLICE_ROWS = 4096;
    private static final double RAM_BUFFER_MB = 128.0;

    private static final SpaceType SPACE_TYPE = SpaceType.INNER_PRODUCT;
    private static final VectorSimilarityFunction SIMILARITY = VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;

    // Faiss SQ method parameters: BHNSW16,Flat with 1-bit SQ encoder.
    private static final int HNSW_M = 16;
    private static final int HNSW_EF_CONSTRUCTION = 256;
    private static final int HNSW_EF_SEARCH = 256;
    private static final String FAISS_PARAMETERS_JSON =
        "{" + "\"index_description\":\"BHNSW" + HNSW_M + ",Flat\"," + "\"spaceType\":\"" + SPACE_TYPE.getValue() + "\","
        + "\"name\":\"hnsw\"," + "\"data_type\":\"float\"," + "\"parameters\":{" + "\"ef_search\":" + HNSW_EF_SEARCH + ","
        + "\"ef_construction\":" + HNSW_EF_CONSTRUCTION + "," + "\"m\":" + HNSW_M + "," + "\"encoder\":{\"name\":\"sq\",\"bits\":1}" + "}"
        + "}";

    private static final String ID_FIELD = "_id";
    private static final String VECTOR_FIELD = "target_field";

    private static final long QUEUE_OFFER_TIMEOUT_MS = 2_000L;

    /* ========================================================================================= */

    @SneakyThrows
    public void testIndex() {
        final long startNanos = System.nanoTime();
        logParameters();

        final int numIndexThreads = Math.max(1, NUM_CORES - 1);
        final Path rootDir = Path.of(ROOT_DIR);
        Files.createDirectories(rootDir);
        cleanExistingShards(rootDir);

        final Path paramsFile = rootDir.resolve("indexing-parameters.json");
        writeParametersJson(paramsFile);
        log.info("Wrote indexing parameters to {}", paramsFile);

        final Codec codec = new UnitTestCodec(Faiss1040ScalarQuantizedKnnVectorsFormat::new);

        final Directory[] directories = new Directory[NUM_SHARDS];
        final IndexWriter[] writers = new IndexWriter[NUM_SHARDS];
        try {
            for (int s = 0; s < NUM_SHARDS; s++) {
                final Path shardDir = rootDir.resolve("shard-" + s);
                Files.createDirectories(shardDir);
                directories[s] = FSDirectory.open(shardDir);
                final IndexWriterConfig cfg = new IndexWriterConfig().setCodec(codec)
                    .setMergePolicy(NoMergePolicy.INSTANCE)
                    .setRAMBufferSizeMB(RAM_BUFFER_MB)
                    .setUseCompoundFile(false);
                writers[s] = new IndexWriter(directories[s], cfg);
                log.info("Opened IndexWriter for shard {} at {}", s, shardDir);
            }

            final BlockingQueue<Task> queue = new LinkedBlockingQueue<>(QUEUE_SIZE);
            log.info("Created task queue with capacity {}", QUEUE_SIZE);

            final IndexerThread[] indexers = new IndexerThread[numIndexThreads];
            final Thread[] threads = new Thread[numIndexThreads];
            for (int i = 0; i < numIndexThreads; i++) {
                indexers[i] = new IndexerThread("indexer-" + i, writers, queue);
                threads[i] = new Thread(indexers[i], "indexer-" + i);
                threads[i].start();
            }
            log.info("Spawned {} indexer threads", numIndexThreads);

            ingestVectors(queue);

            waitUntilIdle(indexers, queue, "after ingest");

            log.info("Enqueueing {} PoisonPills", numIndexThreads);
            for (int i = 0; i < numIndexThreads; i++) {
                putWithBackpressureLogging(queue, PoisonPill.INSTANCE);
            }
            for (final Thread t : threads) {
                t.join();
            }
            log.info("All indexer threads joined");
        } finally {
            for (int s = 0; s < writers.length; s++) {
                closeQuietly(writers[s], "IndexWriter[" + s + "]");
            }
            for (int s = 0; s < directories.length; s++) {
                closeQuietly(directories[s], "Directory[" + s + "]");
            }
        }

        final long elapsedMs = (System.nanoTime() - startNanos) / 1_000_000L;
        log.info("Indexing finished in {} ms", elapsedMs);
    }

    /* =========================================================================================
     * Ingest
     * =========================================================================================
     */

    private void ingestVectors(BlockingQueue<Task> queue) throws InterruptedException {
        final BulkRequest[] buckets = new BulkRequest[NUM_SHARDS];
        for (int s = 0; s < NUM_SHARDS; s++) {
            buckets[s] = new BulkRequest(s);
        }

        try (HdfFile hdf = new HdfFile(Path.of(HDF5_PATH).toFile())) {
            final Dataset dataset = (Dataset) hdf.getByPath(HDF5_TRAIN_DATASET);
            final int[] dims = dataset.getDimensions();
            if (dims.length != 2) {
                throw new IllegalStateException("Expected 2-D '" + HDF5_TRAIN_DATASET + "' dataset, got " + dims.length + "-D");
            }
            final int totalRows = dims[0];
            final int dimensions = dims[1];
            final int limit = (NUM_DOCS < 0) ? totalRows : Math.min(NUM_DOCS, totalRows);
            log.info("Beginning ingest of {} vectors (dim={}, dataset has {} rows)", limit, dimensions, totalRows);

            long enqueued = 0L;
            int row = 0;
            while (row < limit) {
                final int sliceRows = Math.min(HDF5_SLICE_ROWS, limit - row);
                final long[] offset = { row, 0 };
                final int[] shape = { sliceRows, dimensions };
                final float[][] slice = (float[][]) dataset.getData(offset, shape);
                for (int i = 0; i < sliceRows; i++) {
                    final int ordinal = row + i;
                    final String docId = Integer.toString(ordinal);
                    final int shard = Math.floorMod(docId.hashCode(), NUM_SHARDS);
                    final BulkRequest bucket = buckets[shard];
                    bucket.add(new VectorRecord(docId, slice[i]));
                    if (bucket.isFull()) {
                        putWithBackpressureLogging(queue, bucket);
                        buckets[shard] = new BulkRequest(shard);
                        enqueued++;
                        if (enqueued % 200L == 0L) {
                            log.info("Ingest progress: {} bulk requests enqueued, ~{} docs read", enqueued, ordinal + 1);
                        }
                    }
                }
                row += sliceRows;
            }

            for (int s = 0; s < NUM_SHARDS; s++) {
                if (!buckets[s].isEmpty()) {
                    putWithBackpressureLogging(queue, buckets[s]);
                    enqueued++;
                }
            }
            log.info("Ingest complete: {} bulk requests total, {} vectors read", enqueued, limit);
        }
    }

    /* =========================================================================================
     * Document mapping — Faiss SQ field with 1-bit encoder
     * =========================================================================================
     */

    private static Document toDocument(VectorRecord record) {
        final Document doc = new Document();
        doc.add(new StringField(ID_FIELD, record.docId(), Field.Store.YES));
        final FieldType vectorFieldType = new FieldType();
        vectorFieldType.setTokenized(false);
        vectorFieldType.setIndexOptions(IndexOptions.NONE);
        vectorFieldType.putAttribute(KNNVectorFieldMapper.KNN_FIELD, "true");
        vectorFieldType.putAttribute(KNNConstants.KNN_METHOD, KNNConstants.METHOD_HNSW);
        vectorFieldType.putAttribute(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName());
        vectorFieldType.putAttribute(KNNConstants.SPACE_TYPE, SPACE_TYPE.getValue());
        vectorFieldType.putAttribute(KNNConstants.PARAMETERS, FAISS_PARAMETERS_JSON);
        vectorFieldType.putAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float");
        vectorFieldType.putAttribute(KNNConstants.SQ_CONFIG, "bits=1");
        vectorFieldType.setVectorAttributes(record.vector().length, VectorEncoding.FLOAT32, SIMILARITY);
        vectorFieldType.freeze();
        doc.add(new KnnFloatVectorField(VECTOR_FIELD, record.vector(), vectorFieldType));
        return doc;
    }

    /* =========================================================================================
     * Concurrency: tasks, indexer threads, idle monitor
     * =========================================================================================
     */

    private sealed interface Task permits BulkRequest, PoisonPill {}

    private record VectorRecord(String docId, float[] vector) {}

    private static final class BulkRequest implements Task {
        final int shardId;
        final java.util.List<VectorRecord> records = new java.util.ArrayList<>(BULK_SIZE);

        BulkRequest(int shardId) {
            this.shardId = shardId;
        }

        boolean isFull() {
            return records.size() >= BULK_SIZE;
        }

        boolean isEmpty() {
            return records.isEmpty();
        }

        void add(VectorRecord record) {
            records.add(record);
        }
    }

    private static final class PoisonPill implements Task {
        static final PoisonPill INSTANCE = new PoisonPill();

        private PoisonPill() {}
    }

    @Log4j2
    private static final class IndexerThread implements Runnable {
        private final String name;
        private final IndexWriter[] writers;
        private final BlockingQueue<Task> queue;
        volatile boolean idle = true;

        IndexerThread(String name, IndexWriter[] writers, BlockingQueue<Task> queue) {
            this.name = name;
            this.writers = writers;
            this.queue = queue;
        }

        boolean isIdle() {
            return idle;
        }

        @Override
        public void run() {
            log.info("[{}] indexer thread started", name);
            try {
                while (true) {
                    final Task task = queue.take();
                    idle = false;
                    try {
                        if (task instanceof BulkRequest bulk) {
                            handleBulk(bulk);
                        } else if (task instanceof PoisonPill) {
                            log.info("[{}] received PoisonPill — exiting", name);
                            return;
                        }
                    } finally {
                        idle = true;
                    }
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                log.warn("[{}] interrupted; exiting", name);
            } catch (Exception e) {
                log.error("[{}] fatal error; exiting", name, e);
            } finally {
                log.info("[{}] indexer thread exited", name);
            }
        }

        private void handleBulk(BulkRequest bulk) throws IOException {
            final IndexWriter writer = writers[bulk.shardId];
            for (final VectorRecord record : bulk.records) {
                writer.addDocument(toDocument(record));
            }
            log.debug("[{}] indexed {} docs into shard {}", name, bulk.records.size(), bulk.shardId);
        }
    }

    private static void waitUntilIdle(IndexerThread[] threads, BlockingQueue<Task> queue, String label) throws InterruptedException {
        final long pollMs = 100L;
        final int confirmConsecutive = 2;
        final long progressLogIntervalMs = 5_000L;
        log.info("Waiting for engine to become idle: {}", label);
        int consecutiveIdle = 0;
        long lastProgressLogNs = System.nanoTime();
        while (consecutiveIdle < confirmConsecutive) {
            Thread.sleep(pollMs);
            final int queueSize = queue.size();
            int busy = 0;
            for (final IndexerThread t : threads) {
                if (!t.isIdle()) busy++;
            }
            if (queueSize == 0 && busy == 0) {
                consecutiveIdle++;
            } else {
                consecutiveIdle = 0;
            }
            final long nowNs = System.nanoTime();
            if ((nowNs - lastProgressLogNs) / 1_000_000L >= progressLogIntervalMs) {
                log.info("Idle wait '{}': queueSize={}, busyThreads={}/{}", label, queueSize, busy, threads.length);
                lastProgressLogNs = nowNs;
            }
        }
        log.info("Engine is idle: {}", label);
    }

    private static void putWithBackpressureLogging(BlockingQueue<Task> queue, Task task) throws InterruptedException {
        long waitedMs = 0L;
        while (true) {
            if (queue.offer(task, QUEUE_OFFER_TIMEOUT_MS, TimeUnit.MILLISECONDS)) {
                if (waitedMs > 0L) {
                    log.info("Main resumed enqueue after {} ms back-pressure wait", waitedMs);
                }
                return;
            }
            waitedMs += QUEUE_OFFER_TIMEOUT_MS;
            log.info("Main is back-pressured: queue full (size={}/{}), waited {} ms so far", queue.size(), QUEUE_SIZE, waitedMs);
        }
    }

    /* =========================================================================================
     * File system + parameter persistence
     * =========================================================================================
     */

    private static void cleanExistingShards(Path rootDir) throws IOException {
        if (!Files.isDirectory(rootDir)) return;
        try (var stream = Files.newDirectoryStream(rootDir, "shard-*")) {
            for (final Path shard : stream) {
                if (!Files.isDirectory(shard)) continue;
                log.info("Removing existing shard directory {}", shard);
                deleteRecursively(shard);
            }
        }
    }

    private static void deleteRecursively(Path path) throws IOException {
        Files.walkFileTree(
            path, new java.nio.file.SimpleFileVisitor<>() {
                @Override
                public java.nio.file.FileVisitResult visitFile(Path file, java.nio.file.attribute.BasicFileAttributes attrs)
                    throws IOException {
                    Files.delete(file);
                    return java.nio.file.FileVisitResult.CONTINUE;
                }

                @Override
                public java.nio.file.FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
                    if (exc != null) throw exc;
                    Files.delete(dir);
                    return java.nio.file.FileVisitResult.CONTINUE;
                }
            }
        );
    }

    private static void writeParametersJson(Path path) throws IOException {
        final String json =
            "{\n" + "  \"rootDir\": \"" + ROOT_DIR + "\",\n" + "  \"numShards\": " + NUM_SHARDS + ",\n" + "  \"hdf5Path\": \"" + HDF5_PATH
            + "\",\n" + "  \"numCores\": " + NUM_CORES + ",\n" + "  \"numDocs\": " + (NUM_DOCS < 0 ? "null" : NUM_DOCS) + ",\n"
            + "  \"queueSize\": " + QUEUE_SIZE + ",\n" + "  \"spaceType\": \"" + SPACE_TYPE.getValue() + "\",\n" + "  \"similarity\": \""
            + SIMILARITY.name() + "\",\n" + "  \"hnswM\": " + HNSW_M + ",\n" + "  \"hnswEfConstruction\": " + HNSW_EF_CONSTRUCTION + ",\n"
            + "  \"hnswEfSearch\": " + HNSW_EF_SEARCH + ",\n" + "  \"vectorFormat\": \"Faiss1040ScalarQuantizedKnnVectorsFormat\"\n"
            + "}\n";
        Files.writeString(path, json, StandardCharsets.UTF_8);
    }

    private static void closeQuietly(AutoCloseable closeable, String label) {
        if (closeable == null) return;
        try {
            closeable.close();
        } catch (Exception e) {
            log.warn("Error closing {}: {}", label, e.toString());
        }
    }

    private static void logParameters() {
        log.info("=== IndexingMain parameters ===");
        log.info("  rootDir              = {}", ROOT_DIR);
        log.info("  hdf5Path             = {}", HDF5_PATH);
        log.info("  numShards            = {}", NUM_SHARDS);
        log.info("  numCores             = {}", NUM_CORES);
        log.info("  numDocs              = {}", NUM_DOCS < 0 ? "all" : NUM_DOCS);
        log.info("  queueSize            = {}", QUEUE_SIZE);
        log.info("  bulkSize             = {}", BULK_SIZE);
        log.info("  ramBufferMb          = {}", RAM_BUFFER_MB);
        log.info("  spaceType            = {}", SPACE_TYPE.getValue());
        log.info("  similarity           = {}", SIMILARITY);
        log.info("  hnsw.m               = {}", HNSW_M);
        log.info("  hnsw.efConstruction  = {}", HNSW_EF_CONSTRUCTION);
        log.info("  hnsw.efSearch        = {}", HNSW_EF_SEARCH);
        log.info("  vectorFormat         = Faiss1040ScalarQuantizedKnnVectorsFormat");
    }
}
