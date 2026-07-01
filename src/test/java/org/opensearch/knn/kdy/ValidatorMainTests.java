/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.kdy;

import io.jhdf.HdfFile;
import io.jhdf.api.Dataset;
import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.MultiReader;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.FloatVectorSimilarityQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Recall benchmark for radial search over the {@code shard-*} indexes produced by
 * {@code IndexingMainTests} (and optionally compacted by {@code ForceMergeMainTests}).
 *
 * <p>For each query in the HDF5 {@code /test} dataset, runs {@link FloatVectorSimilarityQuery}
 * across all shards, optionally rescores 1-bit Faiss SQ hits with exact float similarity, and
 * compares against the ground-truth IDs in {@code /radial_neighbors}.
 *
 * <p>Reports: per-query recall, recall avg/median/min/max over all queries and over the
 * "large-truth" subset (truth size > {@link #LARGE_TRUTH_THRESHOLD}), plus per-shard segment counts.
 */
@Log4j2
public class ValidatorMainTests extends KNNTestCase {

    /* =========================================================================================
     * PARAMETERS — edit and re-run.
     * =========================================================================================
     */
    private static final String ROOT_DIR = "/Users/kdooyong/workspace/radial-test/data/sq-1m";
    private static final String HDF5_PATH =
        "/Users/kdooyong/workspace/radial-test/tmp/java-explore/data/cohere-1m-radial-ground-truth-auto-threshold.hdf5";
    private static final String HDF5_TEST_DATASET = "/test";
    private static final String HDF5_RADIAL_NEIGHBORS_DATASET = "/radial_neighbors";

    private static final int NUM_QUERIES = 70;
    private static final float MIN_SCORE = 155.495f - 2;
    private static final boolean RESCORE = true;
    private static final int NUM_CORES = 6;

    // Constants that mirror tmp/java-explore/ValidatorMain
    private static final VectorSimilarityFunction SIMILARITY = VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;
    private static final String VECTOR_FIELD = "target_field";
    private static final String ID_FIELD = "_id";
    private static final int UNBOUNDED_HITS = Integer.MAX_VALUE;
    private static final int LARGE_TRUTH_THRESHOLD = 50;
    private static final float TRAVERSAL_RATIO = 0.8F;

    /* ========================================================================================= */

    @SneakyThrows
    public void testValidate() {
        final long startNanos = System.nanoTime();
        logParameters();

        final Path rootDir = Path.of(ROOT_DIR);
        if (!Files.isDirectory(rootDir)) {
            throw new IllegalStateException("rootDir is not a directory: " + rootDir);
        }

        final List<Path> shardDirs = listShardDirs(rootDir);
        if (shardDirs.isEmpty()) {
            throw new IllegalStateException("No shard-* directories found under " + rootDir);
        }

        final Directory[] directories = new Directory[shardDirs.size()];
        final DirectoryReader[] readers = new DirectoryReader[shardDirs.size()];
        for (int s = 0; s < shardDirs.size(); s++) {
            directories[s] = FSDirectory.open(shardDirs.get(s));
            readers[s] = DirectoryReader.open(directories[s]);
            log.info("Opened shard {} at {} ({} docs)", s, shardDirs.get(s), readers[s].numDocs());
        }

        try (MultiReader multi = new MultiReader(readers, false)) {
            final float[][] testVectors = readFloat2D(HDF5_TEST_DATASET, NUM_QUERIES);
            final long[][] groundTruth = readLong2D(HDF5_RADIAL_NEIGHBORS_DATASET, NUM_QUERIES);

            final int numQueries = testVectors.length;
            final double[] recalls = new double[numQueries];
            final AtomicBoolean rescoreEverActed = new AtomicBoolean(false);

            final int workerCount = Math.min(NUM_CORES, numQueries);
            log.info("Running {} queries across {} worker thread(s)", numQueries, workerCount);

            final ExecutorService pool = Executors.newFixedThreadPool(workerCount, r -> {
                final Thread t = new Thread(r);
                t.setName("validator-" + t.threadId());
                t.setDaemon(true);
                return t;
            });
            try {
                final List<Callable<Void>> tasks = new ArrayList<>(numQueries);
                for (int q = 0; q < numQueries; q++) {
                    final int qi = q;
                    tasks.add(() -> {
                        final IndexSearcher searcher = new IndexSearcher(multi);
                        runOneQuery(searcher, multi, testVectors[qi], groundTruth[qi], qi, recalls, rescoreEverActed);
                        return null;
                    });
                }
                final List<Future<Void>> futures = pool.invokeAll(tasks);
                for (final Future<Void> f : futures) {
                    f.get();
                }
            } finally {
                pool.shutdown();
            }

            final List<Double> recallsLargeTruth = new ArrayList<>();
            for (int q = 0; q < numQueries; q++) {
                if (countNonNegative(groundTruth[q]) > LARGE_TRUTH_THRESHOLD) {
                    recallsLargeTruth.add(recalls[q]);
                }
            }

            printSummary(recalls, rescoreEverActed.get());
            final double[] largeTruthArr = recallsLargeTruth.stream().mapToDouble(Double::doubleValue).toArray();
            printRecallStats("Large-truth recall (truth > " + LARGE_TRUTH_THRESHOLD + ")", largeTruthArr, numQueries);
            printSegmentCounts(shardDirs, readers);
        } finally {
            for (final DirectoryReader r : readers) {
                closeQuietly(r, "DirectoryReader");
            }
            for (final Directory d : directories) {
                closeQuietly(d, "Directory");
            }
        }

        final long elapsedMs = (System.nanoTime() - startNanos) / 1_000_000L;
        log.info("Validator finished in {} ms", elapsedMs);
    }

    /* =========================================================================================
     * Per-query: search + (optional) rescore + recall
     * =========================================================================================
     */

    private static void runOneQuery(
        IndexSearcher searcher,
        IndexReader multi,
        float[] query,
        long[] groundTruthRow,
        int q,
        double[] recallsOut,
        AtomicBoolean rescoreEverActed
    ) throws IOException {
        final FloatVectorSimilarityQuery vectorQuery = new FloatVectorSimilarityQuery(
            VECTOR_FIELD,
            query,
            TRAVERSAL_RATIO * MIN_SCORE,
            MIN_SCORE
        );
        final TopDocs hits = searcher.search(vectorQuery, UNBOUNDED_HITS);

        final Set<Integer> resultIds;
        int prunedByRescore = 0;
        if (RESCORE) {
            final int[] rescoreCounts = new int[2];
            resultIds = collectIdsWithRescore(multi, hits, query, rescoreCounts);
            prunedByRescore = rescoreCounts[1];
            if (prunedByRescore > 0) {
                rescoreEverActed.set(true);
            }
        } else {
            resultIds = collectIds(multi, hits);
        }

        final Set<Long> truth = parseGroundTruth(groundTruthRow);
        final double recall = computeRecall(resultIds, truth);
        recallsOut[q] = recall;

        log.info(
            "Query {}: hits={}, truth={}, recall={}{}",
            q,
            hits.scoreDocs.length,
            truth.size(),
            String.format(java.util.Locale.ROOT, "%.4f", recall),
            prunedByRescore > 0 ? " (rescore pruned " + prunedByRescore + " of " + hits.scoreDocs.length + ")" : ""
        );
    }

    private static Set<Integer> collectIds(IndexReader reader, TopDocs hits) throws IOException {
        final Set<Integer> out = new HashSet<>();
        final StoredFields stored = reader.storedFields();
        for (final ScoreDoc sd : hits.scoreDocs) {
            final String id = stored.document(sd.doc).get(ID_FIELD);
            Objects.requireNonNull(id);
            out.add(Integer.parseInt(id));
        }
        return out;
    }

    private static Set<Integer> collectIdsWithRescore(IndexReader reader, TopDocs hits, float[] query, int[] excludeCounts)
        throws IOException {
        final Set<Integer> out = new HashSet<>();
        final StoredFields stored = reader.storedFields();
        final float threshold = MIN_SCORE;
        final List<LeafReaderContext> leaves = reader.leaves();

        final ScoreDoc[] sorted = hits.scoreDocs.clone();
        Arrays.sort(sorted, Comparator.comparingInt(a -> a.doc));

        int leafIdx = -1;
        FloatVectorValues values = null;
        KnnVectorValues.DocIndexIterator iterator = null;
        int leafBase = 0;
        int leafEnd = 0;

        for (final ScoreDoc sd : sorted) {
            excludeCounts[0]++;
            while (leafIdx < 0 || sd.doc >= leafEnd) {
                leafIdx++;
                final LeafReaderContext leafCtx = leaves.get(leafIdx);
                leafBase = leafCtx.docBase;
                leafEnd = (leafIdx + 1 < leaves.size()) ? leaves.get(leafIdx + 1).docBase : Integer.MAX_VALUE;
                values = leafCtx.reader().getFloatVectorValues(VECTOR_FIELD);
                if (values == null) {
                    throw new IllegalStateException(
                        "Leaf " + leafIdx + " has no '" + VECTOR_FIELD + "' vector values, but search returned doc " + sd.doc
                    );
                }
                iterator = values.iterator();
            }
            final int localDoc = sd.doc - leafBase;
            final int advanced = iterator.advance(localDoc);
            if (advanced != localDoc) {
                throw new IllegalStateException(
                    "Vector iterator could not locate doc "
                        + sd.doc
                        + " (leaf "
                        + leafIdx
                        + ", localDoc "
                        + localDoc
                        + ", advanced to "
                        + advanced
                        + ")"
                );
            }
            final float[] original = values.vectorValue(iterator.index());
            final float trueSimilarity = SIMILARITY.compare(query, original);
            if (trueSimilarity >= threshold) {
                final String id = stored.document(sd.doc).get(ID_FIELD);
                if (id == null) {
                    throw new IllegalStateException("Doc " + sd.doc + " has no stored '" + ID_FIELD + "' field");
                }
                out.add(Integer.parseInt(id));
            } else {
                excludeCounts[1]++;
            }
        }
        return out;
    }

    /* =========================================================================================
     * Recall + reporting
     * =========================================================================================
     */

    private static Set<Long> parseGroundTruth(long[] row) {
        final Set<Long> out = new HashSet<>();
        for (final long v : row) {
            if (v < 0) break;
            out.add(v);
        }
        return out;
    }

    private static int countNonNegative(long[] row) {
        int count = 0;
        for (final long v : row) {
            if (v < 0) break;
            count++;
        }
        return count;
    }

    private static double computeRecall(Set<Integer> resultIds, Set<Long> truth) {
        if (truth.isEmpty()) return 1.0;
        int hit = 0;
        for (final Long t : truth) {
            if (resultIds.contains(t.intValue())) hit++;
        }
        return ((double) hit) / truth.size();
    }

    private static void printSummary(double[] recalls, boolean rescoreEverActed) {
        log.info("=== Validation Summary ===");
        log.info("  similarity          = {}", SIMILARITY);
        log.info("  minScore            = {}", MIN_SCORE);
        log.info("  rescoreRequested    = {}", RESCORE);
        log.info("  rescoreActuallyHit  = {}", rescoreEverActed);
        printRecallStats("Recall (all queries)", recalls, recalls.length);
    }

    private static void printRecallStats(String label, double[] recalls, int totalQueries) {
        log.info("--- {} ---", label);
        log.info("  numQueries          = {} (of {} total)", recalls.length, totalQueries);
        if (recalls.length == 0) {
            log.info("  no queries matched this bucket; skipping recall stats");
            return;
        }
        final double[] sorted = recalls.clone();
        Arrays.sort(sorted);
        double sum = 0.0;
        for (final double r : recalls)
            sum += r;
        final double avg = sum / recalls.length;
        final int n = sorted.length;
        final double median = (n % 2 == 1) ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
        final double min = sorted[0];
        final double max = sorted[n - 1];
        log.info("  recall.avg          = {}", String.format(java.util.Locale.ROOT, "%.4f", avg));
        log.info("  recall.median       = {}", String.format(java.util.Locale.ROOT, "%.4f", median));
        log.info("  recall.min          = {}", String.format(java.util.Locale.ROOT, "%.4f", min));
        log.info("  recall.max          = {}", String.format(java.util.Locale.ROOT, "%.4f", max));
    }

    private static void printSegmentCounts(List<Path> shardDirs, DirectoryReader[] readers) {
        log.info("--- Segments per shard ---");
        int totalSegments = 0;
        for (int s = 0; s < readers.length; s++) {
            final int segs = readers[s].leaves().size();
            totalSegments += segs;
            log.info("  shard {} ({}): {} segment(s), {} docs", s, shardDirs.get(s).getFileName(), segs, readers[s].numDocs());
        }
        log.info("  total: {} segment(s) across {} shard(s)", totalSegments, readers.length);
    }

    /* =========================================================================================
     * HDF5 helpers
     * =========================================================================================
     */

    private static float[][] readFloat2D(String datasetPath, int limitRows) {
        try (HdfFile hdf = new HdfFile(Path.of(HDF5_PATH).toFile())) {
            final Dataset ds = (Dataset) hdf.getByPath(datasetPath);
            final int[] dims = ds.getDimensions();
            if (dims.length != 2) {
                throw new IllegalStateException("Expected 2-D dataset at " + datasetPath + ", got " + dims.length + "-D");
            }
            final int rows = (limitRows < 0) ? dims[0] : Math.min(limitRows, dims[0]);
            final int cols = dims[1];
            final long[] offset = { 0, 0 };
            final int[] shape = { rows, cols };
            final float[][] data = (float[][]) ds.getData(offset, shape);
            log.info("Read float dataset {}: {} rows x {} cols (of {} total)", datasetPath, rows, cols, dims[0]);
            return data;
        }
    }

    private static long[][] readLong2D(String datasetPath, int limitRows) {
        try (HdfFile hdf = new HdfFile(Path.of(HDF5_PATH).toFile())) {
            final Dataset ds = (Dataset) hdf.getByPath(datasetPath);
            final int[] dims = ds.getDimensions();
            if (dims.length != 2) {
                throw new IllegalStateException("Expected 2-D dataset at " + datasetPath + ", got " + dims.length + "-D");
            }
            final int rows = (limitRows < 0) ? dims[0] : Math.min(limitRows, dims[0]);
            final int cols = dims[1];
            final long[] offset = { 0, 0 };
            final int[] shape = { rows, cols };
            final long[][] data = (long[][]) ds.getData(offset, shape);
            log.info("Read long dataset {}: {} rows x {} cols (of {} total)", datasetPath, rows, cols, dims[0]);
            return data;
        }
    }

    /* =========================================================================================
     * File system helpers
     * =========================================================================================
     */

    private static List<Path> listShardDirs(Path rootDir) throws IOException {
        final List<Path> shards = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(rootDir, "shard-*")) {
            for (final Path p : stream) {
                if (Files.isDirectory(p)) shards.add(p);
            }
        }
        shards.sort((a, b) -> Integer.compare(shardIndex(a), shardIndex(b)));
        return shards;
    }

    private static int shardIndex(Path shard) {
        final String name = shard.getFileName().toString();
        return Integer.parseInt(name.substring("shard-".length()));
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
        log.info("=== ValidatorMain parameters ===");
        log.info("  rootDir         = {}", ROOT_DIR);
        log.info("  hdf5Path        = {}", HDF5_PATH);
        log.info("  numQueries      = {}", NUM_QUERIES);
        log.info("  minScore        = {}", MIN_SCORE);
        log.info("  rescore         = {}", RESCORE);
        log.info("  numCores        = {}", NUM_CORES);
        log.info("  similarity      = {}", SIMILARITY);
        log.info("  vectorField     = {}", VECTOR_FIELD);
        log.info("  idField         = {}", ID_FIELD);
    }
}
