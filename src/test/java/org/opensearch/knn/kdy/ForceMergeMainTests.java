/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.kdy;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040ScalarQuantizedKnnVectorsFormat;
import org.opensearch.knn.index.codec.util.UnitTestCodec;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Force-merges every {@code shard-*} directory under {@link #ROOT_DIR} down to
 * {@link #TARGET_SEGMENTS} segments using {@link Faiss1040ScalarQuantizedKnnVectorsFormat}.
 * One worker thread per shard, capped at {@link #NUM_CORES}.
 *
 * <p>Run after {@code IndexingMainTests} has produced unmerged shards.
 */
@Log4j2
public class ForceMergeMainTests extends KNNTestCase {

    /* =========================================================================================
     * PARAMETERS — edit and re-run.
     * =========================================================================================
     */
    private static final String ROOT_DIR = "/Users/kdooyong/workspace/radial-test/data/sq-1m";
    private static final int TARGET_SEGMENTS = 1;
    private static final int NUM_CORES = 6;

    /* ========================================================================================= */

    @SneakyThrows
    public void testForceMerge() {
        final long startNanos = System.nanoTime();
        logParameters();

        final Path rootDir = Path.of(ROOT_DIR);
        if (!Files.isDirectory(rootDir)) {
            throw new IllegalStateException("--root is not a directory: " + rootDir);
        }

        final List<Path> shardDirs = listShardDirs(rootDir);
        if (shardDirs.isEmpty()) {
            throw new IllegalStateException("No shard-* directories found under " + rootDir);
        }

        final int workerCount = Math.min(NUM_CORES, shardDirs.size());
        log.info(
            "Running force-merge on {} shard(s) using {} worker thread(s), targetSegments={}",
            shardDirs.size(),
            workerCount,
            TARGET_SEGMENTS
        );

        final ExecutorService pool = Executors.newFixedThreadPool(workerCount, r -> {
            final Thread t = new Thread(r);
            t.setName("force-merge-" + t.threadId());
            t.setDaemon(true);
            return t;
        });
        try {
            final List<Callable<Void>> tasks = new ArrayList<>(shardDirs.size());
            for (int s = 0; s < shardDirs.size(); s++) {
                final int shardId = s;
                final Path shardDir = shardDirs.get(s);
                tasks.add(() -> {
                    forceMergeShard(shardId, shardDir);
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

        final long elapsedMs = (System.nanoTime() - startNanos) / 1_000_000L;
        log.info("Force-merge finished in {} ms", elapsedMs);
    }

    private static void forceMergeShard(int shardId, Path shardDir) throws IOException {
        final long t0 = System.nanoTime();
        final Codec codec = new UnitTestCodec(Faiss1040ScalarQuantizedKnnVectorsFormat::new);
        try (Directory directory = FSDirectory.open(shardDir)) {
            final IndexWriterConfig cfg = new IndexWriterConfig().setCodec(codec)
                .setMergePolicy(new TieredMergePolicy())
                .setUseCompoundFile(false);
            try (IndexWriter writer = new IndexWriter(directory, cfg)) {
                final int before = writer.getDocStats().numDocs;
                log.info("[shard-{}] forceMerge start: dir={}, target={}, docs={}", shardId, shardDir, TARGET_SEGMENTS, before);
                writer.forceMerge(TARGET_SEGMENTS, true);
                writer.commit();
                final long elapsedMs = (System.nanoTime() - t0) / 1_000_000L;
                log.info("[shard-{}] forceMerge done:  target={}, elapsed={} ms", shardId, TARGET_SEGMENTS, elapsedMs);
            }
        }
    }

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

    private static void logParameters() {
        log.info("=== ForceMergeMain parameters ===");
        log.info("  rootDir         = {}", ROOT_DIR);
        log.info("  targetSegments  = {}", TARGET_SEGMENTS);
        log.info("  numCores        = {}", NUM_CORES);
        log.info("  vectorFormat    = Faiss1040ScalarQuantizedKnnVectorsFormat");
    }
}
