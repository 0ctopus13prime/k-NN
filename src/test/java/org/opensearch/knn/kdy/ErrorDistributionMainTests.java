/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.kdy;

import com.carrotsearch.randomizedtesting.annotations.TimeoutSuite;
import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.TimeUnits;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.SplittableRandom;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Measures quantization-induced score error on Faiss SQ (1-bit) indexes without
 * relying on an external query set.
 *
 * <p>For each segment under each shard the test emits two distributions of signed
 * relative score error {@code (score_full - score_quant) / score_full}:
 * <ul>
 *   <li><b>Option A — self-score gap (SMOKE TEST, NOT a slack input).</b> Per doc {@code v},
 *       compare {@code similarity.compare(v, v)} against the ADC score of the doc against itself.
 *       Useful only as a sanity check that the SQ codec is wired up correctly — if A's positive
 *       tail is already large, Option B's numbers will be unreliable. Self-score sits at the
 *       maximum possible {@code s_full} for every space type, so relative error here is the
 *       most favorable case structurally; it is not a bound on what real off-diagonal queries
 *       experience.</li>
 *   <li><b>Option B — data-as-queries (slack-setting input).</b> Sample proxy queries from
 *       stored docs, score each against every doc in the same segment via the SQ codec's ADC
 *       path, and compare to the exact float similarity. Two filters guard the percentile:
 *       <ul>
 *         <li>{@code SCORE_BAND_LOW} (env var, optional) — if set, only pairs with
 *             {@code s_full ≥ band_low} are recorded. This is the population whose error can
 *             actually cause wrong reject/accept at the radial boundary; without it, far-pair
 *             relative-error noise dominates the histogram.</li>
 *         <li>One-sided (positive-tail) percentile reported alongside two-sided. Slack
 *             {@code min_score' = min_score * (1 - eps)} only addresses {@code s_quant < s_full}
 *             (positive {@code rel_err}); negative {@code rel_err} causes false accepts, which
 *             the rescore pass handles separately.</li>
 *       </ul></li>
 * </ul>
 *
 * <p>Per-segment percentiles plus a global rollup histogram (bin width 0.5%, range ±20%)
 * are logged so the output can be pasted into a notebook for plotting.
 *
 * <p>Configured via environment variables {@code DATA_DIR}, {@code VECTOR_FIELD},
 * {@code SPACE_TYPE} (e.g. {@code l2}, {@code innerproduct}). Optional:
 * {@code SCORE_BAND_LOW} (float, default unset → no filter), {@code NUM_CORES} (int).
 * Shards are integer-named subdirectories under {@code DATA_DIR}.
 */
@Log4j2
@TimeoutSuite(millis = 40 * TimeUnits.HOUR)
public class ErrorDistributionMainTests extends KNNTestCase {

    /* =========================================================================================
     * PARAMETERS — env vars are required; constants below are knobs.
     * =========================================================================================
     */
    private static final double OPTION_B_QUERY_SAMPLE_RATE = 0.20;
    private static final int OPTION_B_QUERY_CAP = 2000;
    private static final long SEED = 42L;

    // Reservoir kept per accumulator for percentile estimation. 100k samples → ~0.4 MB per
    // segment-level accumulator and merges to a global reservoir of the same size.
    private static final int RESERVOIR_SIZE = 100_000;
    // Per-worker (per-batch / per-proxy-query) reservoir size. Workers feed the segment
    // reservoir via merge(), and Accumulator.merge() samples FROM other.reservoir — so
    // workers that record() millions of pairs MUST keep a reservoir or the segment
    // reservoir stays empty and percentiles come back as NaN. 4k per worker, then the
    // segment-level reservoir of 100k aggregates fairly across all workers.
    private static final int WORKER_RESERVOIR_SIZE = 4_000;

    // Histogram of signed relative error: (-inf, -20%], then 80 bins of width 0.5%, then (+20%, +inf).
    private static final double HIST_RANGE = 0.20;
    private static final double HIST_BIN_WIDTH = 0.005;
    private static final int HIST_INNER_BINS = (int) Math.round((2 * HIST_RANGE) / HIST_BIN_WIDTH);
    private static final int HIST_TOTAL_BINS = HIST_INNER_BINS + 2;
    // Skip pairs whose |s_full| is below this floor — relative error blows up near zero
    // and would dominate the histogram with meaningless values.
    private static final double SCORE_DENOM_FLOOR = 1e-6;

    /* ========================================================================================= */

    @SneakyThrows
    public void testErrorDistribution() {
        final long startNanos = System.nanoTime();

        final Path dataDir = Path.of(requireEnv("DATA_DIR"));
        final String vectorField = requireEnv("VECTOR_FIELD");
        final SpaceType spaceType = SpaceType.getSpace(requireEnv("SPACE_TYPE"));
        final int numCores = parseIntEnv("NUM_CORES", Math.max(2, Runtime.getRuntime().availableProcessors() - 2));
        // SCORE_BAND_LOW: optional. When set, Option B only records (q, v) pairs whose exact
        // similarity sits above this floor. Use it to restrict the population to the score band
        // that the radial collector actually operates in — far-pair relative error is huge
        // and operationally irrelevant, and without this filter it dominates the histogram tail.
        final float scoreBandLow = parseFloatEnv("SCORE_BAND_LOW", Float.NEGATIVE_INFINITY);
        final VectorSimilarityFunction similarity = spaceType.getKnnVectorSimilarityFunction().getVectorSimilarityFunction();

        logParameters(dataDir, vectorField, spaceType, numCores, scoreBandLow);

        if (Files.isDirectory(dataDir) == false) {
            throw new IllegalStateException("DATA_DIR is not a directory: " + dataDir);
        }

        final List<Path> shardDirs = listShardDirs(dataDir);
        if (shardDirs.isEmpty()) {
            throw new IllegalStateException("No integer-named shard directories found under " + dataDir);
        }

        final ExecutorService pool = Executors.newFixedThreadPool(numCores, r -> {
            final Thread t = new Thread(r);
            t.setName("error-dist-" + t.threadId());
            t.setDaemon(true);
            return t;
        });

        final Accumulator globalA = Accumulator.create("A", RESERVOIR_SIZE);
        final Accumulator globalB = Accumulator.create("B", RESERVOIR_SIZE);

        try {
            for (final Path shardDir : shardDirs) {
                final int shardIdInt = shardIndex(shardDir);
                processShard(shardIdInt, shardDir, vectorField, similarity, scoreBandLow, pool, globalA, globalB);
            }
        } finally {
            pool.shutdown();
        }

        log.info("============================== GLOBAL ROLLUP ==============================");
        logAccumulator("global A (self-score gap)", globalA);
        logAccumulator("global B (data-as-queries)", globalB);
        logHistogram("global A histogram", globalA);
        logHistogram("global B histogram", globalB);

        log.info("============================== GLOBAL HISTOGRAM CSV ==============================");
        logHistogramCsv("global A (self-score gap)", globalA);
        logHistogramCsv("global B (data-as-queries)", globalB);

        log.info("============================== SLACK HEADLINE ==============================");
        logHeadline("global A (self-score gap, SMOKE)", globalA);
        logHeadline("global B (data-as-queries, SLACK)", globalB);

        final long elapsedMs = (System.nanoTime() - startNanos) / 1_000_000L;
        log.info("Error distribution test finished in {} ms", elapsedMs);
    }

    /* =========================================================================================
     * Shard / segment orchestration
     * =========================================================================================
     */

    private static void processShard(
        int shardId,
        Path shardDir,
        String vectorField,
        VectorSimilarityFunction similarity,
        float scoreBandLow,
        ExecutorService pool,
        Accumulator globalA,
        Accumulator globalB
    ) throws Exception {
        final Path indexDir = shardDir.resolve("index");
        if (Files.isDirectory(indexDir) == false) {
            throw new IllegalStateException("Expected Lucene index directory at " + indexDir);
        }
        log.info("[shard-{}] opening {}", shardId, indexDir);
        try (final Directory directory = FSDirectory.open(indexDir); final DirectoryReader reader = DirectoryReader.open(directory)) {
            final List<LeafReaderContext> leaves = reader.leaves();
            log.info("[shard-{}] {} doc(s) across {} segment(s)", shardId, reader.numDocs(), leaves.size());

            final Accumulator shardA = Accumulator.create("A", RESERVOIR_SIZE);
            final Accumulator shardB = Accumulator.create("B", RESERVOIR_SIZE);

            for (int segIdx = 0; segIdx < leaves.size(); segIdx++) {
                final LeafReaderContext leaf = leaves.get(segIdx);
                final Accumulator segA = Accumulator.create("A", RESERVOIR_SIZE);
                final Accumulator segB = Accumulator.create("B", RESERVOIR_SIZE);

                final long t0 = System.nanoTime();
                runOptionA(leaf, vectorField, similarity, pool, segA);
                final long t1 = System.nanoTime();
                runOptionB(leaf, vectorField, similarity, scoreBandLow, shardId, segIdx, pool, segB);
                final long t2 = System.nanoTime();

                log.info(
                    "[shard-{} seg-{}] A: {} pairs in {} ms | B: {} pairs in {} ms",
                    shardId,
                    segIdx,
                    segA.count(),
                    (t1 - t0) / 1_000_000L,
                    segB.count(),
                    (t2 - t1) / 1_000_000L
                );
                logAccumulator(String.format(Locale.ROOT, "shard-%d seg-%d A", shardId, segIdx), segA);
                logAccumulator(String.format(Locale.ROOT, "shard-%d seg-%d B", shardId, segIdx), segB);

                shardA.merge(segA);
                shardB.merge(segB);
            }

            log.info("---------- shard-{} rollup ----------", shardId);
            logAccumulator(String.format(Locale.ROOT, "shard-%d A", shardId), shardA);
            logAccumulator(String.format(Locale.ROOT, "shard-%d B", shardId), shardB);
            globalA.merge(shardA);
            globalB.merge(shardB);
        }
    }

    /* =========================================================================================
     * Option A — Self-score gap (per-doc reconstruction-quality probe).
     *
     * For every document v in the segment, we measure the gap between two scores of v
     * against itself:
     *
     *   s_full  = similarity.compare(v, v)
     *             - Exact float-precision similarity of v with itself.
     *             - For each space type this is a constant maximum:
     *                 L2                       -> 1.0  (Lucene 1/(1+d^2), d=0)
     *                 (MAXIMUM_)INNER_PRODUCT  -> Lucene's scaled max-inner-product of v with v
     *                                            (≈ 0.5 * ||v||^2 mapping for cohere-style vectors)
     *                 COSINE                   -> 1.0
     *               In every case, it's the score that the radial collector would compare
     *               `min_score` against if we somehow had the *true* vector at hand.
     *
     *   s_quant = scorer.score()  on the same doc, where the scorer comes from
     *             FloatVectorValues.scorer(v), i.e. the *quantized* ADC scoring path
     *             that the radial collector actually invokes during HNSW traversal.
     *             At search time Lucene 4-bit quantizes the query (v here) using the
     *             segment's OptimizedScalarQuantizer, computes the ADC distance against
     *             the stored 1-bit code + correction factors for the document, and
     *             returns a Lucene-normalized score.
     *
     * The error we record is the signed relative score gap:
     *
     *   rel_err = (s_full - s_quant) / s_full
     *
     * Interpretation: this is an "ideal-case" lower bound on the score error caused by
     * quantization. The query *is* the stored doc, so q and v come from exactly the same
     * distribution — there is no extra error from a "different-distribution" query. If
     * this gap is already X%, then for arbitrary queries the gap will be at least that bad,
     * and typically larger. Concretely: if we set min_score' = min_score * (1 - eps),
     * eps has to be at least the worst self-score gap we ever see, otherwise the radial
     * collector will reject docs whose quantized self-score has drifted below min_score
     * even though they should clearly survive (the doc is literally identical to the query).
     *
     * This is the cheap "everyone's a query" sanity baseline: one pass over the segment,
     * one ADC score per doc, no sampling.
     *
     * Parallelism:
     *   - The whole segment is split into batches of OPTION_A_BATCH_SIZE consecutive ords.
     *   - Each batch task gets its own FloatVectorValues.copy() (FloatVectorValues is not
     *     thread-safe — its IndexInput cursor is per-instance) and its own Accumulator.
     *   - Tasks are submitted to the shared pool via ExecutorCompletionService. The outer
     *     thread also drains via cs.take(), which means even if the pool is saturated by
     *     other shards' tasks the outer thread keeps progress moving — no deadlock.
     *   - On exception we cancel remaining futures so the pool doesn't keep doing useless
     *     work for a segment we've already abandoned.
     * =========================================================================================
     */

    // Per-batch task granularity. Big enough that ADC scoring dominates dispatch overhead
    // (~4k docs * ~tens of micros per ADC score = ~tens of ms of useful work per task),
    // small enough that work-stealing stays balanced across the pool.
    private static final int OPTION_A_BATCH_SIZE = 4096;

    private static void runOptionA(
        LeafReaderContext leaf,
        String vectorField,
        VectorSimilarityFunction similarity,
        ExecutorService pool,
        Accumulator segAcc
    ) throws Exception {
        // template is only used to read size() — it's not handed to any worker (a copy() is
        // created inside each task). leaf.reader().getFloatVectorValues(...) returns the
        // SQ-aware ScalarQuantizedFloatVectorValues, whose .scorer(q) routes through
        // Lucene104ScalarQuantizedVectorsReader and produces the same ADC scorer that the
        // radial-search collector consumes at query time.
        final FloatVectorValues template = leaf.reader().getFloatVectorValues(vectorField);
        if (template == null) {
            return;
        }
        final int size = template.size();
        if (size == 0) {
            return;
        }

        // Fan out: one task per [from, to) ord range. CompletionService lets the outer
        // thread drain results in completion order rather than submission order, so we
        // start merging the first finished batches while the slowest are still running.
        final CompletionService<Accumulator> cs = new ExecutorCompletionService<>(pool);
        final List<Future<Accumulator>> futures = new ArrayList<>();
        for (int start = 0; start < size; start += OPTION_A_BATCH_SIZE) {
            final int from = start;
            final int to = Math.min(size, start + OPTION_A_BATCH_SIZE);
            futures.add(cs.submit(() -> runOptionABatch(leaf, vectorField, similarity, from, to)));
        }

        try {
            // Drain in completion order. Each part Accumulator is merged into segAcc on
            // the outer thread — segAcc.merge() is not thread-safe, but only one thread
            // (this one) ever calls it. Workers only write to their own local Accumulator.
            for (int i = 0; i < futures.size(); i++) {
                final Accumulator part = cs.take().get();
                segAcc.merge(part);
            }
        } catch (Exception e) {
            // A worker threw — stop the rest from wasting CPU on stale work and propagate.
            for (final Future<Accumulator> f : futures) {
                f.cancel(true);
            }
            throw e;
        }
    }

    private static Accumulator runOptionABatch(
        LeafReaderContext leaf,
        String vectorField,
        VectorSimilarityFunction similarity,
        int fromOrd,
        int toOrd
    ) throws IOException {
        // Per-task accumulator. It needs its own reservoir so that on merge() the
        // segment-level accumulator has something to sample from — segment-level
        // reservoir merge is fed by other.reservoir, not by repeated record() calls.
        final Accumulator acc = Accumulator.create("A", WORKER_RESERVOIR_SIZE);

        // Each task owns its own FloatVectorValues. FloatVectorValues internally holds an
        // IndexInput cursor; sharing across threads would race the cursor and return the
        // wrong vector. copy() opens a new slice over the same file.
        final FloatVectorValues values = leaf.reader().getFloatVectorValues(vectorField).copy();

        for (int ord = fromOrd; ord < toOrd; ord++) {
            // Read the full-precision vector for ord. .clone() because the scorer below will
            // be using a *separate* copy() of FloatVectorValues, and we want v (used both as
            // the query passed to .scorer() and as the argument to similarity.compare(v,v))
            // to be a stable snapshot independent of either cursor.
            final float[] v = values.vectorValue(ord).clone();

            // ord -> docId. The HNSW iterator works in docId space, not ord space, so to
            // advance the scorer's iterator to "this doc" we need its docId. For dense
            // segments these are equal, but for sparse/deleted-doc segments they diverge.
            final int doc = values.ordToDoc(ord);

            // Build the quantized ADC scorer for "query = v". Internally this:
            //   1. 4-bit quantizes v against the segment centroid using OptimizedScalarQuantizer
            //   2. Computes the query's (lowerInterval, upperInterval, additionalCorrection,
            //      quantizedComponentSum) so the per-doc 1-bit code is enough to compute a
            //      Lucene-normalized similarity score
            //   3. Returns a VectorScorer whose .iterator() walks docs in order and whose
            //      .score() returns the ADC score for the current doc.
            // This is the exact code path the radial-search KnnCollector consumes — so the
            // error we measure here is the error the collector compares against min_score.
            final FloatVectorValues scorerValues = values.copy();
            final VectorScorer scorer = scorerValues.scorer(v);
            if (scorer == null) {
                continue;
            }
            // Advance the scorer to v's own docId. If for some reason the iterator can't
            // reach it (deleted doc, sparse segment artifact), skip — recording would be noise.
            final DocIdSetIterator si = scorer.iterator();
            final int advanced = si.advance(doc);
            if (advanced != doc) {
                continue;
            }

            // s_quant: ADC score of v against (the stored 1-bit code for) v.
            final float sQuant = scorer.score();
            // s_full: exact similarity. For COSINE/L2 this is the constant 1.0; for
            // (MAXIMUM_)INNER_PRODUCT it's the per-vector self-inner-product after Lucene's
            // normalization. similarity.compare honors the same normalization the scorer used,
            // so s_full and s_quant are directly comparable.
            final float sFull = similarity.compare(v, v);

            // Accumulator records signed relative error and updates histogram + Welford +
            // (when segAcc owns a reservoir) reservoir sampling. Per-pair work is O(1).
            acc.record(sFull, sQuant);
        }
        return acc;
    }

    /* =========================================================================================
     * Option B — Data-as-queries (realistic-distribution score-error probe).
     *
     * Motivation: Option A only tells us "how badly does quantization corrupt the score of v
     * against itself?". Real radial-search queries aren't identical to stored docs — they're
     * neighbors, sometimes far neighbors. The score errors we actually need to slack against
     * happen on (q, v) pairs where q != v. We don't have a query set, so we manufacture one
     * from the data: a random sample of stored docs is reused as proxy queries.
     *
     * Why this is defensible: for cohere/text-embedding-style indexes, queries and docs come
     * from the same embedding model and live on roughly the same distribution. The error
     * distribution under "doc-as-query" closely tracks the error distribution under real
     * queries — same logic that lets ANN-recall evals use held-out docs as queries when no
     * query set is available. It is *not* tight for asymmetric models (e.g. encoder/decoder
     * pairs where query and passage embeddings have different statistics), but for the
     * 1-bit-SQ MIPS / L2 cases we care about, it's the right baseline.
     *
     * What we score per (q, v) pair:
     *
     *   q       = full-precision vector of a randomly-chosen "proxy query" doc
     *   v       = full-precision vector of some other doc in the same segment
     *
     *   s_quant = scorer.score()  for scorer = floatVectorValues.scorer(q)
     *             - 4-bit-quantizes q via OptimizedScalarQuantizer
     *             - reads v's stored 1-bit code + correction factors
     *             - runs the ADC distance + Lucene score normalization
     *             - identical code path to what the radial collector receives
     *
     *   s_full  = similarity.compare(q, v)
     *             - exact float-precision similarity, same normalization as s_quant
     *
     *   rel_err = (s_full - s_quant) / s_full
     *
     * Interpretation: this is the distribution we actually want to set our slack to.
     * If at the 99th percentile rel_err is +3%, then setting min_score' = min_score * (1 - 0.03)
     * means we are 99% likely to *not* drop a doc whose true score is above min_score because
     * its quantized score sank below. Plot histogram + CDF and pick the percentile that matches
     * your recall target.
     *
     * Sampling:
     *   - Per segment, sample M = min(OPTION_B_QUERY_CAP, ceil(0.20 * size)) proxy queries.
     *     20% gives a representative slice without exploding pair count; the hard cap keeps
     *     a 10M-doc segment from blowing up into 2M proxies (≈ 2T pairs).
     *   - Each proxy query scores against EVERY doc in the segment (size pairs per query).
     *     Total pairs per segment ≈ M * size, e.g. 2000 * 1M = 2B pairs — heavy but bounded.
     *     Run time on this is dominated by ADC scoring; the pool size you choose drives wall time.
     *   - The RNG seed is (SEED ^ shardId<<32 ^ segIdx) so two runs with the same data give
     *     the same proxy ords — results are reproducible without locking the RNG across threads.
     *
     * Parallelism:
     *   - One task per proxy query. Each task owns its own FloatVectorValues.copy() pair
     *     (one cursor for the ADC scorer's iterator, one for reading the full-precision v)
     *     and its own Accumulator. Workers never touch shared state during the hot loop.
     *   - Submitted to the *same* pool as Option A via CompletionService so the outer thread
     *     drains as workers finish. The outer thread doing real work (merging) is what makes
     *     this deadlock-safe even when the pool is saturated by other shards' tasks.
     * =========================================================================================
     */

    private static void runOptionB(
        LeafReaderContext leaf,
        String vectorField,
        VectorSimilarityFunction similarity,
        float scoreBandLow,
        int shardId,
        int segIdx,
        ExecutorService pool,
        Accumulator segAcc
    ) throws Exception {
        // template is read-only on the outer thread (for size() and to materialize query
        // vectors). It's never handed to workers.
        final FloatVectorValues template = leaf.reader().getFloatVectorValues(vectorField);
        if (template == null) {
            return;
        }
        final int size = template.size();
        if (size <= 1) {
            // Segment with 0 or 1 docs — no meaningful pairs to score.
            return;
        }
        // M = min(cap, ceil(rate * size)). ceil() so tiny segments still get at least one query.
        final int sampled = Math.min(OPTION_B_QUERY_CAP, (int) Math.ceil(size * OPTION_B_QUERY_SAMPLE_RATE));
        if (sampled <= 0) {
            return;
        }

        // Floyd's algorithm picks M distinct ords from [0, size). Seed mixes shardId and
        // segIdx so picks are deterministic across runs but distinct across segments.
        final int[] queryOrds = sampleOrds(size, sampled, SEED ^ ((long) shardId << 32) ^ segIdx);

        // Pre-materialize query vectors on the main thread, BEFORE submitting workers.
        // Two reasons:
        //   1. FloatVectorValues.vectorValue() returns an internal buffer that is overwritten
        //      on the next call. .clone() snapshots the vector so worker threads see stable q.
        //   2. We do not want to share the `template` cursor across workers.
        final float[][] queryVectors = new float[queryOrds.length][];
        for (int i = 0; i < queryOrds.length; i++) {
            queryVectors[i] = template.vectorValue(queryOrds[i]).clone();
        }

        // Fan out: one task per proxy query. Each task is independent — no synchronization
        // is required inside scoreOneQuery. The outer thread merges results in completion
        // order, which means a long-tail slow query won't block faster queries from being
        // folded into segAcc.
        final CompletionService<Accumulator> cs = new ExecutorCompletionService<>(pool);
        final List<Future<Accumulator>> futures = new ArrayList<>(queryOrds.length);
        for (int i = 0; i < queryOrds.length; i++) {
            final float[] q = queryVectors[i];
            futures.add(cs.submit(() -> scoreOneQuery(leaf, vectorField, similarity, scoreBandLow, q)));
        }

        try {
            for (int i = 0; i < queryOrds.length; i++) {
                final Accumulator part = cs.take().get();
                segAcc.merge(part);
            }
        } catch (Exception e) {
            // A worker threw — cancel the rest to free pool slots immediately.
            for (final Future<Accumulator> f : futures) {
                f.cancel(true);
            }
            throw e;
        }
    }

    /**
     * Score one proxy query against EVERY doc in the segment. Two parallel cursors:
     *   - scorerValues: feeds the ADC scorer (built once per query, internally quantizes q).
     *   - fullValues:   used to read each doc's full-precision vector for the exact comparison.
     * Each cursor must be its own FloatVectorValues.copy() because the underlying IndexInput
     * is stateful and FloatVectorValues is not thread-safe.
     *
     * Cost: O(segment_size) ADC scores + O(segment_size) exact similarities + O(segment_size)
     * float-vector reads. For a 1M segment that's ~3M index-input reads and the dominant cost.
     */
    private static Accumulator scoreOneQuery(
        LeafReaderContext leaf,
        String vectorField,
        VectorSimilarityFunction similarity,
        float scoreBandLow,
        float[] query
    ) throws IOException {
        // Per-task reservoir is required for the segment-level merge to receive samples
        // (Accumulator.merge feeds the segment reservoir from other.reservoir, not from
        // raw record() calls).
        final Accumulator acc = Accumulator.create("B", WORKER_RESERVOIR_SIZE);
        // scorer() builds and caches per-query state (4-bit quantization of `query`,
        // correction terms, centroid dot product) inside the returned scorer. We pay that
        // ONCE per query, then amortize it across every doc in the segment.
        final FloatVectorValues scorerValues = leaf.reader().getFloatVectorValues(vectorField).copy();
        final FloatVectorValues fullValues = leaf.reader().getFloatVectorValues(vectorField).copy();
        final VectorScorer scorer = scorerValues.scorer(query);
        if (scorer == null) {
            return acc;
        }
        // Walk every doc the scorer can see. scorer.iterator() returns a DocIdSetIterator
        // over docIds (not ords); .score() returns the ADC score for the *current* doc.
        final DocIdSetIterator si = scorer.iterator();
        for (int doc = si.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = si.nextDoc()) {
            // s_quant: ADC score of (4-bit-quantized query) vs. (1-bit code + corrections of doc).
            // Identical numeric path that the radial-search KnnCollector receives.
            final float sQuant = scorer.score();

            // s_full: same docId, but read the doc's float vector and compute the exact
            // similarity. We advance a *separate* cursor — we cannot reuse the scorer's
            // iterator because the scorer's internal cursor must stay on `doc` for .score().
            final KnnVectorValues.DocIndexIterator fi = fullValues.iterator();
            final int advanced = fi.advance(doc);
            if (advanced != doc) {
                // Iterator couldn't reach the doc — likely a deleted-doc skip. Skip recording.
                continue;
            }
            final float[] v = fullValues.vectorValue(fi.index());
            final float sFull = similarity.compare(query, v);

            // Population filter: only pairs in the radial collector's operating band contribute
            // to the slack estimate. Without this, the heavy tail of far-pair noise (where
            // s_full is tiny and rel_err blows up) dominates the percentiles we read out.
            if (sFull < scoreBandLow) {
                continue;
            }

            // Record the signed relative score gap. Per-pair work is O(1) into the
            // accumulator's histogram + Welford counters.
            acc.record(sFull, sQuant);
        }
        return acc;
    }


    /* =========================================================================================
     * Sampling
     * =========================================================================================
     */

    /**
     * Floyd's algorithm — sample {@code k} distinct ords from {@code [0, n)} deterministically.
     */
    private static int[] sampleOrds(int n, int k, long seed) {
        if (k >= n) {
            final int[] all = new int[n];
            for (int i = 0; i < n; i++) all[i] = i;
            return all;
        }
        final SplittableRandom rng = new SplittableRandom(seed);
        final java.util.HashSet<Integer> picked = new java.util.HashSet<>(k * 2);
        for (int j = n - k; j < n; j++) {
            final int t = rng.nextInt(j + 1);
            if (!picked.add(t)) {
                picked.add(j);
            }
        }
        final int[] out = new int[picked.size()];
        int idx = 0;
        for (final int v : picked) out[idx++] = v;
        return out;
    }

    /* =========================================================================================
     * Accumulator — histogram + Welford + bounded reservoir for percentiles.
     * Merge-safe: every operation operates on local state and merges happen single-threaded.
     * =========================================================================================
     */

    static final class Accumulator {
        final String label;
        long count;
        double mean;
        double m2;
        double minErr = Double.POSITIVE_INFINITY;
        double maxErr = Double.NEGATIVE_INFINITY;
        final long[] hist = new long[HIST_TOTAL_BINS];
        final double[] reservoir;
        long seenForReservoir;
        final SplittableRandom rng;

        private Accumulator(String label, int reservoirSize) {
            this.label = label;
            this.reservoir = reservoirSize > 0 ? new double[reservoirSize] : new double[0];
            // Per-accumulator RNG with a distinct stream — merge() folds reservoirs deterministically.
            this.rng = new SplittableRandom(SEED ^ label.hashCode());
        }

        static Accumulator create(String label, int reservoirSize) {
            return new Accumulator(label, reservoirSize);
        }

        void record(float sFull, float sQuant) {
            if (Float.isNaN(sFull) || Float.isNaN(sQuant)) return;
            if (Math.abs(sFull) < SCORE_DENOM_FLOOR) return;
            final double err = ((double) sFull - sQuant) / sFull;
            count++;
            // Welford
            final double delta = err - mean;
            mean += delta / count;
            m2 += delta * (err - mean);
            if (err < minErr) minErr = err;
            if (err > maxErr) maxErr = err;
            hist[binFor(err)]++;
            // Reservoir sampling
            if (reservoir.length > 0) {
                seenForReservoir++;
                if (seenForReservoir <= reservoir.length) {
                    reservoir[(int) (seenForReservoir - 1)] = err;
                } else {
                    final long j = (rng.nextLong() & Long.MAX_VALUE) % seenForReservoir;
                    if (j < reservoir.length) {
                        reservoir[(int) j] = err;
                    }
                }
            }
        }

        void merge(Accumulator other) {
            if (other.count == 0) return;
            if (this.count == 0) {
                this.count = other.count;
                this.mean = other.mean;
                this.m2 = other.m2;
            } else {
                final long n1 = this.count, n2 = other.count;
                final double delta = other.mean - this.mean;
                final long n = n1 + n2;
                this.mean += delta * n2 / n;
                this.m2 += other.m2 + delta * delta * n1 * n2 / n;
                this.count = n;
            }
            this.minErr = Math.min(this.minErr, other.minErr);
            this.maxErr = Math.max(this.maxErr, other.maxErr);
            for (int i = 0; i < hist.length; i++) hist[i] += other.hist[i];

            // Reservoir merge: treat other's sampled entries as another stream of records.
            if (reservoir.length > 0 && other.reservoir.length > 0) {
                final int otherFill = (int) Math.min(other.seenForReservoir, other.reservoir.length);
                for (int i = 0; i < otherFill; i++) {
                    seenForReservoir++;
                    if (seenForReservoir <= reservoir.length) {
                        reservoir[(int) (seenForReservoir - 1)] = other.reservoir[i];
                    } else {
                        final long j = (rng.nextLong() & Long.MAX_VALUE) % seenForReservoir;
                        if (j < reservoir.length) {
                            reservoir[(int) j] = other.reservoir[i];
                        }
                    }
                }
            }
        }

        long count() {
            return count;
        }

        double stddev() {
            return count > 1 ? Math.sqrt(m2 / (count - 1)) : 0.0;
        }

        double[] percentiles(double[] pcts) {
            final int fill = (int) Math.min(seenForReservoir, reservoir.length);
            final double[] out = new double[pcts.length];
            if (fill == 0) {
                Arrays.fill(out, Double.NaN);
                return out;
            }
            final double[] sorted = Arrays.copyOf(reservoir, fill);
            Arrays.sort(sorted);
            for (int i = 0; i < pcts.length; i++) {
                final double p = Math.min(1.0, Math.max(0.0, pcts[i]));
                final int idx = (int) Math.min(sorted.length - 1, Math.round(p * (sorted.length - 1)));
                out[i] = sorted[idx];
            }
            return out;
        }

        /**
         * Percentiles over the positive tail of rel_err (clamped at 0). These are the
         * percentiles that map directly to the slack {@code min_score' = min_score * (1 - eps)}:
         * the failure mode it prevents is {@code s_quant < s_full} (positive rel_err).
         * Negative rel_err means quantization over-scored the pair, which causes false
         * accepts — that's the rescore pass's job, not slack's.
         *
         * <p>Computed by treating the reservoir's negative entries as zero before taking
         * the quantile. Result: percentiles_pos[p] = {@code Q_p( max(0, rel_err) )}.
         */
        double[] positiveTailPercentiles(double[] pcts) {
            final int fill = (int) Math.min(seenForReservoir, reservoir.length);
            final double[] out = new double[pcts.length];
            if (fill == 0) {
                Arrays.fill(out, Double.NaN);
                return out;
            }
            final double[] positive = new double[fill];
            for (int i = 0; i < fill; i++) {
                positive[i] = Math.max(0.0, reservoir[i]);
            }
            Arrays.sort(positive);
            for (int i = 0; i < pcts.length; i++) {
                final double p = Math.min(1.0, Math.max(0.0, pcts[i]));
                final int idx = (int) Math.min(positive.length - 1, Math.round(p * (positive.length - 1)));
                out[i] = positive[idx];
            }
            return out;
        }
    }

    private static int binFor(double err) {
        if (err <= -HIST_RANGE) return 0;
        if (err > HIST_RANGE) return HIST_TOTAL_BINS - 1;
        final int idx = 1 + (int) Math.floor((err + HIST_RANGE) / HIST_BIN_WIDTH);
        return Math.min(HIST_TOTAL_BINS - 2, Math.max(1, idx));
    }

    private static String binLabel(int bin) {
        if (bin == 0) {
            return String.format(Locale.ROOT, "(-inf, %.1f%%]", -HIST_RANGE * 100);
        }
        if (bin == HIST_TOTAL_BINS - 1) {
            return String.format(Locale.ROOT, "(%.1f%%, +inf)", HIST_RANGE * 100);
        }
        final double lo = -HIST_RANGE + (bin - 1) * HIST_BIN_WIDTH;
        final double hi = lo + HIST_BIN_WIDTH;
        return String.format(Locale.ROOT, "(%.2f%%, %.2f%%]", lo * 100, hi * 100);
    }

    /* =========================================================================================
     * Logging helpers
     * =========================================================================================
     */

    private static void logAccumulator(String label, Accumulator acc) {
        if (acc.count == 0) {
            log.info("  [{}] no samples", label);
            return;
        }
        final double[] qs = new double[] { 0.50, 0.90, 0.95, 0.99, 0.999, 1.0 };
        final double[] pcts = acc.percentiles(qs);
        final double[] pctsPos = acc.positiveTailPercentiles(qs);
        log.info(
            "  [{}] n={} mean={} stddev={} min={} max={}",
            label,
            acc.count(),
            fmtPct(acc.mean),
            fmtPct(acc.stddev()),
            fmtPct(acc.minErr),
            fmtPct(acc.maxErr)
        );
        log.info(
            "  [{}] two-sided percentiles : p50={} p90={} p95={} p99={} p99.9={} pmax={}",
            label,
            fmtPct(pcts[0]),
            fmtPct(pcts[1]),
            fmtPct(pcts[2]),
            fmtPct(pcts[3]),
            fmtPct(pcts[4]),
            fmtPct(pcts[5])
        );
        // Slack-relevant view: only the positive tail (rel_err > 0 = quantization undershot).
        // The 99th of this column is the eps you would use in min_score' = min_score * (1 - eps).
        log.info(
            "  [{}] positive-tail (SLACK) : p50={} p90={} p95={} p99={} p99.9={} pmax={}",
            label,
            fmtPct(pctsPos[0]),
            fmtPct(pctsPos[1]),
            fmtPct(pctsPos[2]),
            fmtPct(pctsPos[3]),
            fmtPct(pctsPos[4]),
            fmtPct(pctsPos[5])
        );
    }

    /**
     * One-line slack readout. Prints the 95th percentile of the positive tail of rel_err
     * in plain English — e.g. "95% of errors are expected to be <= 4.000%". This is the
     * eps you would plug into {@code min_score' = min_score * (1 - eps)} to keep ~95% of
     * border-case docs from being wrongly rejected. Also prints p99 / p99.9 alongside so
     * you can pick a tighter tolerance if your recall target is stricter.
     */
    private static void logHeadline(String label, Accumulator acc) {
        if (acc.count == 0) {
            log.info("  [{}] no samples", label);
            return;
        }
        final double[] qs = new double[] { 0.95, 0.99, 0.999 };
        final double[] pos = acc.positiveTailPercentiles(qs);
        log.info("  [{}] 95% of errors are expected to be <= {}", label, fmtPct(pos[0]));
        log.info("  [{}] 99% of errors are expected to be <= {}", label, fmtPct(pos[1]));
        log.info("  [{}] 99.9% of errors are expected to be <= {}", label, fmtPct(pos[2]));
    }

    private static void logHistogram(String label, Accumulator acc) {
        log.info("---------- {} (bin,count,pct) ----------", label);
        if (acc.count == 0) {
            log.info("  no samples");
            return;
        }
        for (int b = 0; b < HIST_TOTAL_BINS; b++) {
            if (acc.hist[b] == 0) continue;
            final double pct = 100.0 * acc.hist[b] / acc.count;
            log.info("  {} | {} | {}", binLabel(b), acc.hist[b], String.format(Locale.ROOT, "%.3f%%", pct));
        }
    }

    /**
     * CSV dump of the histogram — one row per non-empty bin, comma-separated, trailing comma.
     * Column layout:
     *   bin_lo_pct , bin_hi_pct , count , pct_of_total ,
     * The leading `CSV,` prefix on every line lets you filter these out of a full run log with:
     *   grep '^CSV,' run.log | cut -d, -f2-
     */
    private static void logHistogramCsv(String label, Accumulator acc) {
        log.info("---------- CSV dump: {} ----------", label);
        // Header row.
        log.info("CSV,bin_lo_pct,bin_hi_pct,count,pct_of_total,");
        if (acc.count == 0) {
            return;
        }
        for (int b = 0; b < HIST_TOTAL_BINS; b++) {
            if (acc.hist[b] == 0) continue;
            final double loPct;
            final double hiPct;
            if (b == 0) {
                // Underflow bin (-inf, -HIST_RANGE].
                loPct = Double.NEGATIVE_INFINITY;
                hiPct = -HIST_RANGE * 100;
            } else if (b == HIST_TOTAL_BINS - 1) {
                // Overflow bin (+HIST_RANGE, +inf).
                loPct = HIST_RANGE * 100;
                hiPct = Double.POSITIVE_INFINITY;
            } else {
                final double lo = -HIST_RANGE + (b - 1) * HIST_BIN_WIDTH;
                loPct = lo * 100;
                hiPct = (lo + HIST_BIN_WIDTH) * 100;
            }
            final double pct = 100.0 * acc.hist[b] / acc.count;
            log.info(
                String.format(
                    Locale.ROOT,
                    "CSV,%s,%s,%d,%.6f,",
                    fmtCsvBound(loPct),
                    fmtCsvBound(hiPct),
                    acc.hist[b],
                    pct
                )
            );
        }
    }

    private static String fmtCsvBound(double v) {
        if (Double.isInfinite(v)) {
            return v > 0 ? "+inf" : "-inf";
        }
        return String.format(Locale.ROOT, "%.3f", v);
    }

    private static String fmtPct(double v) {
        if (Double.isNaN(v) || Double.isInfinite(v)) return String.valueOf(v);
        return String.format(Locale.ROOT, "%+.3f%%", v * 100);
    }

    /* =========================================================================================
     * File system / env
     * =========================================================================================
     */

    private static List<Path> listShardDirs(Path rootDir) throws IOException {
        final List<Path> shards = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(rootDir)) {
            for (final Path p : stream) {
                if (Files.isDirectory(p) == false) {
                    continue;
                }
                final String name = p.getFileName().toString();
                if (name.chars().allMatch(Character::isDigit)) {
                    shards.add(p);
                }
            }
        }
        shards.sort(Comparator.comparingInt(ErrorDistributionMainTests::shardIndex));
        return shards;
    }

    private static int shardIndex(Path shard) {
        return Integer.parseInt(shard.getFileName().toString());
    }

    private static String requireEnv(String name) {
        final String v = System.getenv(name);
        if (v == null || v.isBlank()) {
            throw new IllegalStateException("Required environment variable is missing: " + name);
        }
        return v;
    }

    private static int parseIntEnv(String name, int defaultValue) {
        final String v = System.getenv(name);
        if (v == null || v.isBlank()) {
            return defaultValue;
        }
        try {
            final int parsed = Integer.parseInt(v.trim());
            if (parsed < 1) {
                throw new IllegalStateException(name + " must be >= 1, got: " + parsed);
            }
            return parsed;
        } catch (NumberFormatException e) {
            throw new IllegalStateException(name + " must be an integer, got: " + v, e);
        }
    }

    private static float parseFloatEnv(String name, float defaultValue) {
        final String v = System.getenv(name);
        if (v == null || v.isBlank()) {
            return defaultValue;
        }
        try {
            return Float.parseFloat(v.trim());
        } catch (NumberFormatException e) {
            throw new IllegalStateException(name + " must be a float, got: " + v, e);
        }
    }

    private static void logParameters(
        Path dataDir,
        String vectorField,
        SpaceType spaceType,
        int numCores,
        float scoreBandLow
    ) {
        log.info("=== ErrorDistributionMain parameters ===");
        log.info("  DATA_DIR                  = {}", dataDir);
        log.info("  VECTOR_FIELD              = {}", vectorField);
        log.info("  SPACE_TYPE                = {}", spaceType.getValue());
        log.info("  similarityFunction        = {}", spaceType.getKnnVectorSimilarityFunction().getVectorSimilarityFunction());
        log.info("  optionB.querySampleRate   = {}", OPTION_B_QUERY_SAMPLE_RATE);
        log.info("  optionB.queryCap          = {}", OPTION_B_QUERY_CAP);
        log.info("  numCores                  = {}", numCores);
        log.info(
            "  scoreBandLow              = {} (Option B records only pairs with s_full >= this)",
            scoreBandLow == Float.NEGATIVE_INFINITY ? "unset (no filter)" : String.valueOf(scoreBandLow)
        );
        log.info("  reservoirSize             = {}", RESERVOIR_SIZE);
        log.info("  histogram.range           = ±{}%", HIST_RANGE * 100);
        log.info("  histogram.binWidth        = {}%", HIST_BIN_WIDTH * 100);
        log.info("  histogram.totalBins       = {}", HIST_TOTAL_BINS);
    }

    // Silence the unused-import warning for Callable (kept for readability of the CompletionService block).
    @SuppressWarnings("unused")
    private static Callable<Void> unusedCallableHook() {
        return () -> null;
    }
}
