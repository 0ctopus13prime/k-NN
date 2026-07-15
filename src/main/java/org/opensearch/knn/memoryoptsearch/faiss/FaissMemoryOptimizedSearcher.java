/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import com.google.common.annotations.VisibleForTesting;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.KnnVectorValues.DocIndexIterator;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.OrdinalTranslatedKnnCollector;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.common.RobustUniqueRandomIterator;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.query.memoryoptsearch.RadiusVectorSimilarityCollector;
import org.opensearch.knn.index.query.memoryoptsearch.RadiusVectorSimilarityCollector1;
import org.opensearch.knn.index.util.WarmupUtil;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;
import org.opensearch.knn.memoryoptsearch.faiss.cagra.FaissCagraHNSW;

import java.io.IOException;
import java.util.Arrays;
import java.util.SplittableRandom;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.EXHAUSTIVE_BULK_SCORE_ORDS;

/**
 * This searcher directly reads FAISS index file via the provided {@link IndexInput} then perform vector search on it.
 */
@Log4j2
public class FaissMemoryOptimizedSearcher implements VectorSearcher {
    /** Max number of proxy-query pairs to sample when estimating quantization slack. */
    private static final int EPS_SAMPLE_SIZE = 10_000;
    /** Percentile of positive-tail rel_err used as the slack. p99 keeps 99% of border docs from being wrongly rejected. */
    private static final double EPS_PERCENTILE = 0.95;
    /** RNG seed for reproducible proxy-query sampling. */
    private static final long EPS_SEED = 42L;
    /** Fixed slack when we can't measure — 0 = behave identically to the pre-slack code path. */
    private static final float EPS_FALLBACK = 0f;

    private final IndexInput indexInput;
    private final FaissIndex faissIndex;
    private final FlatVectorsScorer flatVectorsScorer;
    private final FaissHNSW hnsw;
    private final VectorSimilarityFunction vectorSimilarityFunction;
    private boolean isAdc;
    /**
     * Segment-level quantization slack, computed once in the constructor by sampling proxy queries
     * from the corpus and measuring rel_err = (s_full - s_quant) / s_full at each pair. Applied to
     * {@link org.opensearch.knn.index.query.memoryoptsearch.RadiusVectorSimilarityCollector} at
     * search time so borderline docs (whose ADC score dipped below min_score) are still accepted.
     */
    private final float eps;

    /**
     * Constructor that accepts a pre-loaded {@link FaissIndex}. The factory is responsible for
     * loading the index and applying any transformations (e.g., replacing null flat storage for Faiss SQ (for 1 bit)).
     */
    public FaissMemoryOptimizedSearcher(
        final IndexInput indexInput,
        final FaissIndex faissIndex,
        final FieldInfo fieldInfo,
        final FlatVectorsScorer flatVectorsScorer
    ) {
        this.indexInput = indexInput;
        this.faissIndex = faissIndex;
        final KNNVectorSimilarityFunction knnVectorSimilarityFunction = faissIndex.getVectorSimilarityFunction();

        if (knnVectorSimilarityFunction != KNNVectorSimilarityFunction.HAMMING) {
            vectorSimilarityFunction = knnVectorSimilarityFunction.getVectorSimilarityFunction();
        } else {
            vectorSimilarityFunction = null;
        }

        this.isAdc = FieldInfoExtractor.isAdc(fieldInfo);
        this.flatVectorsScorer = flatVectorsScorer;
        this.hnsw = extractFaissHnsw(faissIndex);
        this.eps = estimateEps();
        System.out.println("********************************** EPS=" + this.eps + ", II=[" + indexInput + "]");
    }

    private static FaissHNSW extractFaissHnsw(final FaissIndex faissIndex) {
        if (faissIndex instanceof FaissIdMapIndex idMapIndex) {
            return idMapIndex.getFaissHnsw();
        }

        throw new IllegalArgumentException("Faiss index [" + faissIndex.getIndexType() + "] does not have HNSW as an index.");
    }

    /**
     * Compute the segment-level quantization slack {@code eps} once, in the constructor.
     *
     * <p>Method:
     * <ol>
     *   <li>Pick {@code N = min(size, EPS_SAMPLE_SIZE)} random ordinals as proxy queries.</li>
     *   <li>For each proxy query {@code q}, pick one random target ord {@code v} (never equal to {@code q}).</li>
     *   <li>Compute {@code s_full = similarity.compare(q_float, v_float)} using the same
     *       {@link VectorSimilarityFunction} the real search uses.</li>
     *   <li>Compute {@code s_quant = randomVectorScorer(q_float).score(v)} — the exact ADC path
     *       the real search's collector consumes.</li>
     *   <li>Record positive-tail {@code rel_err = max(0, (s_full - s_quant) / s_full)}, skipping
     *       pairs with near-zero {@code s_full}.</li>
     *   <li>Return the {@code EPS_PERCENTILE}-th percentile of the collected rel_err values.</li>
     * </ol>
     *
     * <p>Runs sequentially in the ctor; ~10K scoring calls per segment finishes in well under
     * a second on typical hardware. Skipped (fallback = 0) only for the {@code isAdc=true} path
     * (byte-encoded queries don't have a proxy-query story yet) or for degenerate segments with
     * fewer than 2 vectors. Tiny corpora just get a noisier estimate — same code path, same
     * scoring, just fewer unique pairs contributing to the percentile.
     *
     * @return positive slack in {@code [0, 1)}; {@code 0} disables widening.
     */
    private float estimateEps() {
        // POC scope: only wire up eps estimation for the float-query path. For byte-encoded
        // fields the "sample a doc as a query" story is different and out of scope for now.
        if (isAdc) {
            log.info("[FaissMemoryOptimizedSearcher] isAdc=true — skipping eps estimation, using fallback eps={}", EPS_FALLBACK);
            return EPS_FALLBACK;
        }

        try {
            final int size = Math.toIntExact(faissIndex.getTotalNumberOfVectors());
            // No small-segment guard: for tiny corpora the estimate is noisier (many duplicate
            // pairs across the loop) but still reflects the actual quantization error the
            // collector will see. Prefer a noisy real number over a hardcoded fallback.
            if (size < 2) {
                // Can't sample a distinct (q, v) pair — bail out to the fallback.
                log.info("[FaissMemoryOptimizedSearcher] segment size {} < 2 — using fallback eps={}", size, EPS_FALLBACK);
                return EPS_FALLBACK;
            }

            final int sampleSize = Math.min(size, EPS_SAMPLE_SIZE);
            final SplittableRandom rng = new SplittableRandom(EPS_SEED);

            // Two cursors hoisted outside the loop:
            // floatValues — reads q_float and v_float for the exact-similarity computation.
            // scorerValues — feeds the ADC scorer factory. Its own cursor so it doesn't
            // race with `floatValues` when we read v_float after scoring.
            // FloatVectorValues is not thread-safe, but the ctor runs on a single thread.
            final FloatVectorValues floatValues = faissIndex.getFloatValues(indexInput.clone());
            final KnnVectorValues scorerValues = faissIndex.getFloatValues(indexInput.clone());

            final float[] positiveRelErrs = new float[sampleSize];
            int recorded = 0;

            for (int i = 0; i < sampleSize; i++) {
                final int qOrd = rng.nextInt(size);
                // Pick a target that is not the query itself. Self-pairs are Option A territory —
                // they hit the max s_full and don't reflect real query-doc dynamics.
                int vOrd;
                do {
                    vOrd = rng.nextInt(size);
                } while (vOrd == qOrd);

                // Snapshot the query vector before we advance the cursor to the target.
                // vectorValue() returns an internal buffer that gets overwritten on next call.
                final float[] qFloat = floatValues.vectorValue(qOrd).clone();

                // The scorer bakes the query vector in (4-bit-quantizes qFloat and caches the
                // correction terms), so we DO need a fresh one per query. The doc-side view
                // (`scorerValues`) is stateless in this respect and reused.
                final RandomVectorScorer scorer = flatVectorsScorer.getRandomVectorScorer(vectorSimilarityFunction, scorerValues, qFloat);
                final float sQuant = scorer.score(vOrd);

                final float[] vFloat = floatValues.vectorValue(vOrd);
                final float sFull = vectorSimilarityFunction.compare(qFloat, vFloat);

                // Skip degenerate denominators — relative error is undefined near zero.
                if (Float.isNaN(sFull) || Float.isNaN(sQuant) || Math.abs(sFull) < 1e-6f) {
                    continue;
                }

                final float relErr = (sFull - sQuant) / sFull;
                // Positive tail only: rel_err > 0 means quantization undershot, which is what
                // the slack is designed to correct. Negative rel_err (quant over-scored) is
                // the false-accept problem that the caller's rescore pass handles.
                if (relErr > 0f) {
                    positiveRelErrs[recorded++] = relErr;
                }
            }

            if (recorded == 0) {
                log.warn("[FaissMemoryOptimizedSearcher] no positive rel_err samples collected — using fallback eps={}", EPS_FALLBACK);
                return EPS_FALLBACK;
            }

            // Percentile via sort + index. For 10K samples this is trivial.
            final float[] sorted = Arrays.copyOf(positiveRelErrs, recorded);
            Arrays.sort(sorted);
            final int idx = Math.min(sorted.length - 1, (int) Math.round(EPS_PERCENTILE * (sorted.length - 1)));
            final float estimated = sorted[idx];

            // Clamp to [0, 1) — the collector rejects values outside this range.
            final float clamped = Math.max(0f, Math.min(0.99f, estimated));
            log.info(
                "[FaissMemoryOptimizedSearcher] eps estimated: size={} sampled={} positive={} p{}={} clamped={}",
                size,
                sampleSize,
                recorded,
                (int) (EPS_PERCENTILE * 100),
                estimated,
                clamped
            );
            return clamped;
        } catch (Exception e) {
            log.warn("[FaissMemoryOptimizedSearcher] eps estimation failed ({}), using fallback eps={}", e.toString(), EPS_FALLBACK);
            return EPS_FALLBACK;
        }
    }

    /**
     * @return the segment-level quantization slack computed at construction time. Applied to
     * radial collectors via {@code RadiusVectorSimilarityCollector.applySlack(eps)}.
     */
    public float getEps() {
        return eps;
    }

    @Override
    public void search(float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        applyEpsIfRadial(knnCollector);
        final KnnVectorValues knnVectorValues = isAdc
            ? faissIndex.getByteValues(indexInput.clone())
            : faissIndex.getFloatValues(indexInput.clone());

        search(
            VectorEncoding.FLOAT32,
            flatVectorsScorer.getRandomVectorScorer(vectorSimilarityFunction, knnVectorValues, target),
            knnCollector,
            acceptDocs
        );
    }

    @Override
    public void search(byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        applyEpsIfRadial(knnCollector);
        search(
            VectorEncoding.BYTE,
            flatVectorsScorer.getRandomVectorScorer(vectorSimilarityFunction, faissIndex.getByteValues(indexInput.clone()), target),
            knnCollector,
            acceptDocs
        );
    }

    /**
     * If the caller-supplied collector is a {@link RadiusVectorSimilarityCollector}, widen its
     * accept net by the segment's estimated quantization slack. No-op for other collector types
     * (top-k, etc.) and no-op when {@code eps == 0}.
     *
     * <p>Deliberately runs on the raw {@code knnCollector} parameter before any wrapping happens
     * in {@link #createKnnCollector(KnnCollector, RandomVectorScorer)} — the wrapper decorators
     * are not radial collectors themselves.
     */
    private void applyEpsIfRadial(final KnnCollector knnCollector) {
        if (eps <= 0f) {
            return;
        }
        // if (knnCollector instanceof RadiusVectorSimilarityCollector radialKnnCollector) {
        //     radialKnnCollector.applySlack(eps);
        // }
        if (knnCollector instanceof RadiusVectorSimilarityCollector1 radialKnnCollector) {
            radialKnnCollector.applySlack(eps);
        }
    }

    /**
     * Returns a {@link FaissScorableByteVectorValues} that wraps the raw byte vectors from the
     * FAISS index with scoring support via {@link FlatVectorsScorer}.
     * <p>Each call creates a new instance backed by a fresh index input slice.
     */
    @Override
    public ByteVectorValues getByteVectorValues(DocIndexIterator iterator) throws IOException {
        return new FaissScorableByteVectorValues(
            faissIndex.getByteValues(indexInput.clone()),
            flatVectorsScorer,
            vectorSimilarityFunction,
            iterator
        );
    }

    @Override
    public void warmUp() throws IOException {
        // Warm up graph
        final IndexInput warmUpIndexInput = indexInput.clone();
        WarmupUtil.readAll(warmUpIndexInput);

        // Warm up flat vectors
        // This can warm up .veb, .vec or .faiss
        if (faissIndex.getVectorEncoding() == VectorEncoding.FLOAT32) {
            WarmupUtil.readAll(faissIndex.getFloatValues(warmUpIndexInput));
        } else if (faissIndex.getVectorEncoding() == VectorEncoding.BYTE) {
            WarmupUtil.readAll(faissIndex.getByteValues(warmUpIndexInput));
        }
    }

    @Override
    public void close() throws IOException {
        indexInput.close();
    }

    private void search(
        final VectorEncoding vectorEncoding,
        final RandomVectorScorer scorer,
        final KnnCollector knnCollector,
        final AcceptDocs acceptDocs
    ) throws IOException {
        if (faissIndex.getTotalNumberOfVectors() == 0 || knnCollector.k() == 0) {
            return;
        }

        if (!this.isAdc && faissIndex.getVectorEncoding() != vectorEncoding) {
            throw new IllegalArgumentException(
                "Search for vector encoding ["
                    + vectorEncoding
                    + "] is not supported in "
                    + "an index vector whose encoding is ["
                    + faissIndex.getVectorEncoding()
                    + "]"
            );
        }

        // Set up required components for vector search
        final KnnCollector collector = createKnnCollector(knnCollector, scorer);
        final Bits acceptedOrds = scorer.getAcceptOrds(acceptDocs.bits());

        if (knnCollector.k() < scorer.maxOrd()) {
            // Do ANN search with Lucene's HNSW graph searcher.
            HnswGraphSearcher.search(scorer, collector, new FaissHnswGraph(hnsw, indexInput.clone()), acceptedOrds);
        } else {
            // if k is larger than the number of vectors we expect to visit in an HNSW search,
            // we can just iterate over all vectors and collect them.
            int numVectors = scorer.maxOrd();
            int[] ords = new int[EXHAUSTIVE_BULK_SCORE_ORDS];
            float[] scores = new float[EXHAUSTIVE_BULK_SCORE_ORDS];
            int numOrds = 0;
            for (int i = 0; i < numVectors; i++) {
                if (acceptedOrds == null || acceptedOrds.get(i)) {
                    if (knnCollector.earlyTerminated()) {
                        break;
                    }
                    ords[numOrds++] = i;
                    if (numOrds == ords.length) {
                        knnCollector.incVisitedCount(numOrds);
                        if (scorer.bulkScore(ords, scores, numOrds) > knnCollector.minCompetitiveSimilarity()) {
                            for (int j = 0; j < numOrds; j++) {
                                knnCollector.collect(scorer.ordToDoc(ords[j]), scores[j]);
                            }
                        }
                        numOrds = 0;
                    }
                }
            }

            if (numOrds > 0) {
                knnCollector.incVisitedCount(numOrds);
                if (scorer.bulkScore(ords, scores, numOrds) > knnCollector.minCompetitiveSimilarity()) {
                    for (int j = 0; j < numOrds; j++) {
                        knnCollector.collect(scorer.ordToDoc(ords[j]), scores[j]);
                    }
                }
            }
        }
    }

    @VisibleForTesting
    KnnCollector createKnnCollector(final KnnCollector knnCollector, final RandomVectorScorer scorer) {
        final KnnCollector ordinalTranslatedKnnCollector = new OrdinalTranslatedKnnCollector(knnCollector, scorer::ordToDoc);

        if (hnsw instanceof FaissCagraHNSW cagraHNSW && (knnCollector.getSearchStrategy() instanceof KnnSearchStrategy.Seeded) == false) {
            // If there are provided entry points, then we should honor it and ensure searching to start based on them instead of
            // search with randomly selected points.
            return new KnnCollector.Decorator(ordinalTranslatedKnnCollector) {
                @Override
                public KnnSearchStrategy getSearchStrategy() {
                    return RandomEntryPointsKnnSearchStrategy.getInstance(
                        cagraHNSW.getNumBaseLevelSearchEntryPoints(),
                        cagraHNSW.getTotalNumberOfVectors(),
                        knnCollector.getSearchStrategy()
                    );
                }
            };
        }

        return ordinalTranslatedKnnCollector;
    }

    /**
     * Knn search strategy having a doc-id-iterator returning random document ids.
     * This is not designed for general purpose, it is particularly designed for populating random document ids for Cagra index.
     * Note that doc-id-iterator returns a random ids in `nextDoc` method without sorting, and might return duplicated ids.
     */
    static class RandomEntryPointsKnnSearchStrategy extends KnnSearchStrategy.Seeded {

        public static RandomEntryPointsKnnSearchStrategy getInstance(
            final int numberOfEntryPoints,
            final long totalNumberOfVectors,
            final KnnSearchStrategy originalStrategy
        ) {

            int entryPoints = getTotalNumberOfEntryPoints(numberOfEntryPoints, Math.toIntExact(totalNumberOfVectors));

            final DocIdSetIterator docIdSetIterator = generateRandomEntryPoints(entryPoints, Math.toIntExact(totalNumberOfVectors));

            return new RandomEntryPointsKnnSearchStrategy(docIdSetIterator, entryPoints, originalStrategy);
        }

        private RandomEntryPointsKnnSearchStrategy(
            final DocIdSetIterator entryPoints,
            final int numberOfEntryPoints,
            final KnnSearchStrategy originalStrategy
        ) {
            super(entryPoints, numberOfEntryPoints, originalStrategy);
        }

        private static int getTotalNumberOfEntryPoints(int numberOfEntryPoints, int totalVectors) {
            return numberOfEntryPoints >= totalVectors ? totalVectors : numberOfEntryPoints;
        }

        private static DocIdSetIterator generateRandomEntryPoints(final int numberOfEntryPoints, int totalNumberOfVectors) {
            if (numberOfEntryPoints >= totalNumberOfVectors) {
                return DocIdSetIterator.all(totalNumberOfVectors);
            }
            return new DocIdSetIterator() {
                final RobustUniqueRandomIterator robustUniqueRandomIterator = new RobustUniqueRandomIterator(
                    totalNumberOfVectors,
                    numberOfEntryPoints
                );

                @Override
                public int docID() {
                    throw new UnsupportedOperationException("DISI in RandomEntryPointsKnnSearchStrategy does not support docID()");
                }

                @Override
                public int nextDoc() {
                    return robustUniqueRandomIterator.next();
                }

                @Override
                public int advance(int targetDoc) {
                    throw new UnsupportedOperationException("DISI in RandomEntryPointsKnnSearchStrategy does not support advance(int)");
                }

                @Override
                public long cost() {
                    throw new UnsupportedOperationException("DISI in RandomEntryPointsKnnSearchStrategy does not support cost()");
                }
            };
        }
    }
}
