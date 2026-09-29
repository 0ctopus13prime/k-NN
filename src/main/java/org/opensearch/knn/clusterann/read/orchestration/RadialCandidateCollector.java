/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AbstractKnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.util.hnsw.NeighborQueue;

/**
 * First-pass collector for a radial (threshold) search over quantized codes.
 *
 * <p>A radial query has no {@code k}: it wants every vector whose exact similarity clears a threshold. The scan only has
 * quantized estimates, so this collector admits every vector whose <em>estimated</em> score clears a lower, relaxed
 * threshold and leaves the final decision to an exact pass over the stored vectors. The relaxation is what keeps a
 * neighbour whose estimate landed a little under the line; the exact pass is what removes the ones the relaxation let in.
 *
 * <p>Bounded by {@code maxCandidates} so the exact pass, and this collector's memory, cannot grow with the radius. When
 * the bound is hit the lowest-estimated candidates are dropped first and {@link #truncated()} reports it, so the caller
 * can say the result is a lower bound rather than the whole set.
 *
 * <p>{@link #minCompetitiveSimilarity()} is what the block scorers prune against: the admission threshold, raised to the
 * weakest kept candidate once the collector is full. It never reports the exact threshold, since a block whose best
 * estimate is under the exact line may still hold a true match.
 *
 * <p>Collects ordinals, not doc ids: the exact pass reads vectors by ordinal. Single-threaded, one instance per segment
 * per query.
 */
public final class RadialCandidateCollector extends AbstractKnnCollector {

    private final float admissionThreshold;
    private final int maxCandidates;

    /** Min-heap on estimated score, so the top is the weakest candidate kept. */
    private final NeighborQueue queue;

    private boolean truncated;

    /**
     * @param admissionThreshold lowest estimated score a vector needs to become a candidate
     * @param maxCandidates most candidates kept; the weakest are dropped beyond it
     */
    public RadialCandidateCollector(final float admissionThreshold, final int maxCandidates) {
        super(Math.max(1, maxCandidates), Long.MAX_VALUE);
        if (maxCandidates < 1) {
            throw new IllegalArgumentException("maxCandidates must be at least 1, got: " + maxCandidates);
        }
        this.admissionThreshold = admissionThreshold;
        this.maxCandidates = maxCandidates;
        this.queue = new NeighborQueue(maxCandidates, false);
    }

    @Override
    public boolean collect(final int ord, final float estimatedScore) {
        if (estimatedScore < admissionThreshold) {
            return false;
        }
        final boolean full = queue.size() >= maxCandidates;
        final boolean added = queue.insertWithOverflow(ord, estimatedScore);
        if (full) {
            // Either a weaker candidate was evicted or this one was rejected: the set is no longer complete.
            truncated = true;
        }
        return added;
    }

    @Override
    public float minCompetitiveSimilarity() {
        if (queue.size() >= maxCandidates) {
            return Math.max(admissionThreshold, queue.topScore());
        }
        return admissionThreshold;
    }

    /** Candidates as {@code (ordinal, estimated score)}, weakest first. {@code totalHits} is the visited count. */
    @Override
    public TopDocs topDocs() {
        final ScoreDoc[] docs = new ScoreDoc[queue.size()];
        for (int i = 0; i < docs.length; i++) {
            final float score = queue.topScore();
            final int ord = queue.pop();
            docs[i] = new ScoreDoc(ord, score);
        }
        final TotalHits.Relation relation = truncated ? TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO : TotalHits.Relation.EQUAL_TO;
        return new TopDocs(new TotalHits(visitedCount(), relation), docs);
    }

    @Override
    public int numCollected() {
        return queue.size();
    }

    /** Whether the bound was hit, i.e. some admitted candidate was dropped. */
    public boolean truncated() {
        return truncated;
    }

    public float admissionThreshold() {
        return admissionThreshold;
    }

    /**
     * The estimated-score threshold that admits a vector into the candidate set, for an exact threshold of
     * {@code resultSimilarity} and a relative slack in the <em>raw metric</em>.
     *
     * <p>Similarity scores are non-linear transforms of a distance or a dot product, and their scale differs by metric
     * (L2 and cosine scores live in {@code (0, 1]}, maximum-inner-product scores can be in the hundreds), so a fraction of
     * the score would mean something different for every field. The slack is therefore applied to the quantity the
     * quantizer actually estimates and mapped back:
     *
     * <ul>
     *   <li>EUCLIDEAN, {@code s = 1 / (1 + d²)}: admit up to distance {@code d · (1 + slack)}</li>
     *   <li>MAXIMUM_INNER_PRODUCT, {@code s = dot + 1} for {@code dot >= 0} else {@code 1 / (1 - dot)}: admit down to
     *       {@code dot - slack · max(|dot|, 1)}</li>
     *   <li>COSINE and DOT_PRODUCT, {@code s = (1 + x) / 2}: admit down to {@code x - slack}, floored at -1</li>
     * </ul>
     *
     * A slack of 0 admits only estimates already at or above the exact threshold.
     */
    public static float admissionThreshold(final VectorSimilarityFunction similarity, final float resultSimilarity, final float slack) {
        if (slack < 0f) {
            throw new IllegalArgumentException("slack must be >= 0, got: " + slack);
        }
        switch (similarity) {
            case EUCLIDEAN: {
                if (resultSimilarity <= 0f) {
                    return resultSimilarity; // an infinite radius admits everything
                }
                final float d2 = Math.max(0f, 1f / resultSimilarity - 1f);
                final float d = (float) Math.sqrt(d2) * (1f + slack);
                return 1f / (1f + d * d);
            }
            case MAXIMUM_INNER_PRODUCT: {
                final float dot = resultSimilarity >= 1f ? resultSimilarity - 1f : 1f - 1f / Math.max(resultSimilarity, Float.MIN_NORMAL);
                final float admitted = dot - slack * Math.max(Math.abs(dot), 1f);
                return admitted >= 0f ? admitted + 1f : 1f / (1f - admitted);
            }
            case COSINE:
            case DOT_PRODUCT: {
                final float x = 2f * resultSimilarity - 1f;
                final float admitted = Math.max(-1f, x - slack);
                return (1f + admitted) / 2f;
            }
            default:
                throw new IllegalArgumentException("Unsupported similarity: " + similarity);
        }
    }
}
