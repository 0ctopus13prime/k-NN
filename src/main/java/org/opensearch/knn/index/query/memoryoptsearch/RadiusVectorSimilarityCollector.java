/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.memoryoptsearch;

import org.apache.lucene.search.AbstractKnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.knn.KnnSearchStrategy;

import java.util.ArrayList;
import java.util.List;

/**
 * Clone of Lucene's VectorSimilarityCollector, which cannot be used directly due to its package-private visibility.
 */
public class RadiusVectorSimilarityCollector extends AbstractKnnCollector {
    private static final KnnSearchStrategy.Hnsw DEFAULT_STRATEGY = new KnnSearchStrategy.Hnsw(0);

    // Original thresholds as constructed. Immutable, referenced only for logging / recomputation.
    private final float traversalSimilarity, resultSimilarity;
    // Quantization-slack applied on top of the constructed thresholds. See {@link #applySlack(float)}.
    // Kept mutable + separate from the original thresholds so re-applying slack does not compound.
    private float slack;
    private float maxSimilarity;
    private final List<ScoreDoc> scoreDocList;

    /**
     * Perform a similarity-based graph search. The graph is traversed till better scoring nodes are
     * available, or the best candidate is below {@link #traversalSimilarity}. All traversed nodes
     * above {@link #resultSimilarity} are collected.
     *
     * @param traversalSimilarity (lower) similarity score for graph traversal.
     * @param resultSimilarity (higher) similarity score for result collection.
     * @param visitLimit limit on number of nodes to visit.
     */
    public RadiusVectorSimilarityCollector(float traversalSimilarity, float resultSimilarity, long visitLimit) {
        // TODO: add search strategy support
        super(1, visitLimit, DEFAULT_STRATEGY);
        if (traversalSimilarity > resultSimilarity) {
            throw new IllegalArgumentException("traversalSimilarity should be <= resultSimilarity");
        }
        this.traversalSimilarity = traversalSimilarity;
        this.resultSimilarity = resultSimilarity;
        this.slack = 0f;
        this.maxSimilarity = Float.NEGATIVE_INFINITY;
        this.scoreDocList = new ArrayList<>();
    }

    /**
     * Widen the collector's accept net by a quantization-slack factor {@code eps ∈ [0, 1)}.
     * Both the traversal and result thresholds become {@code threshold * (1 - eps)}.
     *
     * <p>Intended use: the vector searcher (which knows the segment's quantization error
     * distribution) calls this at query time so that borderline docs whose quantized score
     * dipped below {@code min_score} are still accepted. A rescore pass at the caller filters
     * out the resulting false accepts.
     *
     * <p>Safe to call more than once — each call replaces the slack, so slacks do not compound.
     *
     * @param eps slack in {@code [0, 1)}. {@code 0} disables the widening entirely.
     * @throws IllegalArgumentException if {@code eps} is outside {@code [0, 1)} or {@code NaN}.
     */
    public void applySlack(float eps) {
        if (Float.isNaN(eps) || eps < 0f || eps >= 1f) {
            throw new IllegalArgumentException("eps must be in [0, 1), got: " + eps);
        }
        this.slack = eps;
    }

    private float effectiveTraversalSimilarity() {
        return traversalSimilarity * (1f - slack);
    }

    private float effectiveResultSimilarity() {
        return resultSimilarity * (1f - slack);
    }

    @Override
    public boolean collect(int docId, float similarity) {
        maxSimilarity = Math.max(maxSimilarity, similarity);
        if (similarity >= effectiveResultSimilarity()) {
            scoreDocList.add(new ScoreDoc(docId, similarity));
        }
        return true;
    }

    @Override
    public float minCompetitiveSimilarity() {
        return Math.min(effectiveTraversalSimilarity(), maxSimilarity);
    }

    @Override
    public TopDocs topDocs() {
        // Results are not returned in a sorted order to prevent unnecessary calculations (because we do
        // not need to maintain the topK)
        TotalHits.Relation relation = earlyTerminated() ? TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO : TotalHits.Relation.EQUAL_TO;
        return new TopDocs(new TotalHits(visitedCount(), relation), scoreDocList.toArray(ScoreDoc[]::new));
    }

    @Override
    public int numCollected() {
        return scoreDocList.size();
    }
}
