/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clusterann;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Weight;
import org.opensearch.common.Nullable;

import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

/**
 * Radial (threshold) search over a ClusterANN field: every document whose exact similarity to the query is at least
 * {@link #resultSimilarity}, scored exactly.
 *
 * <p>The radius arrives already converted to a Lucene-style similarity, the same scale the cluster scorers and
 * {@link org.apache.lucene.index.VectorSimilarityFunction} produce, so one threshold serves both the quantized scan and
 * the exact pass. {@code KNNQueryBuilder} does the conversion from {@code max_distance} / {@code min_score} with the
 * Lucene engine's formulas.
 *
 * <p><b>Not Lucene's {@code FloatVectorSimilarityQuery}, on purpose.</b> That query hands the codec reader a top-k
 * shaped collector whose threshold is package-private and which the timeout machinery wraps, so the reader could not
 * tell a radial request from a top-k one reliably. This query calls the reader's dedicated
 * {@code searchRadial} entry point instead, where both phases — relaxed quantized admission, then exact filtering — are
 * explicit. See {@link ClusterANNRadialWeight}.
 *
 * <p>Nested fields are not supported here, matching the Lucene engine's radial search.
 */
@Log4j2
public class ClusterANNRadialQuery extends Query {

    /**
     * Default relative slack, in the raw metric (distance or dot product), by which a quantized estimate may fall short
     * of the exact threshold and still be rescored. Above 0 so that a true neighbour whose estimate came in slightly
     * under the line is not lost; the exact pass removes what the slack lets through. Scale-free across metrics, see
     * {@code RadialCandidateCollector#admissionThreshold}. Override with {@code -Dclusterann.radial.admission_slack}.
     *
     * <p>Measured on Cohere 100k, 768d, 4-bit, MIP: slack 0.03 and 0.1 gave the same recall (0.97 / 0.93 / 0.87 at radii
     * of 10 / 100 / 1000 docs) while 0 lost 5 to 10 points, and every step down in slack cut the exact pass roughly in
     * half. 0.05 keeps a margin over the measured knee. Coarser codes (1 and 2 bit) estimate less accurately and may want
     * a larger value.
     */
    public static final float DEFAULT_ADMISSION_SLACK = Float.parseFloat(System.getProperty("clusterann.radial.admission_slack", "0.05"));

    /**
     * Default cap on candidates rescored per segment, so a huge radius cannot turn into an exact scan of the segment.
     * Override with {@code -Dclusterann.radial.max_candidates}.
     */
    public static final int DEFAULT_MAX_CANDIDATES = Integer.getInteger("clusterann.radial.max_candidates", 10_000);

    private final String field;
    private final float[] queryVector;
    private final float resultSimilarity;
    private final float admissionSlack;
    private final int maxCandidates;
    @Nullable
    private final Query filter;
    /** Only for logging; a query has no business knowing which shard it is on. */
    private final int shardId;

    public ClusterANNRadialQuery(
        final String field,
        final float[] queryVector,
        final float resultSimilarity,
        @Nullable final Query filter,
        final int shardId
    ) {
        this(field, queryVector, resultSimilarity, DEFAULT_ADMISSION_SLACK, DEFAULT_MAX_CANDIDATES, filter, shardId);
    }

    public ClusterANNRadialQuery(
        final String field,
        final float[] queryVector,
        final float resultSimilarity,
        final float admissionSlack,
        final int maxCandidates,
        @Nullable final Query filter,
        final int shardId
    ) {
        if (admissionSlack < 0f) {
            throw new IllegalArgumentException("admissionSlack must be >= 0, got: " + admissionSlack);
        }
        if (maxCandidates < 1) {
            throw new IllegalArgumentException("maxCandidates must be at least 1, got: " + maxCandidates);
        }
        this.field = Objects.requireNonNull(field, "field");
        this.queryVector = Objects.requireNonNull(queryVector, "queryVector");
        this.resultSimilarity = resultSimilarity;
        this.admissionSlack = admissionSlack;
        this.maxCandidates = maxCandidates;
        this.filter = filter;
        this.shardId = shardId;
    }

    /** Deliberately a no-op: the scan happens when a scorer is asked for. */
    @Override
    public Query rewrite(final IndexSearcher indexSearcher) {
        return this;
    }

    @Override
    public Weight createWeight(final IndexSearcher searcher, final ScoreMode scoreMode, final float boost) throws IOException {
        return new ClusterANNRadialWeight(
            this,
            field,
            queryVector,
            resultSimilarity,
            admissionSlack,
            maxCandidates,
            filterWeight(searcher),
            boost
        );
    }

    /** The filter, conjoined with "this field has a vector", as {@link ClusterANNQuery} does. */
    @Nullable
    private Weight filterWeight(final IndexSearcher searcher) throws IOException {
        if (filter == null) {
            return null;
        }
        final BooleanQuery booleanQuery = new BooleanQuery.Builder().add(filter, BooleanClause.Occur.FILTER)
            .add(new FieldExistsQuery(field), BooleanClause.Occur.FILTER)
            .build();
        final Query rewritten = searcher.rewrite(booleanQuery);
        return searcher.createWeight(rewritten, ScoreMode.COMPLETE_NO_SCORES, 1.0f);
    }

    public String getField() {
        return field;
    }

    public float[] getQueryVector() {
        return queryVector;
    }

    public float getResultSimilarity() {
        return resultSimilarity;
    }

    public float getAdmissionSlack() {
        return admissionSlack;
    }

    public int getMaxCandidates() {
        return maxCandidates;
    }

    @Nullable
    public Query getFilter() {
        return filter;
    }

    @Override
    public void visit(final QueryVisitor visitor) {
        visitor.visitLeaf(this);
    }

    @Override
    public String toString(final String f) {
        return getClass().getSimpleName()
            + "[field="
            + field
            + ", resultSimilarity="
            + resultSimilarity
            + ", admissionSlack="
            + admissionSlack
            + ", maxCandidates="
            + maxCandidates
            + ", filter="
            + filter
            + ", shardId="
            + shardId
            + "]";
    }

    @Override
    public boolean equals(final Object other) {
        if (!sameClassAs(other)) {
            return false;
        }
        final ClusterANNRadialQuery o = (ClusterANNRadialQuery) other;
        return Float.compare(resultSimilarity, o.resultSimilarity) == 0
            && Float.compare(admissionSlack, o.admissionSlack) == 0
            && maxCandidates == o.maxCandidates
            && Objects.equals(field, o.field)
            && Arrays.equals(queryVector, o.queryVector)
            && Objects.equals(filter, o.filter);
    }

    @Override
    public int hashCode() {
        return Objects.hash(field, Arrays.hashCode(queryVector), resultSimilarity, admissionSlack, maxCandidates, filter);
    }
}
