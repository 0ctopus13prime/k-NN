/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clusterann;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.Weight;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.codec.clusterann.vectorformat1030.KNN1030ClusterANNVectorsReader;
import org.opensearch.knn.index.query.KNNScorer;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * One segment per {@link #scorerSupplier}, each answered by the codec reader's
 * {@link KNN1030ClusterANNVectorsReader#searchRadial radial entry point}.
 *
 * <p>Lazy and memoised for the same reasons as {@link ClusterANNWeight}: a segment's scan is paid for only when its
 * scorer is asked for, and Lucene may ask more than once.
 *
 * <p>The reader is reached by unwrapping the leaf to its {@link CodecReader} and asking the per-field vectors reader for
 * this field's reader. A segment whose field was written by a different format is an error rather than an empty result:
 * a radial request on a ClusterANN field can only be answered by a ClusterANN reader, and silently returning nothing
 * would look like "no document within the radius".
 */
final class ClusterANNRadialWeight extends Weight {

    private final String field;
    private final float[] queryVector;
    private final float resultSimilarity;
    private final float admissionSlack;
    private final int maxCandidates;
    @Nullable
    private final Weight filterWeight;
    private final float boost;

    private final Map<Integer, TopDocs> perLeafResults = new ConcurrentHashMap<>();

    ClusterANNRadialWeight(
        final Query query,
        final String field,
        final float[] queryVector,
        final float resultSimilarity,
        final float admissionSlack,
        final int maxCandidates,
        @Nullable final Weight filterWeight,
        final float boost
    ) {
        super(query);
        this.field = field;
        this.queryVector = queryVector;
        this.resultSimilarity = resultSimilarity;
        this.admissionSlack = admissionSlack;
        this.maxCandidates = maxCandidates;
        this.filterWeight = filterWeight;
        this.boost = boost;
    }

    @Override
    public ScorerSupplier scorerSupplier(final LeafReaderContext context) {
        return new ScorerSupplier() {
            @Override
            public Scorer get(final long leadCost) throws IOException {
                final TopDocs topDocs = searchLeaf(context);
                if (topDocs.scoreDocs.length == 0) {
                    return KNNScorer.emptyScorer();
                }
                return new KNNScorer(topDocs, boost);
            }

            @Override
            public long cost() {
                final TopDocs done = perLeafResults.get(context.ord);
                return done == null ? maxCandidates : done.scoreDocs.length;
            }
        };
    }

    /** The radial matches of one segment, in that segment's own doc space, with exact scores. */
    TopDocs searchLeaf(final LeafReaderContext context) throws IOException {
        final TopDocs cached = perLeafResults.get(context.ord);
        if (cached != null) {
            return cached;
        }
        final AcceptDocs acceptDocs = acceptDocs(context);
        TopDocs results = TopDocsCollector.EMPTY_TOPDOCS;
        if (acceptDocs.cost() > 0) {
            final KNN1030ClusterANNVectorsReader reader = clusterReader(context);
            if (reader != null) {
                results = reader.searchRadial(field, queryVector, resultSimilarity, admissionSlack, maxCandidates, acceptDocs);
            }
        }
        perLeafResults.put(context.ord, results);
        return results;
    }

    /**
     * This segment's ClusterANN reader for the field, or {@code null} when the segment holds no vectors for it.
     *
     * @throws IllegalStateException when the field's vectors were written by another format
     */
    @Nullable
    private KNN1030ClusterANNVectorsReader clusterReader(final LeafReaderContext context) {
        final LeafReader leaf = FilterLeafReader.unwrap(context.reader());
        if (!(leaf instanceof CodecReader codecReader)) {
            throw new IllegalStateException("ClusterANN radial search needs a codec reader, got " + leaf.getClass().getName());
        }
        KnnVectorsReader vectorsReader = codecReader.getVectorReader();
        if (vectorsReader == null) {
            return null;
        }
        if (vectorsReader instanceof PerFieldKnnVectorsFormat.FieldsReader perField) {
            vectorsReader = perField.getFieldReader(field);
            if (vectorsReader == null) {
                return null; // no vectors for this field in this segment
            }
        }
        if (vectorsReader instanceof KNN1030ClusterANNVectorsReader clusterReader) {
            return clusterReader;
        }
        throw new IllegalStateException(
            "ClusterANN radial search on field [" + field + "] found a " + vectorsReader.getClass().getSimpleName() + " reader"
        );
    }

    private AcceptDocs acceptDocs(final LeafReaderContext context) throws IOException {
        if (filterWeight == null) {
            return AcceptDocs.fromLiveDocs(context.reader().getLiveDocs(), context.reader().maxDoc());
        }
        return AcceptDocs.fromIteratorSupplier(() -> {
            final Scorer scorer = filterWeight.scorer(context);
            return scorer == null ? DocIdSetIterator.empty() : scorer.iterator();
        }, context.reader().getLiveDocs(), context.reader().maxDoc());
    }

    @Override
    public Explanation explain(final LeafReaderContext context, final int doc) throws IOException {
        final TopDocs topDocs = searchLeaf(context);
        for (final var scoreDoc : topDocs.scoreDocs) {
            if (scoreDoc.doc == doc) {
                return Explanation.match(scoreDoc.score * boost, "exact similarity at or above " + resultSimilarity);
            }
        }
        return Explanation.noMatch("not within the radius (exact similarity below " + resultSimilarity + " or not a candidate)");
    }

    /** Never cacheable: a query vector is effectively unique, so a cache entry would never hit. */
    @Override
    public boolean isCacheable(final LeafReaderContext context) {
        return false;
    }
}
