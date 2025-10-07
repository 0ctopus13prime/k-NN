/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.memoryoptsearch.optimistic;

import org.apache.lucene.codecs.lucene90.IndexedDISI;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;

import java.io.IOException;
import java.util.Map;

public class ReentrantKnnCollectorManager implements KnnCollectorManager {
    private final KnnCollectorManager knnCollectorManager;
    private final Map<Integer, TopDocs> segmentOrdToResults;
    private final String field;
    private final float[] query;

    public ReentrantKnnCollectorManager(
        final KnnCollectorManager knnCollectorManager,
        final Map<Integer, TopDocs> segmentOrdToResults,
        final String field,
        final float[] query
    ) {

        this.knnCollectorManager = knnCollectorManager;
        this.segmentOrdToResults = segmentOrdToResults;
        this.field = field;
        this.query = query;
    }

    @Override
    public KnnCollector newCollector(int visitLimit, KnnSearchStrategy searchStrategy, LeafReaderContext ctx) throws IOException {
        final KnnCollector delegateCollector = knnCollectorManager.newCollector(visitLimit, searchStrategy, ctx);
        final TopDocs seedTopDocs = segmentOrdToResults.get(ctx.ord);
        final FloatVectorValues vectorValues = ctx.reader().getFloatVectorValues(field);
        final VectorScorer vectorScorer = vectorValues.scorer(query);
        DocIdSetIterator vectorIterator = vectorScorer.iterator();

        // Handle sparse
        if (vectorIterator instanceof IndexedDISI indexedDISI) {
            vectorIterator = IndexedDISI.asDocIndexIterator(indexedDISI);
        }
        // Most underlying iterators are indexed, so we can map the seed docs to the vector docs
        if (vectorIterator instanceof KnnVectorValues.DocIndexIterator indexIterator) {
            DocIdSetIterator seedDocs = new SeededMappedDISI(indexIterator, new SeededTopDocsDISI(seedTopDocs, ctx));
            return knnCollectorManager.newCollector(
                visitLimit,
                new KnnSearchStrategy.Seeded(seedDocs, seedTopDocs.scoreDocs.length, searchStrategy),
                ctx
            );
        }
        // could lead to an infinite loop if this ever happens
        assert false;
        return delegateCollector;
    }
}
