/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.memoryoptsearch.optimistic;

import lombok.RequiredArgsConstructor;
import org.apache.lucene.codecs.lucene90.IndexedDISI;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;

import java.io.IOException;
import java.util.Map;

@RequiredArgsConstructor
public class ReentrantKnnCollectorManagerV2 implements KnnCollectorManager {
    private final KnnCollectorManager knnCollectorManager;
    private final Map<Integer, TopDocs> segmentOrdToResults;
    private final float[] query;
    private final String field;

    @Override
    public KnnCollector newCollector(int visitLimit, KnnSearchStrategy searchStrategy, LeafReaderContext ctx) throws IOException {
        // Get delegate collector for empty case
        KnnCollector delegateCollector = knnCollectorManager.newCollector(visitLimit, searchStrategy, ctx);
        TopDocs seedTopDocs = segmentOrdToResults.get(ctx.ord);

        // Get scorer having DISI
        LeafReader reader = ctx.reader();
        FloatVectorValues vectorValues = reader.getFloatVectorValues(field);
        if (vectorValues == null) {
            FloatVectorValues.checkField(reader, field);
            return null;
        }
        VectorScorer scorer = vectorValues.scorer(query);

        if (seedTopDocs.totalHits.value() == 0 || scorer == null) {
            // shouldn't happen - we only come here when there are results
            assert false;
            // on the other hand, it should be safe to return no results?
            return delegateCollector;
        }

        // Get DISI
        DocIdSetIterator vectorIterator = scorer.iterator();

        // Handle sparse
        if (vectorIterator instanceof IndexedDISI indexedDISI) {
            vectorIterator = IndexedDISI.asDocIndexIterator(indexedDISI);
        }

        // Most underlying iterators are indexed, so we can map the seed docs to the vector docs
        if (vectorIterator instanceof KnnVectorValues.DocIndexIterator indexIterator) {
            DocIdSetIterator seedDocs = new SeededMappedDISIV2(indexIterator, new SeededTopDocsDISI(seedTopDocs));
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
