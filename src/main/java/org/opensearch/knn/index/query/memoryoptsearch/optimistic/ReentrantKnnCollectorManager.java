/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.memoryoptsearch.optimistic;

import lombok.RequiredArgsConstructor;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;

import java.io.IOException;
import java.util.Map;

@RequiredArgsConstructor
public class ReentrantKnnCollectorManager implements KnnCollectorManager {
    private final KnnCollectorManager knnCollectorManager;
    private final Map<Integer, TopDocs> segmentOrdToResults;

    @Override
    public KnnCollector newCollector(int visitLimit, KnnSearchStrategy searchStrategy, LeafReaderContext ctx) throws IOException {
        final TopDocs seedTopDocs = segmentOrdToResults.get(ctx.ord);
        final DocIdSetIterator seedDocs = new SeededTopDocsDISI(seedTopDocs);
        return knnCollectorManager.newCollector(
            visitLimit,
            new KnnSearchStrategy.Seeded(seedDocs, seedTopDocs.scoreDocs.length, searchStrategy),
            ctx
        );
    }
}
