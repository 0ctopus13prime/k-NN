/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.TopKnnCollectorManager;
import org.apache.lucene.util.BitSet;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.stream.Collectors;

public class MemoryOptimizedKNNWeight extends KNNWeight {
    private final KnnCollectorManager knnCollectorManager;

    public MemoryOptimizedKNNWeight(KNNQuery query, float boost, final Weight filterWeight, IndexSearcher searcher, int k) {
        super(query, boost, filterWeight);
        this.knnCollectorManager = new TopKnnCollectorManager(k, searcher);
    }

    @Override
    protected Map<Integer, Float> doANNSearch(
        String vectorIndexFileName,
        LeafReaderContext context,
        SegmentReader reader,
        FieldInfo fieldInfo,
        SpaceType spaceType,
        KNNEngine knnEngine,
        VectorDataType vectorDataType,
        byte[] quantizedVector,
        String modelId,
        BitSet filterIdsBitSet,
        int cardinality,
        int k
    ) throws IOException {
        // TODO : Radius search
        // TODO : Float search

        // Determine visit limit
        int visitedLimit = cardinality + 1;
        if (getFilterWeight() == null) {
            visitedLimit = Integer.MAX_VALUE;
        }

        // Create a collector
        final KnnCollector knnCollector = knnCollectorManager.newCollector(visitedLimit, context);

        // Do ANN search
        final BitSet bitSet = cardinality == 0 ? null : filterIdsBitSet;
        final byte[] target = quantizedVector == null ? knnQuery.getByteQueryVector() : quantizedVector;
        reader.getVectorReader().search(knnQuery.getField(), target, knnCollector, bitSet);
        final TopDocs topDocs = knnCollector.topDocs();

        if (topDocs != null && topDocs.scoreDocs != null && topDocs.scoreDocs.length > 0) {
            // Add explanations if required, then return results
            final KNNQueryResult[] results = new KNNQueryResult[topDocs.scoreDocs.length];
            int i = 0;
            for (final ScoreDoc scoreDoc : topDocs.scoreDocs) {
                results[i] = new KNNQueryResult(scoreDoc.doc, scoreDoc.score);
                ++i;
            }

            addExplainIfRequired(results, knnEngine, spaceType);

            return Arrays.stream(results).collect(Collectors.toMap(KNNQueryResult::getId, KNNQueryResult::getScore));
        }

        return Collections.emptyMap();
    }
}
