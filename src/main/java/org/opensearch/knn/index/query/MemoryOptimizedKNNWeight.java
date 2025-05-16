/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.index.ByteVectorValues;
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

        final KnnCollector knnCollector = knnCollectorManager.newCollector(visitedLimit, context);
        final ByteVectorValues byteVectorValues = reader.getByteVectorValues(knnQuery.getField());
        if (byteVectorValues == null) {
            ByteVectorValues.checkField(reader, knnQuery.getField());
            return Collections.emptyMap();
        }

        if (Math.min(knnCollector.k(), byteVectorValues.size()) == 0) {
            return Collections.emptyMap();
        }

        final byte[] target = quantizedVector == null ? knnQuery.getByteQueryVector() : quantizedVector;

        reader.searchNearestVectors(knnQuery.getField(), target, knnCollector, filterIdsBitSet);
        TopDocs topDocs = knnCollector.topDocs();
        if (topDocs != null && topDocs.scoreDocs != null && topDocs.scoreDocs.length > 0) {
            final KNNQueryResult[] results = new KNNQueryResult[topDocs.scoreDocs.length];
            int i = 0;
            for (final ScoreDoc scoreDoc : topDocs.scoreDocs) {
                results[i] = new KNNQueryResult(scoreDoc.doc, scoreDoc.score);
            }

            addExplainIfRequired(results, knnEngine, spaceType);

            return Arrays.stream(results).collect(Collectors.toMap(KNNQueryResult::getId, KNNQueryResult::getScore));
        }

        return Collections.emptyMap();
    }
}
