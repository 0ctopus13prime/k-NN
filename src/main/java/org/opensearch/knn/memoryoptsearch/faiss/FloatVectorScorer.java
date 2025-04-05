/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.IOException;

public class FloatVectorScorer implements VectorScorer {
    private final KnnVectorValues.DocIndexIterator iterator;
    private final RandomVectorScorer scorer;

    public FloatVectorScorer(final float[] target, final FloatVectorValues values, final VectorSimilarityFunction similarityFunction)
        throws IOException {
        iterator = values.iterator();
        scorer = FlatVectorScorerGetter.get().getRandomVectorScorer(similarityFunction, values, target);
    }

    @Override
    public float score() throws IOException {
        return scorer.score(iterator.index());
    }

    @Override
    public DocIdSetIterator iterator() {
        return iterator;
    }
}
