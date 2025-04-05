/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.experimental.UtilityClass;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;

@UtilityClass
public final class FlatVectorScorerGetter {
    private static final FlatVectorsScorer VECTOR_SCORER = FlatVectorScorerUtil.getLucene99FlatVectorsScorer();

    public FlatVectorsScorer get() {
        return VECTOR_SCORER;
    }
}
