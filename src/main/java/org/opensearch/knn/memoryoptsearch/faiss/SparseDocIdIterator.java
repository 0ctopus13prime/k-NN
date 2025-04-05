/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.index.KnnVectorValues;

import java.io.IOException;

public class SparseDocIdIterator extends KnnVectorValues.DocIndexIterator {
    @Override
    public int index() {
        return 0;
    }

    @Override
    public int docID() {
        return 0;
    }

    @Override
    public int nextDoc() throws IOException {
        return 0;
    }

    @Override
    public int advance(int i) throws IOException {
        return 0;
    }

    @Override
    public long cost() {
        return 0;
    }
}
