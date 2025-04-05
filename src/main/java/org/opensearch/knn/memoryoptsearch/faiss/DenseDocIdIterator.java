/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.index.KnnVectorValues;

import java.io.IOException;

public class DenseDocIdIterator extends KnnVectorValues.DocIndexIterator {
    private final int maxDocId;
    private int doc;

    public DenseDocIdIterator(final int totalNumberOfDocs) {
        maxDocId = totalNumberOfDocs - 1;
        doc = -1;
    }

    @Override
    public int index() {
        return doc;
    }

    @Override
    public int docID() {
        return doc;
    }

    @Override
    public int nextDoc() throws IOException {
        if (doc < maxDocId) {
            return ++doc;
        } else {
            return doc = NO_MORE_DOCS;
        }
    }

    @Override
    public int advance(int target) throws IOException {
        if (target < maxDocId) {
            return doc = target;
        }
        return doc = NO_MORE_DOCS;
    }

    @Override
    public long cost() {
        return maxDocId + 1;
    }
}
