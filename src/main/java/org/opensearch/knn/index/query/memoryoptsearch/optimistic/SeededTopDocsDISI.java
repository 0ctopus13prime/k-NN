/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.memoryoptsearch.optimistic;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.TopDocs;

import java.io.IOException;
import java.util.Arrays;

public class SeededTopDocsDISI extends DocIdSetIterator {
    private final int[] sortedDocIds;
    private int idx = -1;

    public SeededTopDocsDISI(final TopDocs topDocs) {
        sortedDocIds = new int[topDocs.scoreDocs.length];
        for (int i = 0; i < topDocs.scoreDocs.length; i++) {
            // Remove the doc base as added by the collector
            sortedDocIds[i] = topDocs.scoreDocs[i].doc;
        }
        Arrays.sort(sortedDocIds);
    }

    @Override
    public int advance(int target) throws IOException {
        return slowAdvance(target);
    }

    @Override
    public long cost() {
        return sortedDocIds.length;
    }

    @Override
    public int docID() {
        if (idx == -1) {
            return -1;
        } else if (idx >= sortedDocIds.length) {
            return DocIdSetIterator.NO_MORE_DOCS;
        } else {
            return sortedDocIds[idx];
        }
    }

    @Override
    public int nextDoc() {
        idx += 1;
        return docID();
    }
}
