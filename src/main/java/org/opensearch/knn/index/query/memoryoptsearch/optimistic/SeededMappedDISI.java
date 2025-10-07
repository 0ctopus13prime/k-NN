/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.memoryoptsearch.optimistic;

import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.search.DocIdSetIterator;

import java.io.IOException;

public class SeededMappedDISI extends DocIdSetIterator {
    private KnnVectorValues.DocIndexIterator indexedDISI;
    private DocIdSetIterator sourceDISI;

    public SeededMappedDISI(KnnVectorValues.DocIndexIterator indexedDISI, DocIdSetIterator sourceDISI) {
        this.indexedDISI = indexedDISI;
        this.sourceDISI = sourceDISI;
    }

    /**
     * Advances the source iterator to the first document number that is greater than or equal to
     * the provided target and returns the corresponding index.
     */
    @Override
    public int advance(int target) throws IOException {
        int newTarget = sourceDISI.advance(target);
        if (newTarget != NO_MORE_DOCS) {
            indexedDISI.advance(newTarget);
        }
        return docID();
    }

    @Override
    public long cost() {
        return sourceDISI.cost();
    }

    @Override
    public int docID() {
        if (indexedDISI.docID() == NO_MORE_DOCS || sourceDISI.docID() == NO_MORE_DOCS) {
            return NO_MORE_DOCS;
        }
        return indexedDISI.index();
    }

    /** Advances to the next document in the source iterator and returns the corresponding index. */
    @Override
    public int nextDoc() throws IOException {
        int newTarget = sourceDISI.nextDoc();
        if (newTarget != NO_MORE_DOCS) {
            indexedDISI.advance(newTarget);
        }
        return docID();
    }
}
