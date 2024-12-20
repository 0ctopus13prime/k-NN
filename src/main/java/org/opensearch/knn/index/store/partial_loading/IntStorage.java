/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store.partial_loading;

import org.apache.lucene.store.IndexInput;

import java.io.IOException;

public class IntStorage extends Storage {
    public IntStorage() {
        elementSize = 4;
    }

    public int readInt(IndexInput indexInput, long index) throws IOException {
        indexInput.seek(baseOffset + 4 * index);
        return indexInput.readInt();
    }
}
