/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store.partial_loading;

import org.apache.lucene.store.IndexInput;

import java.io.IOException;

public class LongStorage extends Storage {
    public LongStorage() {
        elementSize = 8;
    }

    public long readLong(IndexInput indexInput, int id) throws IOException {
        indexInput.seek(baseOffset + 8 * id);
        return indexInput.readLong();
    }
}
