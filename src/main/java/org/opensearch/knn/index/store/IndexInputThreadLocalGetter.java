/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.Getter;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;

import java.io.Closeable;
import java.io.IOException;

public class IndexInputThreadLocalGetter implements Closeable {
    @Getter
    private Directory directory;
    @Getter
    private String vectorFileName;
    private IndexInputWithBuffer globalIndexInput;

    private synchronized IndexInputWithBuffer getII() {
        if (globalIndexInput != null) {
            return globalIndexInput;
        }

        try {
            IndexInput indexInput = directory.openInput(vectorFileName, IOContext.READ);
            return globalIndexInput = new IndexInputWithBuffer(indexInput);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public IndexInputThreadLocalGetter(Directory directory, String vectorFileName) {
        this.directory = directory;
        this.vectorFileName = vectorFileName;
    }

    public void close() {
        // TODO
    }

    public IndexInputWithBuffer getIndexInputWithBuffer() throws IOException {
        IndexInputWithBuffer indexInput = getII();
        return new IndexInputWithBuffer(indexInput.indexInput.clone());
    }
}
