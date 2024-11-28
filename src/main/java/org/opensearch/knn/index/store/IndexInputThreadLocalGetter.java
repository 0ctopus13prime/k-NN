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
    private ThreadLocal<IndexInputWithBuffer> indexInputThreadLocal;

    public IndexInputThreadLocalGetter(Directory directory, String vectorFileName) {
        this.directory = directory;
        this.vectorFileName = vectorFileName;
        indexInputThreadLocal = new ThreadLocal<>();
    }

    public void close() {
        final IndexInputWithBuffer indexInputWithBuffer = indexInputThreadLocal.get();
        if (indexInputWithBuffer != null) {
            try {
                indexInputWithBuffer.close();
            } catch (IOException e) {
                // Ignore
            }
            indexInputThreadLocal.remove();
        }
        indexInputThreadLocal = null;
    }

    public IndexInputWithBuffer getIndexInputWithBuffer() throws IOException {
        IndexInputWithBuffer indexInputWithBuffer = indexInputThreadLocal.get();
        if (indexInputWithBuffer != null) {
            return indexInputWithBuffer;
        }

        final IndexInput indexInput = directory.openInput(vectorFileName, IOContext.READ);
        indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
        indexInputThreadLocal.set(indexInputWithBuffer);
        return indexInputWithBuffer;
    }
}
