/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.index.FieldInfo;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOUtils;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;
import org.opensearch.knn.memoryoptsearch.VectorSearcherFactory;

import java.io.IOException;

/**
 * This factory returns {@link VectorSearcher} that performs vector search directly on FAISS index.
 * Note that we pass `RANDOM` as advice to prevent the underlying storage from performing read-ahead. Since vector search naturally accesses
 * random vector locations, read-ahead does not improve performance. By passing the `RANDOM` context, we explicitly indicate that
 * this searcher will access vectors randomly.
 */
@Log4j2
public class FaissMemoryOptimizedSearcherFactory implements VectorSearcherFactory {

    @Override
    public VectorSearcher createVectorSearcher(
        final SegmentReadState readState,
        final String faissFileName,
        final FieldInfo fieldInfo
    ) throws IOException {
        final IndexInput indexInput = readState.directory.openInput(
            faissFileName, readState.context.withHints(FileTypeHint.DATA, FileDataHint.KNN_VECTORS, DataAccessHint.RANDOM));

        try {
            // Try load it. Not all FAISS index types are currently supported at the moment.
            return new FaissMemoryOptimizedSearcher(indexInput, fieldInfo, readState);
        } catch (UnsupportedFaissIndexException e) {
            // Clean up input stream.
            try {
                IOUtils.close(indexInput);
            } catch (IOException ioException) {}

            throw e;
        }
    }

}
