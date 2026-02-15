/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.kdy;

import lombok.RequiredArgsConstructor;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryHnswIndex;

import java.io.IOException;

@RequiredArgsConstructor
public class FaissBinaryHnswIndexReplacer extends FaissBQIndexHnswReplacer {
    private final String indexType;

    @Override
    protected void doReplace(FaissIndex index, IndexInput indexInput, IndexOutput indexOutput, HnswReplaceContext replaceContex)
        throws IOException {
        final FaissBinaryHnswIndex actualIndex = (FaissBinaryHnswIndex) index;

        // Write index type
        writeIndexType(indexType, indexOutput);

        // Binary header
        copyBinaryCommonHeader(actualIndex, indexOutput);

        // HNSW reordering
        FaissHnswReplacer.transform(indexInput, indexOutput, replaceContex);

        // Transform flat vectors
        replaceHnsw(actualIndex.getStorage(), indexInput, indexOutput, replaceContex);
    }
}
