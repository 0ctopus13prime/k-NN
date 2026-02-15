/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.kdy;

import lombok.RequiredArgsConstructor;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIdMapIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;

import java.io.IOException;

@RequiredArgsConstructor
public class FaissIdMapIndexReplacer extends FaissBQIndexHnswReplacer {
    private final String indexType;

    @Override
    protected void doReplace(FaissIndex index, IndexInput indexInput, IndexOutput indexOutput, HnswReplaceContext replaceContext) throws IOException {
        final FaissIdMapIndex actualIndex = (FaissIdMapIndex) index;

        // Write index type
        writeIndexType(indexType, indexOutput);

        // Copy header
        copyBinaryCommonHeader(actualIndex, indexOutput);

        // Replace HNSW in nested index
        replaceHnsw(actualIndex.getNestedIndex(), indexInput, indexOutput, replaceContext);

        // Copy id mapping. In this POC, it assumes it's a dense case where doc_id == vector ordinal
        indexOutput.writeLong(actualIndex.getTotalNumberOfVectors());
        for (long i = 0; i < actualIndex.getTotalNumberOfVectors(); ++i) {
            indexOutput.writeLong(i);
        }
    }
}
