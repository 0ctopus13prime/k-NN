/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import lombok.RequiredArgsConstructor;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissIndexBinaryFlat;

import java.io.IOException;

@RequiredArgsConstructor
public class FaissIndexBinaryFlatReorderer extends FaissIndexReorderTransformer {
    private final String indexType;

    @Override
    protected void doTransform(FaissIndex index, IndexInput indexInput, IndexOutput indexOutput, ReorderOrdMap reorderOrdMap)
        throws IOException {
        final FaissIndexBinaryFlat actualIndex = (FaissIndexBinaryFlat) index;

        // Copy common header
        copyBinaryCommonHeader(actualIndex, indexOutput);

        // Reorder flat vectors
        final ByteVectorValues vectorValues = actualIndex.getByteValues(indexInput);

        // Reorder vectors
        for (final int oldOrd : reorderOrdMap.newOrd2Old) {
            final byte[] vector = vectorValues.vectorValue(oldOrd);
            indexOutput.writeBytes(vector, 0, vector.length);
        }
    }
}
