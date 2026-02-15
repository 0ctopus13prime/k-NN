/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.kdy;

import lombok.RequiredArgsConstructor;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissIndexBinaryFlat;

import java.io.IOException;

@RequiredArgsConstructor
public class FaissIndexBinaryFlatReplacer extends FaissBQIndexHnswReplacer {
    private final String indexType;

    @Override
    protected void doReplace(FaissIndex index, IndexInput indexInput, IndexOutput indexOutput, HnswReplaceContext replaceContext) throws IOException {
        final FaissIndexBinaryFlat actualIndex = (FaissIndexBinaryFlat) index;

        // Write index type
        writeIndexType(indexType, indexOutput);

        // Copy common header
        copyBinaryCommonHeader(actualIndex, indexOutput);

        // Copy vectors
        final ByteVectorValues vectorValues = actualIndex.getByteValues(indexInput);
        indexOutput.writeLong((long) actualIndex.getTotalNumberOfVectors() * actualIndex.getCodeSize());
        for (int i = 0; i < actualIndex.getTotalNumberOfVectors(); ++i) {
            final byte[] vector = vectorValues.vectorValue(i);
            indexOutput.writeBytes(vector, 0, vector.length);
        }
    }
}
