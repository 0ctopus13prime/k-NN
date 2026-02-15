/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.kdy;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryIndex;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

public abstract class FaissBQIndexHnswReplacer {
    public static void replaceHnsw(
        final FaissIndex index,
        final IndexInput indexInput,
        final IndexOutput indexOutput,
        final HnswReplaceContext replaceContex
    ) throws IOException {
        final FaissBQIndexHnswReplacer replacer = FaissBQHnswReplacerMapping.get(index.getIndexType());
        replacer.doReplace(index, indexInput, indexOutput, replaceContex);
    }

    protected void writeIndexType(final String indexType, final IndexOutput indexOuptut) throws IOException {
        final byte[] indexTypeBytes = indexType.getBytes(StandardCharsets.UTF_8);
        indexOuptut.writeBytes(indexTypeBytes, 0, indexTypeBytes.length);
    }

    protected void copyBinaryCommonHeader(final FaissBinaryIndex binaryIndex, IndexOutput indexOutput) throws IOException {
        // Dimension + Code size + #vectors
        indexOutput.writeInt(binaryIndex.getDimension());
        indexOutput.writeInt(binaryIndex.getCodeSize());
        indexOutput.writeLong(binaryIndex.getTotalNumberOfVectors());

        // Copy trained field (deprecated)
        indexOutput.writeByte(binaryIndex.getIsTrainedDeprecatedField());

        // Copy the original metric type
        indexOutput.writeInt(binaryIndex.getOriginalMetricType());
    }

    protected abstract void doReplace(FaissIndex index, IndexInput indexInput, IndexOutput indexOutput, HnswReplaceContext replaceContex)
        throws IOException;
}
