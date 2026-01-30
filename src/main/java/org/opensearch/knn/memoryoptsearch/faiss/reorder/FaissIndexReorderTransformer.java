/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryIndex;

import java.io.IOException;

public abstract class FaissIndexReorderTransformer {
    public static void transform(
        final FaissIndex index,
        IndexInput indexInput,
        final IndexOutput indexOutput,
        final ReorderOrdMap reorderOrdMap
    ) throws IOException {
        final FaissIndexReorderTransformer transformer = IndexTypeToFaissIndexReordererMapping.get(index.getIndexType());
        transformer.doTransform(index, indexInput, indexOutput, reorderOrdMap);
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

    protected abstract void doTransform(FaissIndex index, IndexInput indexInput, IndexOutput indexOutput, ReorderOrdMap reorderOrdMap)
        throws IOException;
}
