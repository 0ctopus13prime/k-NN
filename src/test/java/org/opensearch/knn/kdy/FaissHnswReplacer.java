/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.kdy;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHNSW;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHNSWIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIdMapIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;

import java.io.IOException;

public class FaissHnswReplacer {
    public static void transform(
        final IndexInput indexInput,
        final IndexOutput indexOutput,
        final HnswReplaceContext replaceContext
    ) throws IOException {
        // Get FP32 hnsw
        final FaissIndex fp32Index = replaceContext.fp32Index;
        final FaissIdMapIndex faissIdMapIndex = (FaissIdMapIndex) fp32Index;
        final FaissHNSWIndex fp32HnswIndex = (FaissHNSWIndex) faissIdMapIndex.getNestedIndex();
        final FaissHNSW fp32HNSW = fp32HnswIndex.getFaissHnsw();

        // Copy FP32 hnsw
        final IndexInput fp32HnswInput = replaceContext.fp32IndexInput;
        fp32HnswInput.seek(fp32HNSW.getStartOffset());
        byte[] copyBuffer = new byte[64 * 1024];
        long left = fp32HNSW.getHnswSectionSize();
        while (left > 0) {
            final int bytesToRead = (int) Math.min(copyBuffer.length, left);
            fp32HnswInput.readBytes(copyBuffer, 0, bytesToRead);
            indexOutput.writeBytes(copyBuffer, bytesToRead);
            left -= bytesToRead;
        }
    }
}
