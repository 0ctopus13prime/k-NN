/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.kdy;

import lombok.RequiredArgsConstructor;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;

@RequiredArgsConstructor
public class HnswReplaceContext {
    public final FaissIndex fp32Index;
    public final IndexInput fp32IndexInput;
}
