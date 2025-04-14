/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.index.FloatVectorValues;

public abstract class NativeFloatVectorValues extends FloatVectorValues {
    public abstract long getFlatVectorsManagerAddress();
}
