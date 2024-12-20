/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store.partial_loading;

import org.apache.lucene.store.IndexInput;

import java.io.IOException;

public abstract class DistanceComputer {
    public abstract float compute(IndexInput indexInput, long index) throws IOException;

    public abstract void computeBatch4(IndexInput indexInput, int[] ids, float[] distances) throws IOException;
}
