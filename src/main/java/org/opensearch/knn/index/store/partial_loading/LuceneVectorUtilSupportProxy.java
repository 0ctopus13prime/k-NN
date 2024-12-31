/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store.partial_loading;

import org.apache.lucene.util.VectorUtil;

public class LuceneVectorUtilSupportProxy {
    public static float squareDistance(float[] a, float[] b) {
        return VectorUtil.squareDistance(a, b);
    }
}
