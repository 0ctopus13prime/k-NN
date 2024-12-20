/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store.partial_loading;

public class FaissIndexHNSW extends FaissIndex {
    public FaissHNSW hnsw;

    public FaissIndexHNSW() {
        this(0, 32, MetricType.METRIC_L2);
    }

    public FaissIndexHNSW(int d, int M, MetricType metricType) {
        super(d, metricType);
        hnsw = new FaissHNSW(M);
    }
}
