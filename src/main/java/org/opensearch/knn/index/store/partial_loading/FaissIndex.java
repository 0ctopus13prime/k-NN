/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store.partial_loading;

public class FaissIndex {
    public int d;
    public long nTotal;
    public boolean isTrained;
    public MetricType metricType;

    public enum MetricType {
        METRIC_INNER_PRODUCT,
        METRIC_L2
    }

    public FaissIndex() {
        this(0, MetricType.METRIC_L2);
    }

    public FaissIndex(int d, MetricType metricType) {
        this.d = d;
        this.nTotal = 0;
        isTrained = true;
        this.metricType = metricType;
    }
}
