/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store.partial_loading;

public class KdyMaxScoreTracker {
    public int k;
    public DistanceMaxHeap distanceMaxHeap;

    public KdyMaxScoreTracker(int k) {
        this.k = k;
        this.distanceMaxHeap = new DistanceMaxHeap(k);
    }
}
