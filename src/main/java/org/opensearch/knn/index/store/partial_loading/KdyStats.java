/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store.partial_loading;

public class KdyStats {
    public static ThreadLocal<KdyStats> TL = ThreadLocal.withInitial(KdyStats::new);

    public long numVectorsVisit;
    public long numBytesRead;

    public void init() {
        numVectorsVisit = 0;
        numBytesRead = 0;
    }

    public void print() {
        synchronized (KdyStats.class) {
            System.out.println("numVectorsVisit = " + numVectorsVisit + ", numBytesRead = " + numBytesRead);
        }
    }
}
