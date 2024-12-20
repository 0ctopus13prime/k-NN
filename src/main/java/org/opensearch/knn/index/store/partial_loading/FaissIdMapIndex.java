/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store.partial_loading;

public class FaissIdMapIndex extends FaissIndex {
    public FaissIndex index;
    public LongStorage idMap = new LongStorage();
}
