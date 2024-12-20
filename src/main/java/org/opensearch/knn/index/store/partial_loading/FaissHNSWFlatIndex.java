/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store.partial_loading;

public class FaissHNSWFlatIndex extends FaissIndexHNSW {
    public FaissIndex index;

    public FaissHNSWFlatIndex() {
        super();
        isTrained = true;
    }
}
