/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import lombok.AllArgsConstructor;
import org.apache.lucene.store.MMapDirectory;
import org.opensearch.knn.index.store.IndexInputThreadLocalGetter;

@AllArgsConstructor
public class PartialLoadingContext {
    private IndexInputThreadLocalGetter indexInputThreadLocalGetter;

    public boolean isMMapOptimizeAvailable() {
        return indexInputThreadLocalGetter.getDirectory() instanceof MMapDirectory;
    }
}
