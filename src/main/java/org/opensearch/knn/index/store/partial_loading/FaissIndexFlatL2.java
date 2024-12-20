/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store.partial_loading;

public class FaissIndexFlatL2 extends FaissIndexFlat {
    public Storage codes = new Storage();
    public long oneVectorByteSize;
}
