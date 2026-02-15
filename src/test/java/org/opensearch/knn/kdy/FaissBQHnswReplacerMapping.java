/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.kdy;

import lombok.experimental.UtilityClass;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIdMapIndex;
import org.opensearch.knn.memoryoptsearch.faiss.UnsupportedFaissIndexException;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryHnswIndex;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissIndexBinaryFlat;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

@UtilityClass
public class FaissBQHnswReplacerMapping {
    private static final Map<String, Function<String, FaissBQIndexHnswReplacer>> INDEX_TYPE_TO_REPLACER;

    static {
        final Map<String, Function<String, FaissBQIndexHnswReplacer>> mapping = new HashMap<>();

        mapping.put(FaissIdMapIndex.IBMP, FaissIdMapIndexReplacer::new);
        mapping.put(FaissBinaryHnswIndex.IBHF, FaissBinaryHnswIndexReplacer::new);
        mapping.put(FaissIndexBinaryFlat.IBXF, FaissIndexBinaryFlatReplacer::new);

        INDEX_TYPE_TO_REPLACER = Collections.unmodifiableMap(mapping);
    }

    public FaissBQIndexHnswReplacer get(final String indexType) {
        final Function<String, FaissBQIndexHnswReplacer> transformerSupplierFunc = INDEX_TYPE_TO_REPLACER.get(indexType);
        if (transformerSupplierFunc != null) {
            return transformerSupplierFunc.apply(indexType);
        }
        throw new UnsupportedFaissIndexException("Index type [" + indexType + "] is not supported.");
    }
}
