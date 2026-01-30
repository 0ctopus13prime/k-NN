/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

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
public class IndexTypeToFaissIndexReordererMapping {
    private static final Map<String, Function<String, FaissIndexReorderTransformer>> INDEX_TYPE_TO_FAISS_INDEX_REORDERER;

    static {
        final Map<String, Function<String, FaissIndexReorderTransformer>> mapping = new HashMap<>();

        mapping.put(FaissIdMapIndex.IBMP, FaissIdMapIndexReorderer::new);
        mapping.put(FaissBinaryHnswIndex.IBHF, FaissBinaryHnswIndexReorderer::new);
        mapping.put(FaissIndexBinaryFlat.IBXF, FaissIndexBinaryFlatReorderer::new);

        INDEX_TYPE_TO_FAISS_INDEX_REORDERER = Collections.unmodifiableMap(mapping);
    }

    public FaissIndexReorderTransformer get(final String indexType) {
        final Function<String, FaissIndexReorderTransformer> transformerSupplierFunc = INDEX_TYPE_TO_FAISS_INDEX_REORDERER.get(indexType);
        if (transformerSupplierFunc != null) {
            return transformerSupplierFunc.apply(indexType);
        }
        throw new UnsupportedFaissIndexException("Index type [" + indexType + "] is not supported.");
    }
}
