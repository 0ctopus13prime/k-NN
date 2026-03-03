/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.bbq;

import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.test.OpenSearchTestCase;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/**
 * Full precision (fp32) vector validation.
 * Loads raw vectors, computes MAXIMUM_INNER_PRODUCT scores via full scan, prints top-k.
 * This serves as the ground truth baseline for recall comparison.
 */
public class FullPrecisionVectorsValidationTests extends OpenSearchTestCase {

    // ===== CONFIGURE THESE =====
    private static final String DATA_PATH = "/Users/kdooyong/workspace/io-opt/tmp/vectors.bin";
    private static final int TOP_K = 100;
    private static final int QUERY_VECTOR_ORDINAL = 0;
    // ===========================

    public void testFullPrecisionRecall() throws Exception {
        float[][] vectors = VectorDataLoader.loadVectors(DATA_PATH);
        int numVectors = vectors.length;
        int dimension = vectors[0].length;

        System.out.println("Loaded " + numVectors + " vectors with dimension " + dimension);

        float[] queryVector = vectors[QUERY_VECTOR_ORDINAL];
        // Full scan using MAXIMUM_INNER_PRODUCT
        List<Map.Entry<Integer, Float>> results = new ArrayList<>();
        for (int i = 0; i < numVectors; i++) {
            if (i == QUERY_VECTOR_ORDINAL) {
                continue;
            }
            float score = KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT.compare(queryVector, vectors[i]);
            results.add(new AbstractMap.SimpleEntry<>(i, score));
        }

        // Sort by score descending
        results.sort(Map.Entry.<Integer, Float>comparingByValue().reversed());

        // Print top-k
        List<Map.Entry<Integer, Float>> topK = results.subList(0, Math.min(TOP_K, results.size()));
        System.out.println("=== Full Precision (fp32) MAXIMUM_INNER_PRODUCT Top-" + TOP_K
            + " (query ordinal=" + QUERY_VECTOR_ORDINAL + ") ===");
        for (Map.Entry<Integer, Float> entry : topK) {
            System.out.println(entry.getKey() + "\t" + entry.getValue());
        }
    }
}
