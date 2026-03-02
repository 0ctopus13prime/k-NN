/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.bbq;

import org.apache.lucene.util.VectorUtil;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.quantization.models.quantizationOutput.BinaryQuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;
import org.opensearch.knn.quantization.quantizer.OneBitScalarQuantizer;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/**
 * Program 3: BQ (OneBitScalarQuantizer) validation.
 * Trains with OneBitScalarQuantizer, quantizes all vectors,
 * then does symmetric bit-vector full scan for top-k using hamming-style dot product.
 */
public class BQRecallValidationTests extends OpenSearchTestCase {

    // ===== CONFIGURE THESE =====
    private static final String DATA_PATH = "/Users/kdooyong/workspace/io-opt/tmp/vectors.bin";
    private static final int TOP_K = 100;
    private static final int QUERY_VECTOR_ORDINAL = 0;
    // ===========================

    public void testBQRecall() throws Exception {
        float[][] vectors = VectorDataLoader.loadVectors(DATA_PATH);
        int dimension = vectors[0].length;
        int numVectors = vectors.length;

        System.out.println("Loaded " + numVectors + " vectors with dimension " + dimension);

        // Step 1: Train quantizer
        final OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        TrainingRequest<float[]> trainingRequest = new TrainingRequest<>(numVectors) {
            @Override
            public float[] getVectorAtThePosition(int position) throws IOException {
                return vectors[position];
            }

            @Override
            public void resetVectorValues() {
                // no-op, stateless access
            }
        };

        QuantizationState state = quantizer.train(trainingRequest);

        // Step 2: Quantize all vectors
        List<byte[]> quantizedVectors = new ArrayList<>();
        BinaryQuantizationOutput output = new BinaryQuantizationOutput(1);
        for (int i = 0; i < numVectors; i++) {
            quantizer.quantize(vectors[i], state, output);
            quantizedVectors.add(output.getQuantizedVectorCopy());
        }

        // Step 3: Full scan with bit dot product
        byte[] queryQuantized = quantizedVectors.get(QUERY_VECTOR_ORDINAL);

        List<Map.Entry<Integer, Float>> results = new ArrayList<>();
        for (int i = 0; i < numVectors; i++) {
            if (i == QUERY_VECTOR_ORDINAL) {
                continue;
            }
            byte[] target = quantizedVectors.get(i);

            // hamming
            final float score = KNNVectorSimilarityFunction.HAMMING.compare(queryQuantized, target);
            results.add(new AbstractMap.SimpleEntry<>(i, score));
        }

        // Sort by score descending (higher bit overlap = more similar)
        results.sort(Map.Entry.<Integer, Float>comparingByValue().reversed());

        // Print top-k
        List<Map.Entry<Integer, Float>> topK = results.subList(0, Math.min(TOP_K, results.size()));
        System.out.println("=== BQ (OneBitScalarQuantizer) Top-" + TOP_K + " (query ordinal=" + QUERY_VECTOR_ORDINAL + ") ===");
        for (Map.Entry<Integer, Float> entry : topK) {
            final float dist = entry.getValue();
            final float sim = VectorUtil.scaleMaxInnerProductScore(dist);
            System.out.println(entry.getKey() + "\t" + dist);
        }
    }
}
