/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.bbq;

import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

import java.io.IOException;

import static org.apache.lucene.codecs.lucene102.Lucene102BinaryQuantizedVectorsFormat.QUERY_BITS;
import static org.apache.lucene.index.VectorSimilarityFunction.COSINE;
import static org.apache.lucene.index.VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;
import static org.apache.lucene.util.quantization.OptimizedScalarQuantizer.transposeHalfByte;

public class Lucene102BinaryFlatVectorsScorer implements FlatVectorsScorer {
    private static final float FOUR_BIT_SCALE = 1f / ((1 << 4) - 1);

    @Override
    public RandomVectorScorerSupplier getRandomVectorScorerSupplier(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues
    ) {
        throw new UnsupportedOperationException();
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues,
        float[] target
    ) throws IOException {
        if (vectorValues instanceof BinarizedByteVectorValues binarizedVectors) {
            OptimizedScalarQuantizer quantizer = binarizedVectors.getQuantizer();
            float[] centroid = binarizedVectors.getCentroid();
            // We make a copy as the quantization process mutates the input
            float[] copy = ArrayUtil.copyOfSubArray(target, 0, target.length);
            if (similarityFunction == COSINE) {
                VectorUtil.l2normalize(copy);
            }
            target = copy;
            byte[] initial = new byte[target.length];
            byte[] quantized = new byte[QUERY_BITS * binarizedVectors.discretizedDimensions() / 8];
            OptimizedScalarQuantizer.QuantizationResult queryCorrections = quantizer.scalarQuantize(target, initial, (byte) 4, centroid);
            transposeHalfByte(initial, quantized);
            return new RandomVectorScorer.AbstractRandomVectorScorer(binarizedVectors) {
                @Override
                public float score(int node) throws IOException {
                    return quantizedScore(quantized, queryCorrections, binarizedVectors, node, similarityFunction);
                }
            };
        }

        throw new UnsupportedOperationException("Lucene102BinaryFlatVectorsScorer only supports BinarizedByteVectorValues");
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues,
        byte[] target
    ) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public String toString() {
        return "Lucene102BinaryFlatVectorsScorer()";
    }

    static float quantizedScore(
        byte[] quantizedQuery,
        OptimizedScalarQuantizer.QuantizationResult queryCorrections,
        BinarizedByteVectorValues targetVectors,
        int targetOrd,
        VectorSimilarityFunction similarityFunction
    ) throws IOException {
        byte[] binaryCode = targetVectors.vectorValue(targetOrd);
        float qcDist = VectorUtil.int4BitDotProduct(quantizedQuery, binaryCode);
        OptimizedScalarQuantizer.QuantizationResult indexCorrections = targetVectors.getCorrectiveTerms(targetOrd);

        // Data vector
        float ax = indexCorrections.lowerInterval();
        // Here we assume `lx` is simply bit vectors, so the scaling isn't necessary
        float lx = indexCorrections.upperInterval() - ax;
        float x1 = indexCorrections.quantizedComponentSum();

        // Query vector
        float ay = queryCorrections.lowerInterval();
        float ly = (queryCorrections.upperInterval() - ay) * FOUR_BIT_SCALE;
        float y1 = queryCorrections.quantizedComponentSum();

        float score = ax * ay * targetVectors.dimension() + ay * lx * x1 + ax * ly * y1 + lx * ly * qcDist;
        score += queryCorrections.additionalCorrection() + indexCorrections.additionalCorrection() - targetVectors.getCentroidDP();
        if (similarityFunction == MAXIMUM_INNER_PRODUCT) {
            return VectorUtil.scaleMaxInnerProductScore(score);
        }
        return Math.max((1f + score) / 2f, 0);
    }
}
