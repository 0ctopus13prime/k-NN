/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.bbq;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

import java.io.IOException;

import static org.apache.lucene.util.quantization.OptimizedScalarQuantizer.discretize;

public abstract class BinarizedByteVectorValues extends ByteVectorValues {

    /**
     * Retrieve the corrective terms for the given vector ordinal. For the dot-product family of
     * distances, the corrective terms are, in order
     *
     * <ul>
     *   <li>the lower optimized interval
     *   <li>the upper optimized interval
     *   <li>the dot-product of the non-centered vector with the centroid
     *   <li>the sum of quantized components
     * </ul>
     *
     * For euclidean:
     *
     * <ul>
     *   <li>the lower optimized interval
     *   <li>the upper optimized interval
     *   <li>the l2norm of the centered vector
     *   <li>the sum of quantized components
     * </ul>
     *
     * @param vectorOrd the vector ordinal
     * @return the corrective terms
     * @throws IOException if an I/O error occurs
     */
    public abstract OptimizedScalarQuantizer.QuantizationResult getCorrectiveTerms(int vectorOrd)
        throws IOException;

    /**
     * @return the quantizer used to quantize the vectors
     */
    public abstract OptimizedScalarQuantizer getQuantizer();

    public abstract float[] getCentroid() throws IOException;

    int discretizedDimensions() {
        return discretize(dimension(), 64);
    }

    /**
     * Return a {@link VectorScorer} for the given query vector.
     *
     * @param query the query vector
     * @return a {@link VectorScorer} instance or null
     */
    public abstract VectorScorer scorer(float[] query) throws IOException;

    @Override
    public abstract BinarizedByteVectorValues copy() throws IOException;

    public float getCentroidDP() throws IOException {
        // this only gets executed on-merge
        float[] centroid = getCentroid();
        return VectorUtil.dotProduct(centroid, centroid);
    }
}
