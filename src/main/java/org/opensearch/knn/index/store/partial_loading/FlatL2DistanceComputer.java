/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store.partial_loading;

import org.apache.lucene.store.IndexInput;

import java.io.IOException;

public class FlatL2DistanceComputer extends DistanceComputer {
    private float[] queryVector;
    private int dimension;
    private float[] floatBuffer1;
    private Storage codes;
    private long oneVectorByteSize;

    public FlatL2DistanceComputer(float[] queryVector, Storage codes, long oneVectorByteSize) {
        this.queryVector = queryVector;
        this.dimension = queryVector.length;
        this.floatBuffer1 = new float[dimension];
        this.codes = codes;
        this.oneVectorByteSize = oneVectorByteSize;
    }

    @Override public float compute(IndexInput indexInput, long index) throws IOException {
        populateFloats(indexInput, index, floatBuffer1);
        return LuceneVectorUtilSupportProxy.squareDistance(queryVector, floatBuffer1);
    }

    private void populateFloats(IndexInput indexInput, long index, float[] floats) throws IOException {
        // System.out.println("populateFloats, index=" + index + ", oneVectorByteSize=" + oneVectorByteSize);
        indexInput.seek(codes.baseOffset + index * oneVectorByteSize);
        indexInput.readFloats(floats, 0, floats.length);
    }

    @Override public void computeBatch4(IndexInput indexInput, int[] ids, float[] distances) throws IOException {
        populateFloats(indexInput, ids[0], floatBuffer1);
        distances[0] = LuceneVectorUtilSupportProxy.squareDistance(queryVector, floatBuffer1);
        populateFloats(indexInput, ids[1], floatBuffer1);
        distances[1] = LuceneVectorUtilSupportProxy.squareDistance(queryVector, floatBuffer1);
        populateFloats(indexInput, ids[2], floatBuffer1);
        distances[2] = LuceneVectorUtilSupportProxy.squareDistance(queryVector, floatBuffer1);
        populateFloats(indexInput, ids[3], floatBuffer1);
        distances[3] = LuceneVectorUtilSupportProxy.squareDistance(queryVector, floatBuffer1);
    }
}
