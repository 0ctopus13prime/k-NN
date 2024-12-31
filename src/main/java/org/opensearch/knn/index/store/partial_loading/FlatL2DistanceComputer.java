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
    private long lastIndex = -1;
    private float lastScore;

    public FlatL2DistanceComputer(float[] queryVector, Storage codes, long oneVectorByteSize) {
        this.queryVector = queryVector;
        this.dimension = queryVector.length;
        this.floatBuffer1 = new float[dimension];
        this.codes = codes;
        this.oneVectorByteSize = oneVectorByteSize;
    }

    @Override public float compute(IndexInput indexInput, long index) throws IOException {
        if (index != lastIndex) {
            populateFloats(indexInput, index, floatBuffer1);
            lastIndex = index;
            return lastScore = LuceneVectorUtilSupportProxy.squareDistance(queryVector, floatBuffer1);
        }
        return lastScore;
    }

    private void populateFloats(IndexInput indexInput, long index, float[] floats) throws IOException {
        indexInput.seek(codes.baseOffset + index * oneVectorByteSize);
        indexInput.readFloats(floats, 0, floats.length);
    }
}
