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
    private float[] floatBuffer2;
    private float[] floatBuffer3;
    private float[] floatBuffer4;
    private Storage codes;
    private long oneVectorByteSize;
    private byte[] bytesBuffer;

    public FlatL2DistanceComputer(float[] queryVector, Storage codes, long oneVectorByteSize) {
        this.queryVector = queryVector;
        this.dimension = queryVector.length;
        this.floatBuffer1 = new float[dimension];
        this.floatBuffer2 = new float[dimension];
        this.floatBuffer3 = new float[dimension];
        this.floatBuffer4 = new float[dimension];
        this.bytesBuffer = new byte[4 * dimension];
        this.codes = codes;
        this.oneVectorByteSize = oneVectorByteSize;
    }

    @Override public float compute(IndexInput indexInput, long index) throws IOException {
        populateFloats(indexInput, index, floatBuffer1);
        float result = 0;
        for (int i = 0; i < dimension; ++i) {
            final float delta = queryVector[i] - floatBuffer1[i];
            // System.out.println("queryVector[i]=" + queryVector[i]
            //                    + ", floatBuffer1[i]=" + floatBuffer1[i]
            //                    + ", delta=" + delta);
            result += delta * delta;
        }
        return result;
    }

    private void populateFloats(IndexInput indexInput, long index, float[] floats) throws IOException {
        // System.out.println("populateFloats, index=" + index + ", oneVectorByteSize=" + oneVectorByteSize);
        codes.readBytes(indexInput, index * oneVectorByteSize, bytesBuffer);
        for (int i = 0, j = 0; i < bytesBuffer.length ; i += 4, ++j) {
            final int intBits = ((255 & bytesBuffer[i])) | ((255 & bytesBuffer[i + 1]) << 8)
                | ((255 & bytesBuffer[i + 2]) << 16) | ((255 & bytesBuffer[i + 3]) << 24);
            // System.out.println("intBits=" + intBits);
            // System.out.println("++++++++ populateFloats, "
            //     + "b[0]=" + bytesBuffer[0] + ", b[1]=" + bytesBuffer[1]
            //     + ", b[2]=" + bytesBuffer[2] + ", b[3]=" + bytesBuffer[3]);
            floats[j] = Float.intBitsToFloat(intBits);
        }
    }

    @Override
    public void computeBatch4(IndexInput indexInput, int[] ids, float[] distances) throws IOException {
        populateFloats(indexInput, ids[0], floatBuffer1);
        populateFloats(indexInput, ids[1], floatBuffer2);
        populateFloats(indexInput, ids[2], floatBuffer3);
        populateFloats(indexInput, ids[3], floatBuffer4);

        float d0 = 0;
        float d1 = 0;
        float d2 = 0;
        float d3 = 0;
        for (int i = 0; i < dimension; i++) {
            float q0 = queryVector[i] - floatBuffer1[i];
            float q1 = queryVector[i] - floatBuffer2[i];
            float q2 = queryVector[i] - floatBuffer3[i];
            float q3 = queryVector[i] - floatBuffer4[i];
            d0 += q0 * q0;
            d1 += q1 * q1;
            d2 += q2 * q2;
            d3 += q3 * q3;
        }

        distances[0] = d0;
        distances[1] = d1;
        distances[2] = d2;
        distances[3] = d3;
    }
}
