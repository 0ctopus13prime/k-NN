/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

public class SimdVectorComputeService {
    public enum SimilarityFunctionType {
        MAXIMUM_INNER_PRODUCT,
        L2,
    }

    public native static void bulkDistanceCalculation(int[] internalVectorIds, float[] scores, int numVectors);

    public native static void saveSearchContext(float[] query, long[] addressAndSize, int nativeFunctionTypeOrd);

    public native static float scoreSingleVector(int internalVectorId);
}
