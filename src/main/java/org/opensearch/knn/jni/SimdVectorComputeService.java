/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.KNNEngine;

import java.security.AccessController;
import java.security.PrivilegedAction;

import static org.opensearch.knn.index.KNNSettings.isFaissAVX2Disabled;
import static org.opensearch.knn.index.KNNSettings.isFaissAVX512Disabled;
import static org.opensearch.knn.index.KNNSettings.isFaissAVX512SPRDisabled;
import static org.opensearch.knn.jni.PlatformUtils.isAVX2SupportedBySystem;
import static org.opensearch.knn.jni.PlatformUtils.isAVX512SPRSupportedBySystem;
import static org.opensearch.knn.jni.PlatformUtils.isAVX512SupportedBySystem;

public class SimdVectorComputeService {
    static {
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {

            // Even if the underlying system supports AVX512 and AVX2, users can override and disable it by setting
            // 'knn.faiss.avx2.disabled', 'knn.faiss.avx512.disabled', or 'knn.faiss.avx512_spr.disabled' to true in the opensearch.yml
            // configuration
            if (!isFaissAVX512SPRDisabled() && isAVX512SPRSupportedBySystem()) {
            } else if (!isFaissAVX512Disabled() && isAVX512SupportedBySystem()) {
            } else if (!isFaissAVX2Disabled() && isAVX2SupportedBySystem()) {
            } else {
                System.loadLibrary(KNNConstants.DEFAULT_SIMD_COMPUTING_JNI_LIBRARY_NAME);
            }

            return null;
        });
    }

    public enum SimilarityFunctionType {
        FP16_MAXIMUM_INNER_PRODUCT,
        FP16_L2,
    }

    public native static void bulkDistanceCalculation(int[] internalVectorIds, float[] scores, int numVectors);

    public native static void saveSearchContext(float[] query, long[] addressAndSize, int nativeFunctionTypeOrd);

    public native static float scoreSingleVector(int internalVectorId);
}
