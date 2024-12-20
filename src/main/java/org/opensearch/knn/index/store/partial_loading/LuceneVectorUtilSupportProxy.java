/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store.partial_loading;

import org.apache.lucene.util.Constants;

public class LuceneVectorUtilSupportProxy {

    private static float fma(float a, float b, float c) {
        return Constants.HAS_FAST_SCALAR_FMA ? Math.fma(a, b, c) : a * b + c;
    }

    public static float squareDistance(float[] a, float[] b) {
        float res = 0.0F;
        int i = 0;
        float acc1;
        if (a.length > 32) {
            acc1 = 0.0F;
            float acc2 = 0.0F;
            float acc3 = 0.0F;
            float acc4 = 0.0F;

            for (int upperBound = a.length & -4; i < upperBound; i += 4) {
                float diff1 = a[i] - b[i];
                acc1 = fma(diff1, diff1, acc1);
                float diff2 = a[i + 1] - b[i + 1];
                acc2 = fma(diff2, diff2, acc2);
                float diff3 = a[i + 2] - b[i + 2];
                acc3 = fma(diff3, diff3, acc3);
                float diff4 = a[i + 3] - b[i + 3];
                acc4 = fma(diff4, diff4, acc4);
            }

            res += acc1 + acc2 + acc3 + acc4;
        }

        while (i < a.length) {
            acc1 = a[i] - b[i];
            res = fma(acc1, acc1, res);
            ++i;
        }

        return res;
    }
}
