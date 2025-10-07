/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.memoryoptsearch.optimistic;

import lombok.experimental.UtilityClass;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.opensearch.knn.index.query.PerLeafResult;

import java.util.List;

@UtilityClass
public class Optimistic2ndSearchUtils {
    public static float findKthLargestScore(final List<PerLeafResult> results, final int k, final int totalResults) {
        assert (totalResults > 0);

        // 1. Flatten all scores into one float[] for fast random access
        float[] scores = new float[totalResults];
        int idx = 0;
        for (PerLeafResult leaf : results) {
            TopDocs td = leaf.getResult();
            if (td == null || td.scoreDocs == null) {
                continue;
            }
            for (ScoreDoc sd : td.scoreDocs) {
                scores[idx++] = sd.score;
            }
        }

        // 3. If fewer than k scores, return the minimum score
        if (totalResults <= k) {
            float min = Float.MAX_VALUE;
            for (int i = 0; i < totalResults; i++)
                if (scores[i] < min) {
                    min = scores[i];
                }
            return min;
        }

        // 4. Otherwise, find kth largest = (N - k)th smallest
        int left = 0, right = totalResults - 1;
        int target = totalResults - k;

        while (left <= right) {
            // Median-of-three pivot selection for better stability
            int mid = (left + right) >>> 1;
            float pivot = medianOfThree(scores[left], scores[mid], scores[right]);

            int i = left, j = right;
            while (i <= j) {
                while (scores[i] < pivot)
                    i++;
                while (scores[j] > pivot)
                    j--;
                if (i <= j) {
                    float tmp = scores[i];
                    scores[i] = scores[j];
                    scores[j] = tmp;
                    i++;
                    j--;
                }
            }

            if (target <= j) {
                right = j;
            } else if (target >= i) {
                left = i;
            } else {
                return scores[target];
            }
        }

        // Should not reach here
        assert (false);

        return scores[target];
    }

    private static float medianOfThree(float a, float b, float c) {
        if (a < b) {
            if (b < c) return b;
            return (a < c) ? c : a;
        } else {
            if (a < c) return a;
            return (b < c) ? c : b;
        }
    }
}
