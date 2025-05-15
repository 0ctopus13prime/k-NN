/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Kdy {
    private static float MIN_SCORE = 0;

    public static float load() {
        try {
            final String filePath = "/home/ec2-user/efs/tmp/min_score";
            final byte[] bytes = Files.readAllBytes(Paths.get(filePath));
            final String content = new String(bytes);  // default charset
            return MIN_SCORE = Float.parseFloat(content);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static float getMinScore() {
        return MIN_SCORE;
    }
}
