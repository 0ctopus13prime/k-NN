/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.bbq;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.EOFException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Shared utility for loading vector data from a comma-delimited text file.
 * Each line contains one vector with float values separated by commas.
 */
public class VectorDataLoader {

    /**
     * Load vectors from a text file where each line is a comma-delimited vector.
     *
     * @param inputDataPath path to the vector data file
     * @return float[][] where each row is a vector
     */
    public static float[][] loadVectors(String inputDataPath) throws IOException {
        try (InputStream fis = Files.newInputStream(Paths.get(inputDataPath));
            BufferedInputStream bis = new BufferedInputStream(fis)) {

            // read dimension line
            StringBuilder sb = new StringBuilder();
            int ch;
            while ((ch = bis.read()) != '\n') {
                if (ch == -1) {
                    throw new EOFException("Unexpected EOF while reading dimension");
                }
                sb.append((char) ch);
            }

            int dim = Integer.parseInt(sb.toString().trim());

            // read remaining bytes
            byte[] remaining = bis.readAllBytes();

            if (remaining.length % 4 != 0) {
                throw new IOException("Binary size is not multiple of 4");
            }

            int totalFloats = remaining.length / 4;

            if (totalFloats % dim != 0) {
                throw new IOException("Float count not divisible by dimension");
            }

            int numVectors = totalFloats / dim;

            ByteBuffer buffer = ByteBuffer.wrap(remaining);
            buffer.order(ByteOrder.LITTLE_ENDIAN);

            List<float[]> result = new ArrayList<>(numVectors);

            for (int i = 0; i < numVectors; i++) {
                float[] vec = new float[dim];
                for (int j = 0; j < dim; j++) {
                    vec[j] = buffer.getFloat();
                }
                result.add(vec);
            }

            return result.toArray(new float[0][]);
        }
    }
}
