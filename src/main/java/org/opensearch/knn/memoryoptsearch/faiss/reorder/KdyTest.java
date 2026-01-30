/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.packed.BlockPackedWriter;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.ThreadLocalRandom;

public class KdyTest {
    public static void main(String... args) throws IOException {
        // 0 -> doc1, 1 -> doc2
        int N = 10000;
        int[] docs = new int[N];
        for (int i = 0, prev = 0; i < N; ++i) {
            docs[i] = prev + 1 + ThreadLocalRandom.current().nextInt(3);
            prev = docs[i];
        }

        final String dirPath = "/tmp/kdy";
        try (final Directory dir = new MMapDirectory(Path.of(dirPath))) {
            final Path testBinPath = Path.of("/tmp/kdy/test.bin");
            if (Files.exists(testBinPath)) {
                Files.delete(testBinPath);
            }
            IndexOutput output = dir.createOutput("test.bin", IOContext.DEFAULT);
            BlockPackedWriter writer = new BlockPackedWriter(output, 64);
            for (int docId : docs) {
                writer.add(docId);
            }
            writer.finish();
        }
    }
}
