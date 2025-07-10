/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import lombok.SneakyThrows;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.common.lucene.store.ByteArrayIndexInput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.memoryoptsearch.FaissHNSWTests;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;

import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;

public class RemoteHnswGraphFlatVectorsReconstructorTests extends KNNTestCase {
    private static final String REMOTE_HNSW_FILE = "faiss_remote_hnsw_150dim_50_vectors.bin";
    private static final String REMOTE_FLAT_VECTORS = "faiss_flat_vectors_150dim_50_vectors.bin";

    @SneakyThrows
    public void testReconstruct() {
        // Load graph file whose flat vectors section is empty
        final IndexInput graphInput = load(REMOTE_HNSW_FILE);
        // Load flat vectors
        final IndexInput flatVectorsInput = load(REMOTE_FLAT_VECTORS);

        // Create temp output directory
        final Path tempDir = createTempDir(UUID.randomUUID().toString());
        final String fileName = "_0_165_target_field.faiss";
        try (final Directory directory = newFSDirectory(tempDir)) {
            try (final IndexOutput destOut = directory.createOutput(fileName, IOContext.DEFAULT)) {
                // Reconstruct HNSW graph + flat vectors
                RemoteHnswGraphFlatVectorsReconstructor.reconstruct(graphInput, flatVectorsInput, destOut);
            }

            // Validate the reconstructed index
            try (final IndexInput input = directory.openInput(fileName, IOContext.READONCE)) {
                FaissIndex.load(input);
            }
        }
    }

    @SneakyThrows
    public static IndexInput load(final String relativePath) {
        final URL hnswWithOneVector = FaissHNSWTests.class.getClassLoader().getResource("data/memoryoptsearch/" + relativePath);
        final byte[] bytes = Files.readAllBytes(Path.of(hnswWithOneVector.toURI()));
        final IndexInput indexInput = new ByteArrayIndexInput("RemoteHnswGraphFlatVectorsReconstructorTests", bytes);
        return indexInput;
    }
}
