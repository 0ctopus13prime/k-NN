/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.store.Directory;

import java.io.IOException;

/**
 * Factory to create {@link org.apache.lucene.codecs.KnnVectorsReader}.
 * Provided parameters will have {@link Directory} and a file name where implementation can rely on it to open an input stream.
 */
public interface VectorSearcherFactory {
    /**
     * Create a non-null {@link org.apache.lucene.codecs.KnnVectorsReader} with given Lucene's {@link Directory}.
     *
     * @param directory Lucene's Directory.
     * @param fileName Logical file name to load.
     * @return Null instance if it is not supported, otherwise return {@link org.apache.lucene.codecs.KnnVectorsReader}
     * @throws IOException
     */
    KnnVectorsReader createVectorSearcher(Directory directory, String fileName) throws IOException;
}
