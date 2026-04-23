/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.PROPERTIES;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;

/**
 * Integration test for error correction (.ver) file creation during index build.
 *
 * Validates that:
 * <ul>
 *   <li>Ingesting documents into a Faiss HNSW SQ 1-bit (x32) index succeeds</li>
 *   <li>Force merge completes without errors (triggers .ver file creation via Phase 4)</li>
 *   <li>KNN search still works after force merge (no regression from .ver writing)</li>
 * </ul>
 *
 * Does NOT validate .ver file contents or error correction rescoring — that is deferred
 * to the search-side implementation.
 */
public class ErrorCorrectionBuildIT extends KNNRestTestCase {

    private static final String INDEX_NAME = "error-correction-build-test";
    private static final String FIELD_NAME = "target_field";
    private static final int TEST_DIMENSION = 128;
    private static final int NUM_DOCS = 500;
    private static final int K = 10;

    /**
     * End-to-end test: ingest 500 docs into a Faiss HNSW SQ 1-bit index, force merge
     * to a single segment, and verify search still returns results.
     *
     * The force merge triggers MemOptimizedScalarQuantizedIndexBuildStrategy.buildAndWriteIndex()
     * which now includes Phase 4 (writing the .ver file). If Phase 4 throws, the merge fails
     * and this test fails.
     */
    public void testErrorCorrectionFileCreatedOnForceMerge() throws Exception {
        // 1. Create index with Faiss HNSW + SQ 1-bit encoder (x32 compression)
        String mapping = buildSQBits1Mapping();
        createKnnIndex(INDEX_NAME, mapping);

        // 2. Ingest 500 docs with random vectors
        addKNNDocs(INDEX_NAME, FIELD_NAME, TEST_DIMENSION, 0, NUM_DOCS);

        // 3. Force merge to single segment — this triggers the merge path which calls
        // buildAndWriteIndex() → Phase 4 writes .ver file
        forceMergeKnnIndex(INDEX_NAME, 1);

        // 4. Verify KNN search still works (regression check)
        // If .ver file creation broke the index, search would fail here.
        validateKNNSearch(INDEX_NAME, FIELD_NAME, TEST_DIMENSION, NUM_DOCS, K);
    }

    /**
     * Build a mapping for Faiss HNSW with SQ encoder at bits=1 and inner product space type.
     * This produces x32 compression which triggers the MemOptimizedScalarQuantized build path.
     */
    private String buildSQBits1Mapping() throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES)
            .startObject(FIELD_NAME)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, TEST_DIMENSION)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, "innerproduct")
            .startObject(PARAMETERS)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ)
            .startObject(PARAMETERS)
            .field(SQ_BITS, 1)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        return builder.toString();
    }
}
