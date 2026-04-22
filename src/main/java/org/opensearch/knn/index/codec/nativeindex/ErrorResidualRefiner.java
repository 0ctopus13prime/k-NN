/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import org.apache.lucene.search.ScoreDoc;

import java.io.IOException;

/**
 * Implemented by KnnVectorsReader subclasses that can refine 1st-phase scores
 * using 4-bit quantized error residuals from the {@code .ver} file.
 *
 * <p>Eligibility is determined externally by checking compression level (x32 / SQ 1-bit).
 * If the field is x32, the caller assumes this interface is available and calls {@link #refine}.
 *
 * <p>The refinement adds a correction term {@code <q', Q_4(r)>} to each phase-1 score,
 * where {@code q' = query - centroid} and {@code Q_4(r)} is the 4-bit quantized residual.
 * This replaces the full-precision vector rescore with ~8x less IO.
 *
 * @see ResidualQuantizer — writes the {@code .ver} file during index build (Phase 4)
 */
public interface ErrorResidualRefiner {

    /**
     * Refine scores for the given documents using error correction residuals.
     * Returns {@code ScoreDoc[]} with corrected scores for all input documents
     * (same length as {@code docIds}). The caller handles top-k selection.
     *
     * @param field        the vector field name
     * @param queryVector  the original query vector (used to compute q' = query - centroid)
     * @param docIds       segment-local document IDs (dense case: docId == ordinal)
     * @param phase1Scores corresponding 1st-phase scores from approximate search
     * @return ScoreDoc[] with refined scores for all input documents
     * @throws IOException if reading the .ver file fails
     */
    ScoreDoc[] refine(String field, float[] queryVector, int[] docIds, float[] phase1Scores) throws IOException;
}
