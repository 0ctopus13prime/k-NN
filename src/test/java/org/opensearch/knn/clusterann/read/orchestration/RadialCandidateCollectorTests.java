/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RadialCandidateCollectorTests {

    @Test
    void testCollect_belowAdmissionThreshold_isRejected() {
        RadialCandidateCollector c = new RadialCandidateCollector(0.5f, 10);
        assertFalse(c.collect(1, 0.49f));
        assertTrue(c.collect(2, 0.5f));
        assertTrue(c.collect(3, 0.9f));
        assertEquals(2, c.numCollected());
        assertFalse(c.truncated());
        assertEquals(0.5f, c.minCompetitiveSimilarity(), 0f);
    }

    @Test
    void testCollect_beyondCapacity_keepsStrongestAndReportsTruncation() {
        RadialCandidateCollector c = new RadialCandidateCollector(0.1f, 3);
        c.collect(1, 0.2f);
        c.collect(2, 0.4f);
        c.collect(3, 0.3f);
        assertEquals(0.2f, c.minCompetitiveSimilarity(), 0f, "full: the weakest kept becomes the bar");
        assertTrue(c.collect(4, 0.5f), "stronger than the weakest: admitted, evicting ord 1");
        assertFalse(c.collect(5, 0.15f), "weaker than the weakest kept: rejected");
        assertTrue(c.truncated());
        assertEquals(0.3f, c.minCompetitiveSimilarity(), 0f);

        TopDocs td = c.topDocs();
        int[] ords = Arrays.stream(td.scoreDocs).mapToInt(sd -> sd.doc).sorted().toArray();
        assertArrayEquals(new int[] { 2, 3, 4 }, ords);
        assertEquals(TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO, td.totalHits.relation());
    }

    @Test
    void testTopDocs_carriesEstimatedScoresAndVisitedCount() {
        RadialCandidateCollector c = new RadialCandidateCollector(0f, 10);
        c.incVisitedCount(7);
        c.collect(11, 0.7f);
        c.collect(12, 0.9f);
        TopDocs td = c.topDocs();
        assertEquals(7, td.totalHits.value());
        assertEquals(TotalHits.Relation.EQUAL_TO, td.totalHits.relation());
        for (ScoreDoc sd : td.scoreDocs) {
            assertEquals(sd.doc == 11 ? 0.7f : 0.9f, sd.score, 0f);
        }
    }

    @Test
    void testConstructor_rejectsNonPositiveCapacity() {
        assertThrows(IllegalArgumentException.class, () -> new RadialCandidateCollector(0f, 0));
    }

    @Test
    void testAdmissionThreshold_isScaleFreeAcrossMetrics() {
        // L2: score 0.5 <-> d = 1; 10% distance slack -> d = 1.1 -> score 1 / (1 + 1.21)
        assertEquals(1f / 2.21f, RadialCandidateCollector.admissionThreshold(VectorSimilarityFunction.EUCLIDEAN, 0.5f, 0.1f), 1e-6f);
        // MIP: score 140 <-> dot 139; 10% slack -> dot 125.1 -> score 126.1
        assertEquals(
            126.1f,
            RadialCandidateCollector.admissionThreshold(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, 140f, 0.1f),
            1e-3f
        );
        // MIP with a negative dot: score 0.5 <-> dot -1; slack relative to max(|dot|, 1) = 1 -> dot -1.1 -> 1 / 2.1
        assertEquals(
            1f / 2.1f,
            RadialCandidateCollector.admissionThreshold(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, 0.5f, 0.1f),
            1e-6f
        );
        // cosine: score 0.9 <-> cos 0.8; slack 0.1 -> cos 0.7 -> score 0.85
        assertEquals(0.85f, RadialCandidateCollector.admissionThreshold(VectorSimilarityFunction.COSINE, 0.9f, 0.1f), 1e-6f);
        // zero slack is the identity for every metric
        for (VectorSimilarityFunction sim : new VectorSimilarityFunction[] {
            VectorSimilarityFunction.EUCLIDEAN,
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
            VectorSimilarityFunction.COSINE }) {
            assertEquals(0.7f, RadialCandidateCollector.admissionThreshold(sim, 0.7f, 0f), 1e-6f, sim.toString());
        }
        assertThrows(
            IllegalArgumentException.class,
            () -> RadialCandidateCollector.admissionThreshold(VectorSimilarityFunction.EUCLIDEAN, 0.5f, -1f)
        );
    }

    private static void assertArrayEquals(int[] expected, int[] actual) {
        assertEquals(Arrays.toString(expected), Arrays.toString(actual));
    }
}
