/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clusterann;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.opensearch.knn.index.codec.clusterann.ClusterANN1030TestCodec;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * Drives {@link ClusterANNRadialQuery} against a real multi-segment ClusterANN index.
 *
 * <p>The corpus is a handful of Gaussian blobs rather than uniform noise: quantized estimates are only meaningful when
 * vectors have structure, and a radius that holds a few hundred documents has to fall inside one blob for the first pass
 * to have anything to find. Doc ids are insertion ordinals (nothing merges or deletes), so a hit indexes straight back
 * into {@code vectors} and brute force needs no side table.
 */
public class ClusterANNRadialQueryTests extends LuceneTestCase {

    private static final String FIELD = "cluster_vec";
    private static final String PARITY_FIELD = "parity";
    private static final int DIM = 16;
    private static final int BLOBS = 8;
    private static final int NUM_DOCS = 4000;
    private static final int DOCS_PER_SEGMENT = 2000;
    private static final float[][] vectors = new float[NUM_DOCS][DIM];

    private Directory directory;
    private DirectoryReader reader;
    private IndexSearcher searcher;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        directory = newDirectory();
        final Random random = new Random(42);
        final float[][] centers = new float[BLOBS][DIM];
        for (float[] c : centers) {
            for (int d = 0; d < DIM; d++)
                c[d] = random.nextFloat();
        }
        final IndexWriterConfig config = new IndexWriterConfig().setCodec(new ClusterANN1030TestCodec(4))
            .setMergePolicy(NoMergePolicy.INSTANCE)
            .setUseCompoundFile(false);
        try (IndexWriter writer = new IndexWriter(directory, config)) {
            for (int i = 0; i < NUM_DOCS; i++) {
                // Random blob per document, so blob membership is independent of the doc-id parity the filter tests use.
                final float[] c = centers[random.nextInt(BLOBS)];
                for (int d = 0; d < DIM; d++) {
                    vectors[i][d] = c[d] + 0.05f * (float) random.nextGaussian();
                }
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField(FIELD, vectors[i], VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField(PARITY_FIELD, i % 2 == 0 ? "even" : "odd", Field.Store.NO));
                writer.addDocument(doc);
                if (i % DOCS_PER_SEGMENT == DOCS_PER_SEGMENT - 1) {
                    writer.flush();
                }
            }
        }
        reader = DirectoryReader.open(directory);
        searcher = new IndexSearcher(reader);
        assertEquals(NUM_DOCS / DOCS_PER_SEGMENT, reader.leaves().size());
    }

    @Override
    public void tearDown() throws Exception {
        reader.close();
        directory.close();
        super.tearDown();
    }

    /** Every returned document really is inside the radius, scored exactly, and the query's own document is among them. */
    public void testHitsAreExactAndWithinRadius() throws IOException {
        final int queryDoc = 7;
        final float[] query = vectors[queryDoc];
        final float threshold = thresholdAtRank(query, 50);
        final TopDocs hits = searcher.search(new ClusterANNRadialQuery(FIELD, query, threshold, null, 0), NUM_DOCS);
        assertTrue("expected some hits", hits.scoreDocs.length > 0);
        boolean sawSelf = false;
        for (final ScoreDoc scoreDoc : hits.scoreDocs) {
            final float exact = VectorSimilarityFunction.EUCLIDEAN.compare(query, vectors[scoreDoc.doc]);
            assertEquals("doc " + scoreDoc.doc + " was not scored exactly", exact, scoreDoc.score, 1e-5f);
            assertTrue("doc " + scoreDoc.doc + " is outside the radius", exact >= threshold);
            sawSelf |= scoreDoc.doc == queryDoc;
        }
        assertTrue("the query's own document must be inside any radius", sawSelf);
    }

    /** Recall against brute force stays high on structured data, at a small and at a mid-sized radius. */
    public void testRecallAgainstBruteForce() throws IOException {
        final float[] query = vectors[123];
        for (final int rank : new int[] { 10, 300 }) {
            final float threshold = thresholdAtRank(query, rank);
            final Set<Integer> truth = withinRadius(query, threshold);
            final TopDocs hits = searcher.search(new ClusterANNRadialQuery(FIELD, query, threshold, null, 0), NUM_DOCS);
            int found = 0;
            for (final ScoreDoc scoreDoc : hits.scoreDocs) {
                if (truth.contains(scoreDoc.doc)) found++;
            }
            final double recall = found / (double) truth.size();
            assertTrue("rank " + rank + ": recall " + recall + " below 0.9 (truth=" + truth.size() + ")", recall >= 0.9);
        }
    }

    /** A radius holding hundreds of documents returns hundreds: the candidate set is not a fixed top-k. */
    public void testLargeRadiusIsNotCappedAtHundred() throws IOException {
        final float[] query = vectors[123];
        final float threshold = thresholdAtRank(query, 400);
        final TopDocs hits = searcher.search(new ClusterANNRadialQuery(FIELD, query, threshold, null, 0), NUM_DOCS);
        assertTrue("only " + hits.scoreDocs.length + " hits for a 400-document radius", hits.scoreDocs.length > 300);
    }

    /** The candidate cap bounds the exact work and surfaces as a truncated result, never as an error. */
    public void testCandidateCapBoundsTheResult() throws IOException {
        final float[] query = vectors[123];
        final float threshold = thresholdAtRank(query, 400);
        final Query capped = new ClusterANNRadialQuery(FIELD, query, threshold, 0.1f, 50, null, 0);
        final TopDocs hits = searcher.search(capped, NUM_DOCS);
        assertTrue("cap of 50 per segment, 2 segments", hits.scoreDocs.length <= 100);
        assertTrue(hits.scoreDocs.length > 0);
    }

    /** Hits stay inside the filter, and a filter matching nothing is an empty result. */
    public void testFilterRestrictsHits() throws IOException {
        final float[] query = vectors[7];
        final float threshold = thresholdAtRank(query, 100);
        final Query evenFilter = new TermQuery(new Term(PARITY_FIELD, "even"));
        final TopDocs even = searcher.search(new ClusterANNRadialQuery(FIELD, query, threshold, evenFilter, 0), NUM_DOCS);
        final TopDocs unfiltered = searcher.search(new ClusterANNRadialQuery(FIELD, query, threshold, null, 0), NUM_DOCS);
        final TopDocs topKEven = searcher.search(new ClusterANNQuery(FIELD, query, 50, evenFilter, null, 0), 50);
        long evenTruth = withinRadius(query, threshold).stream().filter(d -> d % 2 == 0).count();
        assertTrue(
            "filtered radial returned nothing: unfiltered="
                + unfiltered.scoreDocs.length
                + " evenTruth="
                + evenTruth
                + " topKEven="
                + topKEven.scoreDocs.length,
            even.scoreDocs.length > 0
        );
        for (final ScoreDoc scoreDoc : even.scoreDocs) {
            assertEquals("filtered radial search returned an odd document", 0, scoreDoc.doc % 2);
        }
        final TopDocs none = searcher.search(
            new ClusterANNRadialQuery(FIELD, query, threshold, new TermQuery(new Term(PARITY_FIELD, "neither")), 0),
            NUM_DOCS
        );
        assertEquals(0, none.scoreDocs.length);
    }

    /** A radius tighter than the nearest neighbour keeps only the query's own document. */
    public void testTightRadiusReturnsOnlySelf() throws IOException {
        final float[] query = vectors[7];
        final float threshold = thresholdAtRank(query, 1); // between self (score 1.0) and the nearest other document
        final TopDocs hits = searcher.search(new ClusterANNRadialQuery(FIELD, query, threshold, null, 0), NUM_DOCS);
        assertEquals(1, hits.scoreDocs.length);
        assertEquals(7, hits.scoreDocs[0].doc);
    }

    public void testExplain() throws IOException {
        final float[] query = vectors[7];
        final float threshold = thresholdAtRank(query, 10);
        final Query q = new ClusterANNRadialQuery(FIELD, query, threshold, null, 0);
        final Explanation in = searcher.explain(q, 7);
        assertTrue(in.isMatch());
        assertEquals(1.0f, in.getValue().floatValue(), 1e-6f);
        final int farDoc = farthestDoc(query);
        assertFalse(searcher.explain(q, farDoc).isMatch());
    }

    public void testEqualsAndHashCode() {
        final Query a = new ClusterANNRadialQuery(FIELD, vectors[1], 0.5f, null, 0);
        final Query b = new ClusterANNRadialQuery(FIELD, vectors[1], 0.5f, null, 3); // shardId is not identity
        final Query c = new ClusterANNRadialQuery(FIELD, vectors[1], 0.6f, null, 0);
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
        assertNotEquals(a, c);
    }

    public void testRejectsBadParameters() {
        expectThrows(IllegalArgumentException.class, () -> new ClusterANNRadialQuery(FIELD, vectors[1], 0.5f, -0.1f, 10, null, 0));
        expectThrows(IllegalArgumentException.class, () -> new ClusterANNRadialQuery(FIELD, vectors[1], 0.5f, 0.1f, 0, null, 0));
        // zero slack is legal: only estimates already at or above the exact threshold are rescored
        new ClusterANNRadialQuery(FIELD, vectors[1], 0.5f, 0f, 10, null, 0);
    }

    // ------------------------------------------------------------------------------------------------------------

    /** A threshold sitting between the {@code rank}-th and {@code rank+1}-th best exact scores. */
    private static float thresholdAtRank(final float[] query, final int rank) {
        final float[] scores = new float[NUM_DOCS];
        for (int i = 0; i < NUM_DOCS; i++) {
            scores[i] = VectorSimilarityFunction.EUCLIDEAN.compare(query, vectors[i]);
        }
        Arrays.sort(scores);
        final float kth = scores[NUM_DOCS - rank];
        final float next = scores[NUM_DOCS - rank - 1];
        return (kth + next) / 2f;
    }

    private static Set<Integer> withinRadius(final float[] query, final float threshold) {
        final Set<Integer> in = new HashSet<>();
        for (int i = 0; i < NUM_DOCS; i++) {
            if (VectorSimilarityFunction.EUCLIDEAN.compare(query, vectors[i]) >= threshold) in.add(i);
        }
        return in;
    }

    private static int farthestDoc(final float[] query) {
        int worst = 0;
        float worstScore = Float.MAX_VALUE;
        for (int i = 0; i < NUM_DOCS; i++) {
            final float s = VectorSimilarityFunction.EUCLIDEAN.compare(query, vectors[i]);
            if (s < worstScore) {
                worstScore = s;
                worst = i;
            }
        }
        return worst;
    }
}
