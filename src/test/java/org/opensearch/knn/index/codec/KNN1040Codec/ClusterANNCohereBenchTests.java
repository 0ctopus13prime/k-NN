/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.VectorUtil;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.clusterann.codec.QuantizedVectorReader;
import org.opensearch.knn.jni.SimdVectorComputeService;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Milestone-4 sanity harness for the cluster ANN 4-bit (nibble) path: ingest Cohere vectors, build one segment
 * with {@link ClusterANN1040KnnVectorsFormat}, compute brute-force ground truth and report recall + latency.
 *
 * <p>Opt-in only. Skipped unless {@code -Dtests.clusterann.bench.train=/path/cohere-1m-train.json} is given.
 * Other knobs (all {@code tests.clusterann.bench.*}):
 * <ul>
 *   <li>{@code test}: query file (default: sibling {@code cohere-1m-test.json})</li>
 *   <li>{@code count}: number of train vectors to ingest (default 100000)</li>
 *   <li>{@code queries}: number of queries to run (default 200; the first 20 are warm-up and not timed)</li>
 *   <li>{@code k}: top-k (default 100)</li>
 *   <li>{@code bits}: doc bits (default 4)</li>
 *   <li>{@code space}: {@code ip} (default) or {@code l2}</li>
 *   <li>{@code out}: report file (default {@code build/clusterann-bench-&lt;label&gt;.txt})</li>
 *   <li>{@code label}: free-form label for the report (e.g. {@code native} / {@code java})</li>
 * </ul>
 * Toggle the native kernel per JVM with {@code -Dtests.jvm.argline=-Dclusterann.native.dot=false}.
 */
public class ClusterANNCohereBenchTests extends KNNTestCase {

    private static final String FIELD = "vector";
    private static final int WARMUP = 20;

    public void testCohereRecallAndLatency() throws Exception {
        String trainPath = System.getProperty("tests.clusterann.bench.train");
        assumeTrue("set -Dtests.clusterann.bench.train=<cohere-1m-train.json> to run", trainPath != null);

        Path train = Paths.get(trainPath);
        Path test = Paths.get(System.getProperty("tests.clusterann.bench.test", train.resolveSibling("cohere-1m-test.json").toString()));
        int count = Integer.getInteger("tests.clusterann.bench.count", 100_000);
        int numQueries = Integer.getInteger("tests.clusterann.bench.queries", 200);
        int k = Integer.getInteger("tests.clusterann.bench.k", 100);
        int bits = Integer.getInteger("tests.clusterann.bench.bits", 4);
        boolean l2 = "l2".equalsIgnoreCase(System.getProperty("tests.clusterann.bench.space", "ip"));
        String label = System.getProperty("tests.clusterann.bench.label", QuantizedVectorReader.NATIVE_DOT_PRODUCT ? "native" : "java");
        Path out = Paths.get(System.getProperty("tests.clusterann.bench.out", "build/clusterann-bench-" + label + ".txt"));
        VectorSimilarityFunction sim = l2 ? VectorSimilarityFunction.EUCLIDEAN : VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;

        StringBuilder report = new StringBuilder();
        log(report, "label=%s bits=%d space=%s count=%d queries=%d k=%d", label, bits, sim, count, numQueries, k);
        log(report, "native dot product: %s (kernel=%s)", QuantizedVectorReader.NATIVE_DOT_PRODUCT, kernelName());

        // 1. Load data
        long t0 = System.nanoTime();
        float[][] base = readVectors(train, count);
        float[][] queries = readVectors(test, numQueries);
        int dim = base[0].length;
        log(report, "loaded %d base + %d query vectors, dim=%d in %.1fs", base.length, queries.length, dim, sec(t0));

        // 2. Brute-force ground truth (doc id == ordinal because docs are added in order into one segment)
        t0 = System.nanoTime();
        int[][] truth = new int[queries.length][];
        for (int q = 0; q < queries.length; q++) {
            truth[q] = bruteForceTopK(queries[q], base, k, sim);
        }
        log(report, "ground truth (brute force) in %.1fs", sec(t0));

        // 3. Build one segment with the cluster ANN format
        Path indexDir = createTempDir("clusterann-bench");
        Codec codec = clusterAnnCodec(bits);
        t0 = System.nanoTime();
        try (Directory dir = new MMapDirectory(indexDir)) {
            IndexWriterConfig iwc = new IndexWriterConfig().setCodec(codec)
                .setRAMBufferSizeMB(2000)  // Lucene caps this below 2048; larger inputs flush several segments, forceMerge joins them
                .setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH)
                .setUseCompoundFile(false);
            try (IndexWriter writer = new IndexWriter(dir, iwc)) {
                for (float[] v : base) {
                    Document doc = new Document();
                    doc.add(new KnnFloatVectorField(FIELD, v, sim));
                    writer.addDocument(doc);
                }
                writer.commit();
                writer.forceMerge(1);
            }
            log(report, "indexed + forceMerge(1) in %.1fs; files: %s", sec(t0), summarizeFiles(dir));

            // 4. Search: warm-up, then timed
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertEquals("expected a single segment", 1, reader.leaves().size());
                IndexSearcher searcher = new IndexSearcher(reader);
                long[] latencyNs = new long[queries.length];
                double[] recall = new double[queries.length];
                for (int q = 0; q < queries.length; q++) {
                    long s = System.nanoTime();
                    TopDocs td = searcher.search(new KnnFloatVectorQuery(FIELD, queries[q], k), k);
                    latencyNs[q] = System.nanoTime() - s;
                    recall[q] = recall(td, truth[q]);
                }
                int timed = Math.max(0, queries.length - WARMUP);
                long[] lat = Arrays.copyOfRange(latencyNs, WARMUP, queries.length);
                Arrays.sort(lat);
                double meanRecall = Arrays.stream(recall).skip(WARMUP).average().orElse(Double.NaN);
                double minRecall = Arrays.stream(recall).skip(WARMUP).min().orElse(Double.NaN);
                log(report, "queries timed=%d (after %d warm-up)", timed, WARMUP);
                log(report, "recall@%d: mean=%.4f min=%.3f", k, meanRecall, minRecall);
                if (timed > 0) {
                    log(report, "latency us: p50=%.0f p90=%.0f p99=%.0f mean=%.0f max=%.0f",
                        pct(lat, 0.50) / 1e3, pct(lat, 0.90) / 1e3, pct(lat, 0.99) / 1e3,
                        Arrays.stream(lat).average().orElse(0) / 1e3, lat[lat.length - 1] / 1e3);
                }
            }
        }

        Files.createDirectories(out.toAbsolutePath().getParent());
        try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(out, StandardCharsets.UTF_8))) {
            pw.print(report);
        }
        System.out.print(report);
    }

    // ---------------------------------------------------------------------------------------------

    private static Codec clusterAnnCodec(int bits) {
        // Named "Lucene104" so the segment is read back through the stock codec; the per-field format name
        // ("ClusterANN1040KnnVectorsFormat") is resolved through k-NN's META-INF/services registration.
        return new FilterCodec("Lucene104", new Lucene104Codec()) {
            private final KnnVectorsFormat format = new PerFieldKnnVectorsFormat() {
                @Override
                public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                    return new ClusterANN1040KnnVectorsFormat(bits);
                }
            };

            @Override
            public KnnVectorsFormat knnVectorsFormat() {
                return format;
            }
        };
    }

    private static String kernelName() {
        try {
            return SimdVectorComputeService.bulkQuantizedDotProductKernel();
        } catch (Throwable t) {
            return "unavailable (" + t.getClass().getSimpleName() + ")";
        }
    }

    /** One JSON float array per line, e.g. "[0.28, -0.03, ...]". Hand-rolled split: Jackson is not needed here. */
    private static float[][] readVectors(Path file, int limit) throws IOException {
        List<float[]> out = new ArrayList<>(Math.min(limit, 1 << 20));
        try (BufferedReader br = Files.newBufferedReader(file, StandardCharsets.US_ASCII)) {
            String line;
            while (out.size() < limit && (line = br.readLine()) != null) {
                int start = line.indexOf('[') + 1;
                int end = line.lastIndexOf(']');
                if (start <= 0 || end < start) continue;
                String[] parts = line.substring(start, end).split(",");
                float[] v = new float[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    v[i] = Float.parseFloat(parts[i].trim());
                }
                out.add(v);
            }
        }
        if (out.isEmpty()) throw new IOException("no vectors read from " + file);
        return out.toArray(new float[0][]);
    }

    private static int[] bruteForceTopK(float[] q, float[][] base, int k, VectorSimilarityFunction sim) {
        // Keep the k best by similarity score (higher is better for both MIP and Lucene's L2 score).
        float[] bestScore = new float[k];
        int[] bestId = new int[k];
        Arrays.fill(bestScore, Float.NEGATIVE_INFINITY);
        Arrays.fill(bestId, -1);
        int size = 0;
        for (int i = 0; i < base.length; i++) {
            float s = sim == VectorSimilarityFunction.EUCLIDEAN ? -VectorUtil.squareDistance(q, base[i]) : VectorUtil.dotProduct(q, base[i]);
            if (size < k) {
                bestScore[size] = s;
                bestId[size] = i;
                size++;
                if (size == k) heapify(bestScore, bestId, k);
            } else if (s > bestScore[0]) {
                bestScore[0] = s;
                bestId[0] = i;
                siftDown(bestScore, bestId, 0, k);
            }
        }
        return Arrays.copyOf(bestId, size);
    }

    // Min-heap on score so bestScore[0] is the current k-th best.
    private static void heapify(float[] s, int[] id, int n) {
        for (int i = n / 2 - 1; i >= 0; i--) siftDown(s, id, i, n);
    }

    private static void siftDown(float[] s, int[] id, int i, int n) {
        while (true) {
            int l = 2 * i + 1, r = l + 1, m = i;
            if (l < n && s[l] < s[m]) m = l;
            if (r < n && s[r] < s[m]) m = r;
            if (m == i) return;
            float ts = s[i]; s[i] = s[m]; s[m] = ts;
            int ti = id[i]; id[i] = id[m]; id[m] = ti;
            i = m;
        }
    }

    private static double recall(TopDocs td, int[] truth) {
        Set<Integer> expected = new HashSet<>();
        for (int t : truth) expected.add(t);
        int hit = 0;
        for (ScoreDoc sd : td.scoreDocs) {
            if (expected.contains(sd.doc)) hit++;
        }
        return truth.length == 0 ? 0 : (double) hit / truth.length;
    }

    private static double pct(long[] sorted, double p) {
        if (sorted.length == 0) return Double.NaN;
        int idx = (int) Math.min(sorted.length - 1, Math.round(p * (sorted.length - 1)));
        return sorted[idx];
    }

    private static double sec(long startNs) {
        return (System.nanoTime() - startNs) / 1e9;
    }

    private static String summarizeFiles(Directory dir) throws IOException {
        StringBuilder sb = new StringBuilder();
        long total = 0;
        for (String f : dir.listAll()) {
            long len = dir.fileLength(f);
            total += len;
            if (f.endsWith(".clap") || f.endsWith(".clam") || f.endsWith(".vec")) {
                sb.append(f).append('=').append(len / (1024 * 1024)).append("MB ");
            }
        }
        sb.append("total=").append(total / (1024 * 1024)).append("MB");
        return sb.toString();
    }

    private void log(StringBuilder report, String fmt, Object... args) {
        String line = String.format(Locale.ROOT, fmt, args);
        report.append(line).append('\n');
        logger.info("[ClusterANN-BENCH] {}", line);
    }
}
