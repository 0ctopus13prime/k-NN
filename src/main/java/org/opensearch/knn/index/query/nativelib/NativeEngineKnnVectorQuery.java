/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.nativelib;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.TopKnnCollectorManager;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.OrdinalTranslatedKnnCollector;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.common.StopWatch;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;
import org.opensearch.knn.index.query.ExactSearcher;
import org.opensearch.knn.index.query.KNNQuery;
import org.opensearch.knn.index.query.KNNWeight;
import org.opensearch.knn.index.query.ResultUtil;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.knn.index.store.partial_loading.FlatL2DistanceComputer;
import org.opensearch.knn.index.store.partial_loading.KdyFaissHnswGraph;
import org.opensearch.knn.index.store.partial_loading.KdyHNSW;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Callable;


/**
 * {@link KNNQuery} executes approximate nearest neighbor search (ANN) on a segment level.
 * {@link NativeEngineKnnVectorQuery} executes approximate nearest neighbor search but gives
 * us the control to combine the top k results in each leaf and post process the results just
 * for k-NN query if required. This is done by overriding rewrite method to execute ANN on each leaf
 * {@link KNNQuery} does not give the ability to post process segment results.
 */
@Log4j2
@Getter
@RequiredArgsConstructor
public class NativeEngineKnnVectorQuery extends Query {

    private final KNNQuery knnQuery;

    public Weight kdyPartialLoading(IndexSearcher indexSearcher, ScoreMode scoreMode, float boost) throws IOException {
        System.out.println("kdyPartialLoading!!!!!!!!!!!!");

        KnnCollectorManager knnCollectorManager = new TopKnnCollectorManager(100, indexSearcher);
        IndexReader reader = indexSearcher.getIndexReader();
        final KNNWeight knnWeight = (KNNWeight) knnQuery.createWeight(indexSearcher, scoreMode, 1);
        List<LeafReaderContext> leafReaderContexts = reader.leaves();
        List<Callable<TopDocs>> tasks = new ArrayList<>(leafReaderContexts.size());
        for (LeafReaderContext context : leafReaderContexts) {
            tasks.add(() -> kdySearchLeaf(knnQuery.getQueryVector(), context, null, knnCollectorManager, knnWeight));
        }
        TaskExecutor taskExecutor = indexSearcher.getTaskExecutor();
        TopDocs[] perLeafResults = taskExecutor.invokeAll(tasks).toArray(TopDocs[]::new);

        // Merge sort the results
        TopDocs topK = TopDocs.merge(10, perLeafResults);
        if (topK.scoreDocs.length == 0) {
            return new MatchNoDocsQuery().createWeight(indexSearcher, scoreMode, boost);
        }
        return createDocAndScoreQuery(reader, topK).createWeight(indexSearcher, scoreMode, boost);
    }

    private TopDocs kdySearchLeaf(
        float[] queryVector,
        LeafReaderContext ctx,
        Weight filterWeight,
        KnnCollectorManager knnCollectorManager,
        KNNWeight knnWeight)
        throws IOException {
        TopDocs results = kdyGetLeafResults(queryVector, ctx, filterWeight, knnCollectorManager, knnWeight);
        if (ctx.docBase > 0) {
            for (ScoreDoc scoreDoc : results.scoreDocs) {
                scoreDoc.doc += ctx.docBase;
            }
        }
        return results;
    }

    private TopDocs kdyGetLeafResults(
        float[] queryVector,
        LeafReaderContext ctx,
        Weight filterWeight,
        KnnCollectorManager knnCollectorManager,
        KNNWeight knnWeight)
        throws IOException {
        final LeafReader reader = ctx.reader();
        final Bits liveDocs = reader.getLiveDocs();

        if (filterWeight == null) {
            return kdyApproximateSearch(queryVector, ctx, liveDocs, Integer.MAX_VALUE, knnCollectorManager, knnWeight);
        }

        throw new RuntimeException("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX filterWeight != null???");
    }

    private TopDocs kdyApproximateSearch(
        float[] queryVector,
        LeafReaderContext context,
        Bits acceptDocs,
        int visitedLimit,
        KnnCollectorManager knnCollectorManager,
        KNNWeight knnWeight)
        throws IOException {
        KnnCollector knnCollector = knnCollectorManager.newCollector(visitedLimit, context);

        NativeMemoryAllocation nativeMemoryAllocation = knnWeight.kdyGetNativeMemoryAllocation(context);
        if (nativeMemoryAllocation == null) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }

        KdyHNSW kdyHNSW = nativeMemoryAllocation.getPartialLoadingContext().kdyHNSW;
        IndexInput indexInput =
            nativeMemoryAllocation.getPartialLoadingContext().indexInputThreadLocalGetter.getIndexInputWithBuffer().indexInput;
        IndexInput vectorIndexInput = indexInput.clone();

        FlatL2DistanceComputer l2Computer =
            new FlatL2DistanceComputer(queryVector, kdyHNSW.indexFlatL2.codes, kdyHNSW.indexFlatL2.oneVectorByteSize);

        RandomVectorScorer scorer = new RandomVectorScorer() {
            @Override public float score(int node) throws IOException {
                return l2Computer.compute(vectorIndexInput, node);
            }

            @Override public int maxOrd() {
                return Integer.MAX_VALUE;
            }
        };

        final KnnCollector collector =
            new OrdinalTranslatedKnnCollector(knnCollector, scorer::ordToDoc);
        final Bits acceptedOrds = null;
        KdyFaissHnswGraph kdyFaissHnswGraph = new KdyFaissHnswGraph(kdyHNSW, indexInput.clone());
        HnswGraphSearcher.search(scorer, collector, kdyFaissHnswGraph, acceptedOrds);
        TopDocs results = knnCollector.topDocs();
        return results;
    }

    @Override
    public Weight createWeight(IndexSearcher indexSearcher, ScoreMode scoreMode, float boost) throws IOException {
        System.out.println("++++++++++++++++++++++++++++");
        return kdyPartialLoading(indexSearcher, scoreMode, boost);







        /*
        final IndexReader reader = indexSearcher.getIndexReader();
        final KNNWeight knnWeight = (KNNWeight) knnQuery.createWeight(indexSearcher, scoreMode, 1);
        List<LeafReaderContext> leafReaderContexts = reader.leaves();
        List<Map<Integer, Float>> perLeafResults;
        RescoreContext rescoreContext = knnQuery.getRescoreContext();
        final int finalK = knnQuery.getK();
        if (rescoreContext == null) {
            perLeafResults = doSearch(indexSearcher, leafReaderContexts, knnWeight, finalK);
        } else {
            boolean isShardLevelRescoringEnabled = KNNSettings.isShardLevelRescoringEnabledForDiskBasedVector(knnQuery.getIndexName());
            int dimension = knnQuery.getQueryVector().length;
            int firstPassK = rescoreContext.getFirstPassK(finalK, isShardLevelRescoringEnabled, dimension);
            perLeafResults = doSearch(indexSearcher, leafReaderContexts, knnWeight, firstPassK);
            if (isShardLevelRescoringEnabled == true) {
                ResultUtil.reduceToTopK(perLeafResults, firstPassK);
            }

            StopWatch stopWatch = new StopWatch().start();
            perLeafResults = doRescore(indexSearcher, leafReaderContexts, knnWeight, perLeafResults, finalK);
            long rescoreTime = stopWatch.stop().totalTime().millis();
            log.debug("Rescoring results took {} ms. oversampled k:{}, segments:{}", rescoreTime, firstPassK, leafReaderContexts.size());
        }
        ResultUtil.reduceToTopK(perLeafResults, finalK);
        TopDocs[] topDocs = new TopDocs[perLeafResults.size()];
        for (int i = 0; i < perLeafResults.size(); i++) {
            topDocs[i] = ResultUtil.resultMapToTopDocs(perLeafResults.get(i), leafReaderContexts.get(i).docBase);
        }

        TopDocs topK = TopDocs.merge(knnQuery.getK(), topDocs);
        if (topK.scoreDocs.length == 0) {
            return new MatchNoDocsQuery().createWeight(indexSearcher, scoreMode, boost);
        }
        return createDocAndScoreQuery(reader, topK).createWeight(indexSearcher, scoreMode, boost);
         */
    }

    private List<Map<Integer, Float>> doSearch(
        final IndexSearcher indexSearcher,
        List<LeafReaderContext> leafReaderContexts,
        KNNWeight knnWeight,
        int k
    ) throws IOException {
        List<Callable<Map<Integer, Float>>> tasks = new ArrayList<>(leafReaderContexts.size());
        for (LeafReaderContext leafReaderContext : leafReaderContexts) {
            tasks.add(() -> searchLeaf(leafReaderContext, knnWeight, k));
        }
        return indexSearcher.getTaskExecutor().invokeAll(tasks);
    }

    private List<Map<Integer, Float>> doRescore(
        final IndexSearcher indexSearcher,
        List<LeafReaderContext> leafReaderContexts,
        KNNWeight knnWeight,
        List<Map<Integer, Float>> perLeafResults,
        int k
    ) throws IOException {
        List<Callable<Map<Integer, Float>>> rescoreTasks = new ArrayList<>(leafReaderContexts.size());
        for (int i = 0; i < perLeafResults.size(); i++) {
            LeafReaderContext leafReaderContext = leafReaderContexts.get(i);
            int finalI = i;
            rescoreTasks.add(() -> {
                BitSet convertedBitSet = ResultUtil.resultMapToMatchBitSet(perLeafResults.get(finalI));
                final ExactSearcher.ExactSearcherContext exactSearcherContext = ExactSearcher.ExactSearcherContext.builder()
                    .matchedDocs(convertedBitSet)
                    // setting to false because in re-scoring we want to do exact search on full precision vectors
                    .useQuantizedVectorsForSearch(false)
                    .k(k)
                    .isParentHits(false)
                    .knnQuery(knnQuery)
                    .build();
                return knnWeight.exactSearch(leafReaderContext, exactSearcherContext);
            });
        }
        return indexSearcher.getTaskExecutor().invokeAll(rescoreTasks);
    }

    private Query createDocAndScoreQuery(IndexReader reader, TopDocs topK) {
        int len = topK.scoreDocs.length;
        Arrays.sort(topK.scoreDocs, Comparator.comparingInt(a -> a.doc));
        int[] docs = new int[len];
        float[] scores = new float[len];
        for (int i = 0; i < len; i++) {
            docs[i] = topK.scoreDocs[i].doc;
            scores[i] = topK.scoreDocs[i].score;
        }
        int[] segmentStarts = findSegmentStarts(reader, docs);
        return new DocAndScoreQuery(knnQuery.getK(), docs, scores, segmentStarts, reader.getContext().id());
    }

    static int[] findSegmentStarts(IndexReader reader, int[] docs) {
        int[] starts = new int[reader.leaves().size() + 1];
        starts[starts.length - 1] = docs.length;
        if (starts.length == 2) {
            return starts;
        }
        int resultIndex = 0;
        for (int i = 1; i < starts.length - 1; i++) {
            int upper = reader.leaves().get(i).docBase;
            resultIndex = Arrays.binarySearch(docs, resultIndex, docs.length, upper);
            if (resultIndex < 0) {
                resultIndex = -1 - resultIndex;
            }
            starts[i] = resultIndex;
        }
        return starts;
    }

    private Map<Integer, Float> searchLeaf(LeafReaderContext ctx, KNNWeight queryWeight, int k) throws IOException {
        final Map<Integer, Float> leafDocScores = queryWeight.searchLeaf(ctx, k);
        final Bits liveDocs = ctx.reader().getLiveDocs();
        if (liveDocs != null) {
            leafDocScores.entrySet().removeIf(entry -> liveDocs.get(entry.getKey()) == false);
        }
        return leafDocScores;
    }

    @Override
    public String toString(String field) {
        return this.getClass().getSimpleName() + "[" + field + "]..." + KNNQuery.class.getSimpleName() + "[" + knnQuery.toString() + "]";
    }

    @Override
    public void visit(QueryVisitor visitor) {
        visitor.visitLeaf(this);
    }

    @Override
    public boolean equals(Object obj) {
        if (!sameClassAs(obj)) {
            return false;
        }
        return knnQuery == ((NativeEngineKnnVectorQuery) obj).knnQuery;
    }

    @Override
    public int hashCode() {
        return Objects.hash(classHash(), knnQuery.hashCode());
    }
}
