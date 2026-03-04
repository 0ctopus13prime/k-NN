# Goal
Now, we need to test out rerank performance.
FaissMemoryOptimizedSearcher has `rerank` method to use error residual to adjust the final score and do the rerank.
Currently, we're loading the full precision vectors for rerank, and the goal is to identify the performance gap between two.

Current flow is to do ANN search first, then refine scores with reranking. (See `doRescore` method in NativeEngineKnnVectorQuery)
ANN search is navigating through HNSW graph and collect oversampled candidates, then pass them to `doRescore`.
But since for the POC, the function is replaced to call `rerank` in FaissMemoryOptimizedSearcher, we should manually implement rerank for two versions 
to compare the performance.

# Benchmark Logic
1. Replicate FaissBBQRecallValidationTests to ingest and build bbq files (.veb, vemb) and .faiss file.
2. Then, create two methods
  - bbq rerank
    - Do rerank for N times for warm-up. (N is 100 by default).
    - Rerank via FaissMemoryOptimizedSearcher, and collect 50%, 90%, 95%, 99%, 99.9% latency.
  - full precision rerank
      - Do rerank for N times for warm-up. (N is 100 by default).
      - Rerank via VectorReader, and collect 50%, 90%, 95%, 99%, 99.9% latency.
        - It should get full precision vector from `vectors`, and use VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT.compare to get score.
3. Replicating the ANN search logic using FaissMemoryOptimizedSearcher to get TopDocs.
4. Pass TopDocs and VectorReader to each benchmark method to let them run rerank.
