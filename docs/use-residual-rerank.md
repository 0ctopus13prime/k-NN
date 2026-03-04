# Problem definition
Currently, we are doing 2-phase search for 32x quantized index.
We first oversample the candidates from HNSW graph (we expand `k` in here getFirstPassK in RescoreContext).
Collect candidates from HNSW graph (FaissMemoryOptimizedSearcher), then take top oversampled `k` from merged results from
segments, and do the rerank by loading full precision vectors. (see NativeEngineKnnVectorQuery#doRescore. Note that isShardLevelRescoringDisabled is false by default)

This is cost operation. Usually we map full precision vectors, but this is the place where page faults are frequently happening
which degrade the performance.

# POC idea - Use error residual for rerank.
In LVQ paper (Similarity search in the blink of an eye with compressed indices, https://www.vldb.org/pvldb/vol16/p3433-aguerrebere.pdf)
it first encodes error residual separately, then use it for rerank, see the definition and formula below:

> To reduce the effective memory bandwidth during search, we
> compress each vector in two levels, each with a fraction of the
> available bits. After using LVQ for the first level, we quantize the
> residual vector r = x - miu - Q(x) where miu is the mean value. The scalar random variable
> Z = x - miu - Q(x), which models the first-level quantization error,
> follows a uniform distribution in [delta/2, delta/2) (see Equation (1)).
> 
> Thus, we encode each component of r using the scalar quantization
> function
> Q_res(r; B') = Q(x; B', -delta/2, delta/2) where B' is the number of bits used for the residual code.
> Definition 2. We define the two-level Locally-adaptive Vector Quantization (LVQ-B_1 x B_2) of vector x as a pair of vectors &(x),&res(r),
> such that
> - Q(x) is the vector x compressed with LVQ-B_1
> - Q_res(x) = [Q_res(r1; B2), ..., Q_res(rd; B2)] where d is dimension of vector
> where r = x - miu - Q(x) and Q(x) is defined in Equation (6).

# POC Implementation Plan
## Phase 1: Store error residual
We have to store quantized error residual (i.e. when r = x - miu - Q(X), Q_res(r; B')) for two cases
1. flush (see writeField and writeBinarizedVectors in BBQWriter)
2. merge (see mergeOneField and writeFieldForMerge)

We can start B' as a 1-bit for now, but it should be parameterized as i'm planning to test higher bits.

C++ Considerations
1. During building, error residual will not be needed. Therefore, I think I can leave C++ implementation for this.
2. But, passQuantizedVectorsCorrectionFactors in MemOptimizedBBQIndexBuildStrategy should be changed to skip error residual part.
  - Original format is | quantized vector | 4 correction factors |, which not becomes | quantized vector | 4 correction factors | error residual quantized vector |

# Phase 2: Rerank with error residual.
To quickly test out rerank based on error residual, we can quickly do this.

## Required changes in codec layer
Note that <q, v> is dot product between vectors q and v.
For the fp32 inner product, you can use KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT.
In this POC, underlying KnnVectorValues type is always be OffHeapBinarizedVectorValues.

1. We should add the method `void initRerank(float[] queryVector)` in OffHeapBinarizedVectorValues to pre-compute <q, centroid>. (of course we need new member variables for query and <q, centroid>)
2. We should add the method `QuantizedErrorResidual getQuantizedErrorResidual(int node)` in OffHeapBinarizedVectorValues. It should jump to `targetOrd * byteSize + byteSize / 2`, and copy quantized error and 4 correction factors and return.
3. We should add the method `void rerank(int node)` in OffHeapBinarizedVectorValues, where calculating <q, centroid> + <q, Q_res(x)> where <q, centroid> is already pre-computed, and <q, Q_res(x)> is the one distance between query vector and error residual.
4. We need to add `void rerank(KnnVectorValues.DocIndexIterator iterator, Map<Integer, Float> docIdToScore, KnnCollector knnCollector)` method in VectorSearcher.
5. In FaissMemoryOptimizedSearcher which is the sole implementation of VectorSearcher, it should implement `rerank` method.
  - In there, it should get OffHeapBinarizedVectorValues from bbqFlat.getFloatValues() method. It's safe to cast it.
  - Then call `initRerank` and pass query vector.
  - And iterate the given iterator to get doc_id and vector ordinal.
  - Call `rerank` method and add <q, Q(x)> which can be acquired from doc_to_score mapping.
  - Put the score value into collector.
6. In NativeEngines990KnnVectorsReader, getFloatVectorValues should return RerankableKnnVectorValues.
  - RerankableKnnVectorValues should bypasses parameters to VectorSearcher. That's all
  - Currently, it's implementation is empty, need to fill few methods.


## Required changes in ExactSearcher
1. Pass perLeafeResult to ExactSearcher.ExactSearcherContext to build doc to score mapping. i.e. doc id -> <q, Q(x)>
2. In searchLeaf method in ExactSearcher, it should do followings:
  - Build doc to score map i.e. doc id -> <q, Q(x)>
  - Get conjunctioned disi with following:
    - final DocIdSetIterator matchedDocs = getMatchedDocsIterator(
          exactSearcherContext.getMatchedDocsIterator(),
          reader.getFloatVectorValues(...).iterator()
      );
  - Cast `matchedDocs` to KnnVectorValues.DocIndexIterator.
  - Get FloatVectorValues from vectorReader, whose type is RerankableKnnVectorValues.
  - Prepare TopKnnCollector
  - Call rerank, and convert results to TopDocs.



















