# Project outline
This project is to use Lucene BBQ to build Faiss index, and search on it during runtime.
Currently, opensearch supports BQ (binary quantization) for 32x vector quantization.
However, Lucene BBQ performs better than BQ in terms of performance and recall.
Thus, I'm trying to build Faiss index with Lucene BBQ algorithm, more specifically, use Lucene BBQ algorithm to quantize
vectors, then use Lucene BBQ scoring function to build Faiss HNSW graph.

Note that this is POC level, therefore production level code quality is not required.
But I should showcase Lucene BBQ integration on Faiss index should provide better recall and performance than BQ.

In this POC, we assume that it uses inner product metric, and will have 768 dimension data vectors. 
You might find very hacky way to make it happen, but that's ok. This aims at proof of concept that BBQ on Faiss 
does provide better results that what it has.

# Lucene BBQ
For better understanding of Lucene BBQ, refer to org.opensearch.knn.index.codec.nativeindex.bbq package.
BBQWriter is the one that 32x quantizes vectors. BBQReader, namely, it's the one that performs loading quantized vectors
and scoring those with a full precision query vector using Lucene102BinaryFlatVectorsScorer.


# Implementation
## Build HNSW
Long story short, NativeIndexWriter.buildAndWriteIndex is the starting point.
In there, using vector stream (i.e. `knnVectorValuesSupplier`) to 32x quantize vectors to produce .veb and .vemb files.
Then, in MemOptimizedBBQIndexBuildStrategy, it passs quantized vectors to JNI layer which uses custom Faiss Index
implementation (FaissBBQFlat, see jni/include/faiss_bbq_flag.h file) to build HNSW graph structure with the exactly same
scoring function in Lucene102BinaryFlatVectorsScorer. (see Lucene102BinaryFlatVectorsScorer.quantizedScore method)

First the strategy initializes BBQ index. (Start explore from jni/src/org_opensearch_knn_jni_FaissService.cpp)
By initializing BBQ index, it pre-allocate binary std::vector. Note that each vector will be quantized into a binary 
vector along with 4 correction factors. (lower bound, upper bound, additional correction factor, centroid dot product)
So the layout is binary vector (96 bytes) which is followed by 3 floats and 1 integer.
Anyway, it pre-allocates bytes for binary vector with N * 112 bytes. (see jni/src/faiss_index_service.cpp, BinaryIndexService::initBBQIndex method)

Then, by reading all quantized vectors + corresponding correction factors, pass them to the pre-allocated space.
See passQuantizedVectorsCorrectionFactors method for Java data transfer code, and see Java_org_opensearch_knn_jni_FaissService_passBBQVectors function
in jni/src/org_opensearch_knn_jni_FaissService.cpp.

After that, by iterating through the vector stream, pass document ids to JNI layer, which then bypass that to Faiss add_with_ids.
In Faiss, see (jni/external/faiss/faiss/IndexIDMap.cpp, add_with_ids method).
At this time, the top level faiss index is IndexIdMap which has IndexHNSWFlat as a nested index, which in turn has FaissBBQFlat defined in opensearch JNI side.
Note that, since we already have the binary codes + correction terms under FaissBBQFlat::quantizedVectorsAndCorrectionFactors,
we only pass the pointer that points to the staring vector that was not yet added to HNSW. (see Java_org_opensearch_knn_jni_FaissService_addDocsToBBQIndex in jni/src/org_opensearch_knn_jni_FaissService.cpp)
Ultimately, jni/external/faiss/faiss/IndexHNSW.cpp is the core logic for HNSW building. (see hnsw_add_vertices function)
It first gets distance computer from storage (which is FaissBBQFlat in opensearch side), then set the query (which is also from FaissBBQFlat::quantizedVectorsAndCorrectionFactors),
and let it returns distance with vector ordinals. (see BBQDistanceComputer in jni/include/faiss_bbq_flat.h)
Note that I've hardcoded with 112 as element size as followed:
dis->set_query((const float*) (((const uint8_t*) x) + (pt_id - n0) * 112));

After built HNSW graph, calling writeBBQIndex to flush in-memory data structure to disk.
At this time, it's passing io_flags=1 which tell Faiss not to persist the bottom layer FaissBBQFlat as we already have 
them persisted in .veb file. see jni/external/faiss/faiss/impl/index_write.cpp.
Skip code is below:
        if (io_flags & IO_FLAG_SKIP_STORAGE) {
            uint32_t n4 = fourcc("null");
            WRITE1(n4);
        } else {
            write_index(idxhnsw->storage, f);
        }

That's all. Even for the merging segments, it's rebuilding the graph from scratch, so it follows the same steps outlined
above.

## Search
Search happens in 'FaissMemoryOptimizedSearcher' Java file.
It gets the scorer directly from FaissBBQFLat java instance. In which, it delegates the request down to BBQReader to get FlatVectorsReader.
Actual loading logic resides in FaissBBQFLat.tempDoLoadManually where it simply delegates loading to BBQReader.

That's all! once we have scorer and Faiss HNSW then Lucene's HNSW graphs searcher will perform search on the graph.
During the HNSW navigation, it uses Lucene102BinaryFlatVectorsScorer.quantizedScore to perform ADC search with a query vector.

# Recall Issue
Recall is almost 10%. BQ is showing 86% for 10,000 - 100,000 dataset on the other hand.
I must be doing something wrong in somewhere.

