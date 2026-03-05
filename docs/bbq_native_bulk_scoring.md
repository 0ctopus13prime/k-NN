# Goal
IT'S SLOW! That's all.
I'm planning to deprecate hamming distance based BQ scoring to BBQ which uses `quantizedScore` method in Lucene102BinaryFlatVectorsScorer.
But it's slow! I know, since it's using ADC (4 bit query and 1 bit data vectors), which naturally slower than BQ which only using XOR between bit vectors.

Therefore, I'm seeking to offload scoring function to native C++ layer to score multiple vectors at the same time.

# Phase-1 : Make it work, naive C++
I've already done some foundation works.
If MMap is being used, then it will directly extract mapped pointer from IndexInput. (See MemorySegmentAddressExtractorJDK22 for details)
See OffHeapBinarizedVectorValues.load for details. Note that this is for POC, and it only cares dense case for now.
Then, BBQNativeRandomVectorScorer will be created which offload the distance calculation to C++ via SimdVectorComputeService.

## JNI foundation for SIMD
1. Save context
See JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_saveSearchContext in /Users/kdooyong/workspace/io-opt/jni/src/org_opensearch_knn_jni_SimdVectorComputeService.cpp.
We should save query and pointers in somewhere for one transaction at the beginning.

SimdVectorSearchContext* SimilarityFunction::saveSearchContext is the crucial part where selecting similarity function based on the nativeFunctionTypeOrd provided.
(See /Users/kdooyong/workspace/io-opt/jni/src/simd/similarity_function/simd_similarity_function_common.cpp)
In there, you will find that it's calling 'selectSimilarityFunction' to get the function pointer, which then put it into search context.
(See /Users/kdooyong/workspace/io-opt/jni/src/simd/similarity_function/default_simd_similarity_function.cpp for default non-optimized version)

2. Bulk Score / Scoring
Then, we only pass vector ordinals to C++ layer. 
See Java_org_opensearch_knn_jni_SimdVectorComputeService_scoreSimilarityInBulk /Users/kdooyong/workspace/io-opt/jni/src/org_opensearch_knn_jni_SimdVectorComputeService.cpp.
Since it has the pointers and other info (e.g. one vector size etc), it can directly jump on the data vector in memory, and perform distance calculation.

- JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_scoreSimilarityInBulk
- JNIEXPORT jfloat JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_scoreSimilarity

Above functions are actually calling scoring function that was acquired in saveContext.
In there, it first gets float*[] with vector ordinals, and proceed bulk distance calculation.
See /Users/kdooyong/workspace/io-opt/jni/src/simd/similarity_function/arm_neon_simd_similarity_function.cpp for NEON SIMD bulk distance calculation, you will get the idea.

## Implementation plan phase-1 : Naive BBQ scoring.
We only focus on NEON SIMD, therefore, we should put our implementation in /Users/kdooyong/workspace/io-opt/jni/src/simd/similarity_function/arm_neon_simd_similarity_function.cpp.
Note that in phase-1, I'm aiming at 'making it work in C++', and not targetting any SIMD optimizations. Which will be dealt with in later phase.

Missing pieces are
- SimdVectorComputeService.saveBBQSearchContext JNI is missing.
    - In there, we should save quantized query vector, and 4 correction factors and byteSize (e.g. sizeof quantized vectors + 4 * 4)
        - You can copy the quantized vectors to queryVectorSimdAligned.
        - And put 4 correction factors to tmpBuffer. Then during search, it can get lowerInterval as (float*) tmpBuffer[0]. Note that tmpBuffer is using FourBytesAlignedAllocator, it's safe to cast uint8_t* to float*.
        - You can get the sizeof quantized vector with discretize(dimension, 64) / 8 where dimension is already stored in context. See discretize below.
          - public static int discretize(int value, int bucket) {
              return (value + (bucket - 1)) / bucket * bucket;
            }
          - then, put the byte size into roneVectorByteSize.

- Add bbq bulk simd function in `SimilarityFunction* SimilarityFunction::selectSimilarityFunction(const NativeSimilarityFunctionType nativeFunctionType)`
  - Create a dedicated bulk distance calculator for BBQ, and return it when the type is BBQ_IP.
- BBQ calculator in /Users/kdooyong/workspace/io-opt/jni/src/simd/similarity_function/arm_neon_simd_similarity_function.cpp.
  - Let's start non-simd fashion. You can get the scoring idea in /Users/kdooyong/workspace/io-opt/jni/include/faiss_bbq_flat.h.
  - So the idea is that, you get the bulk uint8_t* pointers pointing to data vectors via SimdVectorSearchContext::getVectorPointersInBulk.
  - Then, apply the same logic in faiss_bbq_flat.h.
