# Search path modification Idea
To use error residual to refine scores from 1st phase search, we should expand search path.

1. NativeEngineKnnVectorQuery.createWeight
2. NativeEngineKnnVectorQuery.doRescore
3. If compression is x32 then doRescoreWithErrorRefinement (new method)
4. We have to split documents into segments. i.e. documents per each segment
5. We should get KnnVectorsReader with ((PerFieldKnnVectorsFormat.FieldsReader) segmentReader.getVectorReader()).getFieldReader(
   field.getName()).
6. And we should check the reader is instance of ErrorResidualRefiner (new interface).
7. if knn reader is instance of ErrorResidualRefiner, then we should call ErrorResidualRefiner.refine method.
  - We should pass document ids + scores. Then refine will return ScoreDoc[] having refined scores
8. The knn vectors reader is Faiss1040ScalarQuantizedKnnVectorsReader. It will implement ErrorResidualRefiner.
9. Faiss1040ScalarQuantizedKnnVectorsReader will load .ver file as IndexInput.
10. During refinement, it will clone IndexInput and use it for score refinement.
  - You can use KNN1040ScalarQuantizedUtils.extractQuantizedByteVectorValues to get QuantizedByteVectorValues.
  - From QuantizedByteVectorValues, you can get centroid.
  - This is critical part that determines search performance. We should leverage C++ SIMD and prefetch.
    - For prefetch, see PrefetchHelper.prefetch.
    - For C++, see ArmNeonSQSimilarityFunction in /Users/kdooyong/workspace/error-correction-poc/jni/src/simd/similarity_function/arm_neon_simd_similarity_function.cpp.
11. Merge TopDocs and cut top-k, we can use TopKnnCollector for this purpose.
12. Return top-k results.
