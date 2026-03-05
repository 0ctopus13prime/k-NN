# Goal
Lucene102BinaryFlatVectorsScorer is using BBQNativeRandomVectorScorer to offload bbq scoring in C++ side.
But I'm seeing recall drop to 3%, without the native offload I was getting 99%, definitely we're doing something wrong.
Therefore, we need to make a test case and compare the recall.

# Implementation
We should add a test under /Users/kdooyong/workspace/io-opt/src/test/java/org/opensearch/knn/memoryoptsearch/bbq.
Basically, we should replicate FaissBBQRecallValidationTests to build a Lucene bbq files + .faiss file, then 
get two random vector scorer from Lucene102BinaryFlatVectorsScorer.getRandomVectorScorer and compare the score values.
The caveat is that when Lucene102BinaryFlatVectorsScorer.USE_NATIVE is true, it will return a scorer using C++, 
otherwise it will return a scorer using Lucene's Java implementation, which is the baseline.

So, replicating FaissBBQRecallValidationTests to build the index, then pick a vector and several vectors and call bulkScore to get score values.
Since it's dense case, you can pick any vector ordinals in [0, numDocs).
