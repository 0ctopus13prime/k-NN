# Goal
Even with all the fixes, I'm still seeing low recall 1% while BQ is showing 86% recall for 10K dataset.
I must be doing something wrong in somewhere, so we should do the comparison recall between, to root cause
what exactly has cause the recall distortion.

# Implementation
We should make three different program as unit test format, so that I can easily run each in my Intellij.
The folder location is '/Users/kdooyong/workspace/io-opt/src/test/java/org/opensearch/knn/memoryoptsearch/bbq'
Each program simply take the data set path, and top-k number and vector ordinal used for query vector.
With full scan on quantized vectors, then it should take top-k and print pairs of vector ordinal and score value.
I'll handle the output by redirecting to a file, which will then to be passed to comparing program later.
In this project, it should focus on getting top-k results with score values, that's it.

Note that each test file should end with 'Tests', and test method name should start with 'test'.
Also DO NOT USE 'var'

# Input vector data set format
Each line will have float values with a comma delimiter. All vectors will have the same dimension.
For example, a vector with 8 dimension will be put in line as following:
-1.2, 2, 3, 0.1, -0.11, 7, 5.5, 0.123

Therefore, rather than implementing a function loading vectors in each program, we should make it as a common util class.

We need this function
float[][] loadVectors(String inputDataPath) {
    // load vectors
}

## Implementation details
In the set-up, we prepare three programs to print top-100 results on three cases.

1. Lucene .veb:
It should make a temp directory under /tmp, and read vectors from the input parameter first.
Then, by replicating NativeIndexWriter.writeBBQ to do 32x quantization on input vectors.
And replicating MemOptimizedBBQIndexBuildStrategy.buildAndWriteIndex method to load quantized vectors with BBQReader, 
and pick a query vector and take top-k.

One thing I want is in the program 1 is that, which is critical, you should add implementation in getRandomVectorScorer to take quantized vectors and just do bit product with '&' operator instead of calling VectorUtil.int4BitDotProduct. The method is being called in search, and I want to simulate bit vector to bit vector recall here. 

So, pick a quantized vector from .veb, and it should get the score via bit vector dot product, not VectorUtil.int4BitDotProduct.

Also note that quantized vector is 8 byte aligned.

2. Use BBQDistanceComputer:
Equally, create a temp directory and load vectors.
We first need to ingest vectors by replicating NativeIndexWriter.writeBBQ and MemOptimizedBBQIndexBuildStrategy to have FaissBBQFlat instance in memory,
then pick xth quantized vector and do scanning to take top-k.
It's a bit complicated, since it has a dependency on Lucene's .veb file, so we need to ingest via BBQWriter first (which is covered in NativeIndexWriter.writeBBQ)
For this, we probably need to define a custom function in FaissService. The reason being, FaissBBQFlat is ephemeral, and it will be gone after indexing.
Therefore, we should put the validation logic in JNI side, and in Java we simply pass parameters; top-k, query vector ordinal.

So the expected outputs are below:
- Java program in unit test form
- FaissService new method taking top-k, query vector ordinal
- New signature in /Users/kdooyong/workspace/io-opt/jni/include/org_opensearch_knn_jni_FaissService.h
- Validation logic in /Users/kdooyong/workspace/io-opt/jni/src/org_opensearch_knn_jni_FaissService.cpp, taking query vector and get the top-k and print vector ordinal + score

3. Use BQ.
Do create temp directory + loading vectors.
We should train vectors with OneBitScalarQuantizer via QuantizationService.train to get QuantizationState.
Then, we can get quantized vectors via OneBitScalarQuantizer.quantize method to collect them in List simply.
With all quantized vectors, then we can do the same thing pick xth vector and get the top-k results by scanning.
When calculating hamming distance, you can simply use KNNVectorSimilarityFunction.HAMMING.compare method, then sort score by descending order