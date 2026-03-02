# Problem definition
Currently, we're passing 32x quantized vectors with additional correction factors to C++ layer to build HNSW.
If you see getRandomVectorScorer method in 'Lucene102BinaryFlatVectorsScorer', it's quantizing query vector into 4bits (therefore 8x quantization).

```java
      OptimizedScalarQuantizer.QuantizationResult queryCorrections =
          quantizer.scalarQuantize(target, initial, (byte) 4, centroid);
      transposeHalfByte(initial, quantized);
```

Then in quantizedScore method, it's doing ADC between 4bits vector and 1bit vector as following
```java
    float qcDist = VectorUtil.int4BitDotProduct(quantizedQuery, binaryCode);
    OptimizedScalarQuantizer.QuantizationResult indexCorrections =
        targetVectors.getCorrectiveTerms(targetOrd);
```

This is what we need in MemOptimizedBBQIndexBuildStrategy for better recall!
We first need to pass document id and float[] in FaissService.addDocsToBBQIndex.
Then, we should get jflaot* pointer from the given float[], and pass it to `idMap->add_with_ids(numDocs, vecPtr, &docIds[0])`;
Then, we should put 4bits query vector quantization logic in BBQDistanceComputer's set_query method.
Basically, below logic should be put in the method.
Note that we can update the query float[], it's fine.
```
      float[] centroid = binarizedVectors.getCentroid();
      // We make a copy as the quantization process mutates the input
      byte[] initial = new byte[target.length];
      byte[] quantized = new byte[QUERY_BITS * binarizedVectors.discretizedDimensions() / 8];
      OptimizedScalarQuantizer.QuantizationResult queryCorrections =
          quantizer.scalarQuantize(target, initial, (byte) 4, centroid);
      transposeHalfByte(initial, quantized);
```

Then, we need to update all scoring methods.
All the logic should be the same, but we need to do dot product between 4bit vectors and 1bit vector.
  - operator()
  - distance_batch_4

```
    // dot product between 4bits query vector and 1bit data vector.
    float qcDist = VectorUtil.int4BitDotProduct(quantizedQuery, binaryCode);
```

Note that we don't need to change anything in symmetric_dis since this is distance calculation between data vectors.
