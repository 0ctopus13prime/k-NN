# Problem definition
Currently, we're doing bbq quantization in NativeIndexWriter.writeBBQ.
In there, it's calling addValue to add full precision vector.
Added vectors then put into BBQWriter#FieldWriter#vectors, which stores vector in memory.

But we don't need to store it in memory.
The main reason for needing vectors folds into two:
1. To get cluster centroid.
2. For quantizing.

Therefore, we could read fp32 vector twice to get the centroid, and get the quantized vectors without having them in memory.
Which could alleviate memory pressure in JVM.

# 2phase Faiss bbq quantization
Modify NativeIndexWriter.writeBBQ to read fp32 vectors twice for each purpose.
For this, I think you roughly need to following:

1. Add mergeOneField method in BBQWriter taking FieldInfo and Supplier<KNNVectorValues<?>>.
2. In there, Create FieldWriter.
3. Get KNNVectorValues from the supplier.
4. Pull float[] from the KNNVectorValues.
5. Then add addValueForMerge in FieldWriter.
6. addValueForMerge method should not store float[] in memory. but it should just add to docsWithFieldSet and do summing dimensionSums
7. then replicate writeField for writeFieldForMerge.
8. We should also have writeBinarizedVectorsForMerge to take KNNVectorValues. The only difference is that, it is pulling float[] from
KNNVectorValues, not calling fieldData.getVectors() and remaining logic should be the same.
