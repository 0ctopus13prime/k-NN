/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.bbq;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.IndexBuildSetup;
import org.opensearch.knn.index.codec.nativeindex.QuantizationIndexUtils;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.codec.transfer.OffHeapVectorTransfer;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesIterator;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.INDEX_THREAD_QTY;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.common.KNNVectorUtil.intListToArray;
import static org.opensearch.knn.index.codec.transfer.OffHeapVectorTransferFactory.getVectorTransfer;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

/**
 * Measures the time taken to build a 32x (ONE_BIT) quantized binary HNSW index,
 * replicating the exact AS-IS MemOptimizedNativeIndexBuildStrategy.buildAndWriteIndex logic.
 */
public class BQ32xIndexBuildTimingTests extends KNNTestCase {

    private static final String DATA_PATH = "/Users/kdooyong/workspace/io-opt/tmp/vectors.bin";

    public void testBQ32xIndexBuildTime() throws Exception {
        float[][] vectors = VectorDataLoader.loadVectors(DATA_PATH);
        int dimension = vectors[0].length;
        int numVectors = vectors.length;
        System.out.println("Loaded " + numVectors + " vectors with dimension " + dimension);

        // Dense doc IDs: 0..numVectors-1
        List<Integer> documentIds = new ArrayList<>(numVectors);
        for (int i = 0; i < numVectors; i++) {
            documentIds.add(i);
        }
        List<float[]> floatVectors = new ArrayList<>(numVectors);
        for (float[] v : vectors) {
            floatVectors.add(v);
        }

        // Step 1: Train quantizer
        ScalarQuantizationParams quantizationParams = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();

        System.out.println("Training quantizer...");
        long trainStart = System.currentTimeMillis();
        QuantizationState quantizationState = QuantizationService.getInstance()
            .train(quantizationParams, () -> (KNNVectorValues) createKNNFloatVectorValues(documentIds, floatVectors), documentIds.size());
        long trainEnd = System.currentTimeMillis();
        System.out.println("Training took " + (trainEnd - trainStart) + " ms");

        // Step 2: Build parameters (matching as-is path)
        Map<String, Object> parameters = new HashMap<>();
        parameters.put(NAME, METHOD_HNSW);
        parameters.put(VECTOR_DATA_TYPE_FIELD, VectorDataType.BINARY.getValue());
        parameters.put(SPACE_TYPE, "innerproduct");
        parameters.put(INDEX_THREAD_QTY, 1);
        parameters.put(INDEX_DESCRIPTION_PARAMETER, "BHNSW16,Flat");

        Map<String, Object> methodParameters = new HashMap<>();
        parameters.put(PARAMETERS, methodParameters);
        methodParameters.put(METHOD_PARAMETER_EF_SEARCH, 256);
        methodParameters.put(METHOD_PARAMETER_EF_CONSTRUCTION, 256);
        methodParameters.put(METHOD_ENCODER_PARAMETER, Map.of());

        KNNEngine engine = KNNEngine.FAISS;

        // Step 3: Replicate buildAndWriteIndex exactly
        Path tmpDir = Files.createTempDirectory("bq32x_timing");
        try (Directory directory = new MMapDirectory(tmpDir)) {
            String fileName = "bq32x_test.faiss";
            try (IndexOutput indexOutput = directory.createOutput(fileName, IOContext.DEFAULT)) {
                IndexOutputWithBuffer outputWithBuffer = new IndexOutputWithBuffer(indexOutput);

                // Create KNNVectorValues (fresh iterator for the build)
                KNNVectorValues<?> knnVectorValues = createKNNFloatVectorValues(documentIds, floatVectors);
                initializeVectorValues(knnVectorValues);

                // Prepare index build setup (same as QuantizationIndexUtils.prepareIndexBuild)
                BuildIndexParams indexInfo = BuildIndexParams.builder()
                    .fieldName("test_field")
                    .knnEngine(engine)
                    .vectorDataType(VectorDataType.BINARY)
                    .indexOutputWithBuffer(outputWithBuffer)
                    .parameters(parameters)
                    .quantizationState(quantizationState)
                    .knnVectorValuesSupplier(() -> createKNNFloatVectorValues(documentIds, floatVectors))
                    .totalLiveDocs(numVectors)
                    .build();

                IndexBuildSetup indexBuildSetup = QuantizationIndexUtils.prepareIndexBuild(knnVectorValues, indexInfo);

                System.out.println("Starting index build...");
                long buildStart = System.currentTimeMillis();

                // Initialize the index (same as as-is)
                long indexMemoryAddress = JNIService.initIndex(numVectors, indexBuildSetup.getDimensions(), parameters, engine);

                // Transfer and insert vectors using OffHeapVectorTransfer (same as as-is)
                try (
                    final OffHeapVectorTransfer vectorTransfer = getVectorTransfer(
                        VectorDataType.BINARY,
                        indexBuildSetup.getBytesPerVector(),
                        numVectors
                    )
                ) {
                    final List<Integer> transferredDocIds = new ArrayList<>(vectorTransfer.getTransferLimit());

                    while (knnVectorValues.docId() != NO_MORE_DOCS) {
                        Object vector = QuantizationIndexUtils.processAndReturnVector(knnVectorValues, indexBuildSetup);
                        boolean transferred = vectorTransfer.transfer(vector, false);
                        transferredDocIds.add(knnVectorValues.docId());
                        if (transferred) {
                            long vectorAddress = vectorTransfer.getVectorAddress();
                            JNIService.insertToIndex(
                                intListToArray(transferredDocIds),
                                vectorAddress,
                                indexBuildSetup.getDimensions(),
                                parameters,
                                indexMemoryAddress,
                                engine
                            );
                            transferredDocIds.clear();
                        }
                        knnVectorValues.nextDoc();
                    }

                    boolean flush = vectorTransfer.flush(false);
                    if (flush) {
                        long vectorAddress = vectorTransfer.getVectorAddress();
                        JNIService.insertToIndex(
                            intListToArray(transferredDocIds),
                            vectorAddress,
                            indexBuildSetup.getDimensions(),
                            parameters,
                            indexMemoryAddress,
                            engine
                        );
                        transferredDocIds.clear();
                    }

                    long insertEnd = System.currentTimeMillis();
                    System.out.println("Quantize + insert took " + (insertEnd - buildStart) + " ms");

                    // Write index
                    JNIService.writeIndex(outputWithBuffer, indexMemoryAddress, engine, parameters);

                    long writeEnd = System.currentTimeMillis();
                    System.out.println("Write index took " + (writeEnd - insertEnd) + " ms");
                    System.out.println("Total build time (init + quantize + insert + write): " + (writeEnd - buildStart) + " ms");
                }
            }
        }
    }

    @SuppressWarnings("unchecked")
    private static KNNFloatVectorValues createKNNFloatVectorValues(final List<Integer> documentIds, final List<float[]> vectors) {
        final FloatVectorValues knnVectorValues = new FloatVectorValues() {
            @Override
            public int dimension() {
                return vectors.get(0).length;
            }

            @Override
            public int size() {
                return vectors.size();
            }

            @Override
            public DocIndexIterator iterator() {
                return new DocIndexIterator() {
                    int doc = -1;

                    @Override
                    public int index() {
                        return doc;
                    }

                    @Override
                    public int docID() {
                        return doc;
                    }

                    @Override
                    public int nextDoc() throws IOException {
                        if (doc == NO_MORE_DOCS) {
                            return doc;
                        }

                        ++doc;
                        if (doc >= vectors.size()) {
                            return doc = NO_MORE_DOCS;
                        }
                        return doc;
                    }

                    @Override
                    public int advance(int i) throws IOException {
                        doc = i;
                        if (doc >= vectors.size()) {
                            return doc = NO_MORE_DOCS;
                        }
                        return i;
                    }

                    @Override
                    public long cost() {
                        return vectors.size();
                    }
                };
            }

            @Override
            public float[] vectorValue(int i) throws IOException {
                return vectors.get(i);
            }

            @Override
            public FloatVectorValues copy() throws IOException {
                throw new UnsupportedOperationException("Not supported");
            }

            @Override
            public VectorEncoding getEncoding() {
                return VectorEncoding.FLOAT32;
            }
        };

        final KNNVectorValuesIterator.DocIdsIteratorValues docIdsIteratorValues =
            new KNNVectorValuesIterator.DocIdsIteratorValues(knnVectorValues);

        return new KNNFloatVectorValues(docIdsIteratorValues);
    }
}
