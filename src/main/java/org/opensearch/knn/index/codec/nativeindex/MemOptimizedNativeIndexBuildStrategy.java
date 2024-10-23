/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.codec.transfer.OffHeapVectorTransfer;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.jni.FaissService;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.jni.NmslibService;

import java.io.IOException;
import java.nio.file.Path;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.opensearch.knn.common.KNNVectorUtil.intListToArray;
import static org.opensearch.knn.common.KNNVectorUtil.iterateVectorValuesOnce;
import static org.opensearch.knn.index.codec.transfer.OffHeapVectorTransferFactory.getVectorTransfer;

/**
 * Iteratively builds the index. Iterative builds are memory optimized as it does not require all vectors
 * to be transferred. It transfers vectors in small batches, builds index and can clear the offheap space where
 * the vectors were transferred
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
final class MemOptimizedNativeIndexBuildStrategy implements NativeIndexBuildStrategy {

    private static MemOptimizedNativeIndexBuildStrategy INSTANCE = new MemOptimizedNativeIndexBuildStrategy();

    public static MemOptimizedNativeIndexBuildStrategy getInstance() {
        return INSTANCE;
    }

    /**
     * Builds and writes a k-NN index using the provided vector values and index parameters. This method handles both
     * quantized and non-quantized vectors, transferring them off-heap before building the index using native JNI services.
     *
     * <p>The method first iterates over the vector values to calculate the necessary bytes per vector. If quantization is
     * enabled, the vectors are quantized before being transferred off-heap. Once all vectors are transferred, they are
     * flushed and used to build the index. The index is then written to the specified path using JNI calls.</p>
     *
     * @param indexInfo        The {@link BuildIndexParams} containing the parameters and configuration for building the index.
     * @throws IOException     If an I/O error occurs during the process of building and writing the index.
     */
    public void buildAndWriteIndex(final BuildIndexParams indexInfo) throws IOException {
        final KNNVectorValues<?> knnVectorValues = indexInfo.getVectorValues();
        // Needed to make sure we don't get 0 dimensions while initializing index
        iterateVectorValuesOnce(knnVectorValues);
        KNNEngine engine = indexInfo.getKnnEngine();
        Map<String, Object> indexParameters = indexInfo.getParameters();
        IndexBuildSetup indexBuildSetup = QuantizationIndexUtils.prepareIndexBuild(knnVectorValues, indexInfo);

        // Initialize the index
        long indexMemoryAddress = AccessController.doPrivileged(
            (PrivilegedAction<Long>) () -> JNIService.initIndex(
                indexInfo.getTotalLiveDocs(),
                indexBuildSetup.getDimensions(),
                indexParameters,
                engine
            )
        );

        try (
            final OffHeapVectorTransfer vectorTransfer = getVectorTransfer(
                indexInfo.getVectorDataType(),
                indexBuildSetup.getBytesPerVector(),
                indexInfo.getTotalLiveDocs()
            )
        ) {

            final List<Integer> transferredDocIds = new ArrayList<>(vectorTransfer.getTransferLimit());

            while (knnVectorValues.docId() != NO_MORE_DOCS) {
                Object vector = QuantizationIndexUtils.processAndReturnVector(knnVectorValues, indexBuildSetup);
                // append is false to be able to reuse the memory location
                boolean transferred = vectorTransfer.transfer(vector, false);
                transferredDocIds.add(knnVectorValues.docId());
                if (transferred) {
                    // Insert vectors
                    long vectorAddress = vectorTransfer.getVectorAddress();
                    AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
                        JNIService.insertToIndex(
                            intListToArray(transferredDocIds),
                            vectorAddress,
                            indexBuildSetup.getDimensions(),
                            indexParameters,
                            indexMemoryAddress,
                            engine
                        );
                        return null;
                    });
                    transferredDocIds.clear();
                }
                knnVectorValues.nextDoc();
            }

            boolean flush = vectorTransfer.flush(false);
            // Need to make sure that the flushed vectors are indexed
            if (flush) {
                long vectorAddress = vectorTransfer.getVectorAddress();
                AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
                    JNIService.insertToIndex(
                        intListToArray(transferredDocIds),
                        vectorAddress,
                        indexBuildSetup.getDimensions(),
                        indexParameters,
                        indexMemoryAddress,
                        engine
                    );
                    return null;
                });
                transferredDocIds.clear();
            }

            // Write vector
            AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
                JNIService.writeIndex(indexInfo.getIndexOutputWithBuffer(), indexMemoryAddress, engine, indexParameters);
                return null;
            });

        } catch (Exception exception) {
            throw new RuntimeException(
                "Failed to build index, field name [" + indexInfo.getFieldName() + "], parameters " + indexInfo,
                exception
            );
        }
    }

    // TMP
    public static void main(String... args) throws IOException {
        //
        // FAISS
        //

        final String dataPath = "/Users/kdooyong/tmp/writing-layer/print_perf/data.json";
        final String tmpDirectory = "tmp-" + UUID.randomUUID();
        final Directory directory = new MMapDirectory(Path.of(tmpDirectory));
        final int numData = 10000;
        final int dim = 128;
        Map<String, Object> parameters = new HashMap<>();
        /*
            {
               name=hnsw,
               data_type=float,
               index_description=HNSW16, flat,
               spaceType=l2,
               parameters={ef_search=100, ef_construction=100, encoder={name=flat, parameters={}}},
               indexThreadQty=1
            }
         */

        parameters.put("name", "hnsw");
        parameters.put("data_type", "float");
        parameters.put("index_description", "HNSW16,Flat");
        parameters.put("spaceType", "l2");

        Map<String, Object> innerParameters = new HashMap<>();
        innerParameters.put("ef_search", 10);
        innerParameters.put("ef_construction", 100);

        Map<String, Object> encoderParameters = new HashMap<>();
        encoderParameters.put("name", "flat");
        encoderParameters.put("parameters", Collections.emptyMap());
        innerParameters.put("encoder", encoderParameters);
        parameters.put("parameters", innerParameters);

        parameters.put("indexThreadQty", 1);

        final String fullPath = tmpDirectory + "/output";

        try (final IndexOutput indexOuptut = directory.createOutput("output", IOContext.DEFAULT)) {
            System.out.println("Output : " + tmpDirectory + "/output");
            IndexOutputWithBuffer indexOutputWithBuffer = new IndexOutputWithBuffer(indexOuptut);
            FaissService.kdyBench(numData, dim, dataPath, parameters, indexOutputWithBuffer, fullPath);
//            NmslibService.kdyBench(numData, dim, dataPath, parameters, indexOutputWithBuffer);
        }
        System.out.println("OUT!!!!!!!!");
    }

//    public static void main(String... args) throws IOException {
//        //
//        // NMSLIB
//        //
//
//        final String dataPath = "/Users/kdooyong/tmp/writing-layer/print_perf/data.json";
//        final String tmpDirectory = "tmp-" + UUID.randomUUID();
//        final Directory directory = new MMapDirectory(Path.of(tmpDirectory));
//        final int numData = 10000;
//        final int[] ids = new int[numData];
//        for (int i = 0 ; i < numData ; ++i) {
//            ids[i] = i;
//        }
//        final int dim = 128;
//        Map<String, Object> parameters = new HashMap<>();
//
//        /*
//         * ################# Key: [name]: [hnsw]
//         * ################# Key: [data_type]: [float]
//         * ################# Key: [spaceType]: [l2]
//         * ################# Key: [parameters]: [{ef_construction=100, m=16}]
//         * ################# Key: [indexThreadQty]: [1]
//         */
//
//        parameters.put("name", "hnsw");
//        parameters.put("data_type", "float");
//        parameters.put("spaceType", "l2");
//
//        Map<String, Object> innerParameters = new HashMap<>();
//        innerParameters.put("ef_construction", 100);
//        innerParameters.put("m", 16);
//        parameters.put("parameters", innerParameters);
//
//        parameters.put("indexThreadQty", 1);
//
//        final String fullPath = tmpDirectory + "/output";
//
//        try (final IndexOutput indexOuptut = directory.createOutput("output", IOContext.DEFAULT)) {
//            System.out.println("Output : " + tmpDirectory + "/output");
//            IndexOutputWithBuffer indexOutputWithBuffer = new IndexOutputWithBuffer(indexOuptut);
//            NmslibService.kdyBench(numData, dim, dataPath, ids, parameters, indexOutputWithBuffer, fullPath);
//        } catch (Exception e) {
//            System.out.println("===================");
//            e.printStackTrace();
//            System.out.println("===================");
//        }
//        System.out.println("OUT!!!!!!!!");
//    }
    // TMP
}
