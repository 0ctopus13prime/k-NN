/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.kdy;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsReader;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.opensearch.kdy.SegmentIdExtractor;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.MemOptimizedNativeIndexBuildStrategy;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

public class BQHnswTransformerTests extends KNNTestCase {
    public record TargetFiles(String faissIndexFileName, String flatVectorDataFileName, String flatVectorMetaFileName,
                              String engineLuceneDirectory) {}

    @SneakyThrows
    public void test_gogogo() {
        final String dataDir = "/Users/kdooyong/workspace/io-opt/build/testclusters/integTest-0/data";
        final List<TargetFiles> targetFilesList = findTargetFiles(Path.of(dataDir));

        // Get the number of available logical cores
        int coreCount = Runtime.getRuntime().availableProcessors();

        // Initialize the pool with that count
        final ExecutorService executor = Executors.newFixedThreadPool(coreCount);

        try {
            final double numTargetFiles = targetFilesList.size();
            final AtomicInteger done = new AtomicInteger();
            final List<Future> futures = new ArrayList<>();

            for (final TargetFiles targetFiles : targetFilesList) {
                System.out.println();
                System.out.println("Start replacing...");
                System.out.println("  Lucene dir: " + targetFiles.engineLuceneDirectory);
                System.out.println("  Vec: " + targetFiles.flatVectorDataFileName);
                System.out.println("  VecMeta: " + targetFiles.flatVectorMetaFileName);
                System.out.println("  Faiss: " + targetFiles.faissIndexFileName);
                final Future fut = executor.submit(() -> {
                    try {
                        replaceToBetterHNSW(targetFiles);
                        final int cnt = done.incrementAndGet();
                        System.out.println("Done " + (cnt / numTargetFiles) * 100 + "% ...");
                    } catch (IOException e) {
                        e.printStackTrace();
                        throw new RuntimeException(e);
                    }
                });
                futures.add(fut);
            }

            for (final Future fut : futures) {
                try {
                    fut.get();
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                } catch (ExecutionException e) {
                    throw new RuntimeException(e);
                }
            }
        } finally {
            executor.shutdown();
        }
    }

    public static void switchFiles(Path engineLuceneDirectory, String fileName, String suffix) throws IOException {
        Objects.requireNonNull(engineLuceneDirectory, "engineLuceneDirectory must not be null");
        Objects.requireNonNull(fileName, "fileName must not be null");
        Objects.requireNonNull(suffix, "suffix must not be null");

        Path original = engineLuceneDirectory.resolve(fileName);
        Path suffixed = engineLuceneDirectory.resolve(fileName + suffix);

        if (!Files.exists(original)) {
            throw new IllegalStateException("File not found: " + original);
        }
        if (!Files.exists(suffixed)) {
            throw new IllegalStateException("File not found: " + suffixed);
        }

        System.out.println("Moving [" + suffixed + "] to [" + original + "]");
        Files.move(suffixed, original, StandardCopyOption.REPLACE_EXISTING);
    }

    private static void replaceToBetterHNSW(final TargetFiles targetFiles) throws IOException {
        final String segmentName = targetFiles.flatVectorDataFileName.substring(0, targetFiles.flatVectorDataFileName.indexOf('_', 1));
        final String fieldName = "target_field";
        final int dimension = 768;
        final int fieldNo = 5;
        final int numVectors;
        final int efSearch = 256;
        final int efConstruction = 256;
        final int m = 16;
        final String fp32HnswSuffix = ".fp32";
        final String newHnswSuffix = ".new";

        try (
            final Directory directory = new MMapDirectory(Path.of(targetFiles.engineLuceneDirectory));
            final IndexInput bqFaissIndexInput = directory.openInput(targetFiles.faissIndexFileName, IOContext.DEFAULT)
        ) {

            final FaissIndex bqFaissIndex = FaissIndex.load(bqFaissIndexInput);
            numVectors = bqFaissIndex.getTotalNumberOfVectors();
            System.out.println("Num vectors = " + numVectors + ", loaded from [" + targetFiles.faissIndexFileName + "]");

            // 1. Build FP32 HNSW
            // Prepare segment info
            final SegmentIdExtractor segmentIdExtractor;
            try (final IndexInput vecMetaInput = directory.openInput(targetFiles.flatVectorMetaFileName, IOContext.DEFAULT)) {
                segmentIdExtractor = new SegmentIdExtractor(vecMetaInput);
            }

            final SegmentInfo segmentInfo = new SegmentInfo(
                directory,
                org.apache.lucene.util.Version.LATEST,
                org.apache.lucene.util.Version.LATEST,
                segmentName,
                numVectors,
                false,
                false,
                null,
                Collections.emptyMap(),
                segmentIdExtractor.segmentId,
                Collections.emptyMap(),
                null
            );

            // Field infos
            final FieldInfo[] fieldInfoArr = new FieldInfo[6];
            for (int i = 0; i < fieldInfoArr.length; ++i) {
                final String targetFieldName = i != 5 ? "dummy-" + i : fieldName;
                fieldInfoArr[i] = new FieldInfo(
                    targetFieldName,
                    i,
                    false,
                    false,
                    false,
                    IndexOptions.NONE,
                    DocValuesType.NONE,
                    DocValuesSkipIndexType.NONE,
                    -1,
                    Collections.emptyMap(),
                    0,
                    0,
                    0,
                    dimension,
                    VectorEncoding.FLOAT32,
                    VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
                    false,
                    false
                );
            }
            final FieldInfos fieldInfos = new FieldInfos(fieldInfoArr);

            // Segment state
            final SegmentReadState readState =
                new SegmentReadState(directory, segmentInfo, fieldInfos, IOContext.DEFAULT, segmentIdExtractor.segmentSuffix);

            // Print vectors
            if (false) {
                try (
                    final Lucene99FlatVectorsReader originalReader = new Lucene99FlatVectorsReader(
                        readState,
                                                                                                   DefaultFlatVectorScorer.INSTANCE
                    )
                ) {
                    try {
                        final FloatVectorValues vectorValues = originalReader.getFloatVectorValues(fieldName);
                        final KnnVectorValues.DocIndexIterator iterator = vectorValues.iterator();
                        int doc;
                        while ((doc = iterator.nextDoc()) != NO_MORE_DOCS) {
                            final int ord = iterator.index();
                            final float[] vector = vectorValues.vectorValue(ord);

                            System.out.println("################");
                            System.out.println("doc: " + doc);
                            System.out.println("ord: " + ord);
                            System.out.println("vector: " + Arrays.toString(vector));
                            System.out.println();
                        }
                    } catch (Exception ex) {
                        ex.printStackTrace();
                        throw ex;
                    }
                }
            }

            // Build FP32 HNSW

            // Remove existing reordered files
            try {
                directory.deleteFile(targetFiles.faissIndexFileName + fp32HnswSuffix);
            } catch (NoSuchFileException x) {}

            // Start building FP32 HNSW
            System.out.println("Start building FP32 HNSW ...");
            try (
                final Lucene99FlatVectorsReader originalReader = new Lucene99FlatVectorsReader(readState, DefaultFlatVectorScorer.INSTANCE);
                final IndexOutput indexOutput = directory.createOutput(targetFiles.faissIndexFileName + fp32HnswSuffix, IOContext.DEFAULT)
            ) {
                final MemOptimizedNativeIndexBuildStrategy buildStrategy = MemOptimizedNativeIndexBuildStrategy.getInstance();

                final BuildIndexParams.BuildIndexParamsBuilder buildIndexParamsBuilder = BuildIndexParams.builder();
                buildIndexParamsBuilder.fieldName(fieldName);
                buildIndexParamsBuilder.isFlush(true);
                buildIndexParamsBuilder.indexOutputWithBuffer(new IndexOutputWithBuffer(indexOutput));
                buildIndexParamsBuilder.vectorDataType(VectorDataType.FLOAT);
                buildIndexParamsBuilder.knnEngine(KNNEngine.FAISS);
                final FloatVectorValues vectorValues = originalReader.getFloatVectorValues(fieldName);
                buildIndexParamsBuilder.knnVectorValuesSupplier(KNNVectorValuesFactory.getVectorValuesSupplier(
                    VectorDataType.FLOAT,
                    vectorValues
                ));

                final Map<String, Object> parameters = new HashMap<>();
                parameters.put("name", "hnsw");
                parameters.put("data_type", "float");
                parameters.put("index_description", "HNSW16,Flat");
                parameters.put("spaceType", "innerproduct");
                parameters.put("indexThreadQty", 1);

                final Map<String, Object> innerParameters = new HashMap<>();
                innerParameters.put("ef_search", efSearch);
                innerParameters.put("ef_construction", efConstruction);
                innerParameters.put("m", m);
                innerParameters.put("encoder", Map.of("name", "flat"));
                parameters.put("parameters", innerParameters);

                buildIndexParamsBuilder.parameters(parameters);

                final BuildIndexParams buildIndexParams = buildIndexParamsBuilder.build();

                buildStrategy.buildAndWriteIndex(buildIndexParams);
            }

            // Replace BQ HNSW to FP32 HNSW
            System.out.println("Replacing BQ HNSW to FP32 HNSW ...");
            try {
                directory.deleteFile(targetFiles.faissIndexFileName + newHnswSuffix);
            } catch (NoSuchFileException x) {}

            try (
                final IndexOutput newFaissIndexOutput = directory.createOutput(
                    targetFiles.faissIndexFileName + newHnswSuffix,
                    IOContext.DEFAULT
                ); final IndexInput fp32HnswInput = directory.openInput(targetFiles.faissIndexFileName + fp32HnswSuffix, IOContext.DEFAULT)
            ) {
                final FaissIndex fp32Hnsw = FaissIndex.load(fp32HnswInput);
                final HnswReplaceContext replaceContext = new HnswReplaceContext(fp32Hnsw, fp32HnswInput);
                FaissBQIndexHnswReplacer.replaceHnsw(bqFaissIndex, bqFaissIndexInput, newFaissIndexOutput, replaceContext);
            }

            // Override the existing HNSW
            switchFiles(Path.of(targetFiles.engineLuceneDirectory), targetFiles.faissIndexFileName, newHnswSuffix);
        }
    }

    public static List<TargetFiles> findTargetFiles(Path rootDir) throws IOException {
        Objects.requireNonNull(rootDir, "rootDir must not be null");

        Map<Path, List<Path>> allFiles;
        try (Stream<Path> stream = Files.walk(rootDir)) {
            allFiles = stream.filter(Files::isRegularFile).filter(f -> {
                final String name = f.getFileName().toString(); // Convert to String first
                return name.endsWith(".faiss") || name.endsWith(".faissc") || name.endsWith(".vec") || name.endsWith(".vemf");
            }).collect(Collectors.groupingBy(Path::getParent));
        }

        final List<TargetFiles> results = new ArrayList<>();

        for (Map.Entry<Path, List<Path>> entry : allFiles.entrySet()) {
            Path dir = entry.getKey();

            // Group files in this directory by their "generation" prefix
            Map<String, List<String>> groups = entry.getValue()
                .stream()
                .map(p -> p.getFileName().toString())
                .collect(Collectors.groupingBy(BQHnswTransformerTests::extractGeneration));

            System.out.println("Directory: " + dir + ", groups: " + groups);

            for (Map.Entry<String, List<String>> groupEntry : groups.entrySet()) {
                String gen = groupEntry.getKey();
                final List<String> files = groupEntry.getValue();

                String faiss = findFile(files, ".faiss");
                final String vec = findFile(files, ".vec");
                final String vemf = findFile(files, ".vemf");

                // Step 5: Fallback to .faissc if .faiss is missing
                if (faiss == null) {
                    faiss = findFile(files, ".faissc");
                }

                // Step 6: Validate the triplet
                if (faiss == null || vec == null || vemf == null) {
                    System.out.println("Skipping incomplete generation: " + gen + " in " + dir);
                    continue;
                }

                results.add(new TargetFiles(faiss, vec, vemf, dir.toAbsolutePath().toString()));
            }
        }

        return results;
    }

    private static String findFile(List<String> files, String extension) {
        return files.stream().filter(f -> f.endsWith(extension)).findFirst().orElse(null);
    }

    private static String extractGeneration(String fileName) {
        final int firstUnderscore = fileName.indexOf('_');
        if (firstUnderscore == -1) {
            throw new IllegalStateException("No generation found in " + fileName);
        }

        final int secondUnderscore = fileName.indexOf('_', firstUnderscore + 1);
        if (secondUnderscore == -1) {
            throw new IllegalStateException("No generation found in " + fileName);
        }

        return fileName.substring(0, secondUnderscore);
    }
}
