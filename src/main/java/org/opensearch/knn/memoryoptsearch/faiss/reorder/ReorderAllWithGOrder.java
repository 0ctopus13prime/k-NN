/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

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
import org.opensearch.knn.memoryoptsearch.faiss.FaissHNSW;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHnswGraph;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIdMapIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

public class ReorderAllWithGOrder {
    private static final String reorderSuffix = ".reorder";

    public record TargetFiles(String faissIndexFileName, String flatVectorDataFileName, String flatVectorMetaFileName,
        String engineLuceneDirectory) {
    }

    public static void main(String... args) throws IOException {
        final String searchDirectory = "/Users/kdooyong/workspace/opensearch-gorder/kdy";

        final List<TargetFiles> targetFilesList = findTargetFiles(Path.of(searchDirectory));
        for (final TargetFiles targetFiles : targetFilesList) {
            System.out.println();
            System.out.println("Start reordering ...");
            System.out.println("  Lucene dir: " + targetFiles.engineLuceneDirectory);
            System.out.println("  Vec: " + targetFiles.flatVectorDataFileName);
            System.out.println("  VecMeta: " + targetFiles.flatVectorMetaFileName);
            System.out.println("  Faiss: " + targetFiles.faissIndexFileName);
            reorder(targetFiles);
        }
    }

    public static List<TargetFiles> findTargetFiles(Path rootDir) throws IOException {
        Objects.requireNonNull(rootDir, "rootDir must not be null");

        // 1. Group file names by their parent directory
        Map<Path, List<String>> filesByDir;
        try (Stream<Path> stream = Files.walk(rootDir)) {
            filesByDir = stream.filter(Files::isRegularFile)
                .collect(Collectors.groupingBy(Path::getParent, Collectors.mapping(p -> p.getFileName().toString(), Collectors.toList())));
        }

        final List<TargetFiles> results = new ArrayList<>();

        // 2. Process each directory group
        for (Map.Entry<Path, List<String>> entry : filesByDir.entrySet()) {
            Path dir = entry.getKey();
            List<String> fileNames = entry.getValue();

            // Identify specific files within this directory
            String faiss = findSingle(fileNames, ".faiss", dir);
            String vec = findSingle(fileNames, ".vec", dir);
            String vemf = findSingle(fileNames, ".vemf", dir);

            // Rule: If any file is found, ALL must be found in this directory
            if (faiss != null || vec != null || vemf != null) {
                if (faiss == null || vec == null || vemf == null) {
                    throw new IllegalStateException(
                        String.format("Incomplete file set in %s: faiss=%b, vec=%b, vemf=%b", dir, faiss != null, vec != null, vemf != null)
                    );
                }

                results.add(new TargetFiles(faiss, vec, vemf, dir.toAbsolutePath().toString()));
            }
        }

        return results;
    }

    //
    // Helper to ensure only one file of a specific extension exists in a folder.
    //
    private static String findSingle(List<String> names, String extension, Path dir) {
        List<String> matches = names.stream().filter(n -> n.endsWith(extension)).toList();

        if (matches.size() > 1) {
            throw new IllegalStateException("Multiple " + extension + " files found in: " + dir);
        }
        return matches.isEmpty() ? null : matches.get(0);
    }

    private static void reorder(TargetFiles targetFiles) throws IOException {
        final String segmentName = targetFiles.flatVectorDataFileName.substring(0, targetFiles.flatVectorDataFileName.indexOf('_', 1));
        final String fieldName = "target_field";
        final int dimension = 768;
        final int fieldNo = 5;

        try (final Directory directory = new MMapDirectory(Path.of(targetFiles.engineLuceneDirectory))) {
            try (final IndexInput faissIndexInput = directory.openInput(targetFiles.faissIndexFileName, IOContext.DEFAULT)) {
                final FaissIndex faissIndex = FaissIndex.load(faissIndexInput);
                if (faissIndex instanceof FaissIdMapIndex idMapIndex) {
                    final int numVectors = faissIndex.getTotalNumberOfVectors();
                    System.out.println("Total #vectors: " + numVectors + " (extracted from faiss index)");

                    // Get the permutation
                    // Permutation -> newOrd2Old
                    long s = System.nanoTime();
                    final ReorderOrdMap reorderOrdMap = getOrderMap(idMapIndex, faissIndexInput, targetFiles, directory);
                    long e = System.nanoTime();
                    System.out.println("Reordering took : " + (e - s) / 1e6 + "ms");

                    // Remove existing reordered files
                    for (final String reorderedFile : Arrays.asList(
                        targetFiles.flatVectorDataFileName + reorderSuffix,
                        targetFiles.flatVectorMetaFileName + reorderSuffix
                    )) {
                        try {
                            directory.deleteFile(reorderedFile);
                        } catch (NoSuchFileException x) {}
                    }

                    // Transform .vec file
                    System.out.println("Transforming .vec file: " + targetFiles.flatVectorDataFileName + " ...");
                    s = System.nanoTime();
                    final ReorderedFlatVectorsWriter flatVectorsWriter;
                    try (final IndexInput vecMetaInput = directory.openInput(targetFiles.flatVectorMetaFileName, IOContext.DEFAULT)) {
                        flatVectorsWriter = new ReorderedFlatVectorsWriter(
                            directory,
                            targetFiles.flatVectorMetaFileName + reorderSuffix,
                            targetFiles.flatVectorDataFileName + reorderSuffix,
                            faissIndex.getTotalNumberOfVectors(),
                            vecMetaInput
                        );
                    }

                    // Prepare segment info
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
                        flatVectorsWriter.segmentId,
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
                    final SegmentReadState readState = new SegmentReadState(
                        directory,
                        segmentInfo,
                        fieldInfos,
                        IOContext.DEFAULT,
                        flatVectorsWriter.segmentSuffix
                    );

                    // Reorder .vec file
                    try (
                        flatVectorsWriter;
                        Lucene99FlatVectorsReader flatVectorsReader = new Lucene99FlatVectorsReader(
                            readState,
                            DefaultFlatVectorScorer.INSTANCE
                        )
                    ) {
                        // Now, start reordering .vec file
                        final FloatVectorValues vectorValues = flatVectorsReader.getFloatVectorValues(fieldName);

                        // Reorder vectors
                        final ReorderedFlatVectorsWriter.ReorderedFlatFieldVectorsWriter writer = flatVectorsWriter.addField(
                            fieldInfoArr[fieldNo]
                        );
                        for (int ord = 0; ord < reorderOrdMap.newOrd2Old.length; ++ord) {
                            final float[] vector = vectorValues.vectorValue(reorderOrdMap.newOrd2Old[ord]);
                            writer.addValue(reorderOrdMap.newOrd2Old[ord], vector);
                        }
                        writer.finish();
                    }
                    e = System.nanoTime();
                    System.out.println("Transforming .vec took " + ((e - s) / 1e6 + "ms"));

                    // Test reordered .vec file
                    System.out.println("Validating reordered .vec file ...");
                    try (
                        final ReorderedLucene99FlatVectorsReader reorderedReader = new ReorderedLucene99FlatVectorsReader(
                            readState,
                            DefaultFlatVectorScorer.INSTANCE,
                            true
                        );
                        final Lucene99FlatVectorsReader originalReader = new Lucene99FlatVectorsReader(
                            readState,
                            DefaultFlatVectorScorer.INSTANCE
                        )
                    ) {
                        final ExecutorService pool = Executors.newFixedThreadPool(4);
                        final List<Future> futures = new ArrayList<>();
                        for (int k = 0; k < 100; ++k) {
                            Runnable r = () -> {
                                try {
                                    final FloatVectorValues originalVectorValues = originalReader.getFloatVectorValues(fieldName);
                                    final FloatVectorValues vectorValues = reorderedReader.getFloatVectorValues(fieldName);
                                    final KnnVectorValues.DocIndexIterator iterator = vectorValues.iterator();
                                    int doc;
                                    while ((doc = iterator.nextDoc()) != NO_MORE_DOCS) {
                                        // from reordered one
                                        final int ord = iterator.index();
                                        final float[] vector = vectorValues.vectorValue(ord);

                                        // from original
                                        final float[] expected = originalVectorValues.vectorValue(reorderOrdMap.newOrd2Old[ord]);

                                        if (false) {
                                            System.out.println("################");
                                            System.out.println("doc: " + doc);
                                            System.out.println("ord: " + ord);
                                            System.out.println("vector: " + Arrays.toString(vector));
                                            System.out.println("expected: " + Arrays.toString(expected));
                                            System.out.println();
                                        }
                                    }
                                } catch (Exception ex) {
                                    ex.printStackTrace();
                                }
                            };
                            futures.add(pool.submit(r));
                        }

                        for (Future fut : futures) {
                            try {
                                fut.get();
                            } catch (InterruptedException ex) {
                                throw new RuntimeException(ex);
                            } catch (ExecutionException ex) {
                                throw new RuntimeException(ex);
                            }
                        }
                    }
                } else {
                    throw new IllegalStateException("faissIndex is not FaissIdMapIndex! Actual type: " + faissIndex.getClass());
                }
            }
        }

        System.out.println("Reorder is done!");
        System.out.println("Overriding files ...");
        switchFiles(Path.of(targetFiles.engineLuceneDirectory), targetFiles.faissIndexFileName, reorderSuffix);
        switchFiles(Path.of(targetFiles.engineLuceneDirectory), targetFiles.flatVectorDataFileName, reorderSuffix);
        switchFiles(Path.of(targetFiles.engineLuceneDirectory), targetFiles.flatVectorMetaFileName, reorderSuffix);
    }

    private static ReorderOrdMap getOrderMap(
        FaissIdMapIndex idMapIndex,
        IndexInput faissIndexInput,
        TargetFiles targetFiles,
        Directory directory
    ) throws IOException {

        // Parameters
        final int window = 16;

        // Prepare
        final FaissHNSW faissHNSW = idMapIndex.getFaissHnsw();
        final FaissHnswGraph faissHnswGraph = new FaissHnswGraph(faissHNSW, faissIndexInput);
        final int numVectors = Math.toIntExact(faissHNSW.getTotalNumberOfVectors());

        // Construct incoming vertexes
        final IncomingVertexes incomingVertexes = IncomingVertexes.collectIncomingVertices(faissHnswGraph);

        // Isolated vectors
        List<Integer> isolatedVertexes = new ArrayList<>();

        // Heap
        final UnitHeap unitHeap = new UnitHeap(numVectors);

        // Bit set
        final BitSet popvExist = new BitSet(numVectors);

        // New permutation
        int orderIndex = 0;
        int[] newPermutation = new int[numVectors];

        // Find the first vertex with the largest indegree
        int veryFirstVertex = -1;
        int maxInDegreeFound = -1;
        for (int i = 0; i < numVectors; i++) {
            final int inDegree = incomingVertexes.getDegree(i);
            if (inDegree > maxInDegreeFound) {
                maxInDegreeFound = inDegree;
                veryFirstVertex = i;
            } else if (inDegree + HnswGraphHelper.getOutDegree(faissHnswGraph, i) == 0) {
                unitHeap.update[i] = Integer.MAX_VALUE / 2;
                isolatedVertexes.add(i);
                unitHeap.deleteElement(i);
            }
        }

        // Push the first vertex and remove it from the queue
        newPermutation[orderIndex++] = veryFirstVertex;
        unitHeap.deleteElement(veryFirstVertex);

        final AtomicReference<Boolean> skipIterationForCommonNeighbor = new AtomicReference<>(false);
        final int untilOrderIndex = (numVectors - isolatedVertexes.size());
        while (true) {
            final int ve = newPermutation[orderIndex - 1];
            final int vb = (orderIndex > (window + 1)) ? newPermutation[orderIndex - window - 1] : -1;

            // For incoming vertexes to `ve`
            // e.g.
            // A -> V_e -> P
            // -> X
            // Then increase counter for A, X and P.
            for (int i = incomingVertexes.startOffset(ve), limit1 = incomingVertexes.endOffset(ve); i < limit1; ++i) {
                // Increase incoming vertex count
                final int u = incomingVertexes.incomingVertices[i];
                if (unitHeap.update[u] == 0) {
                    unitHeap.increaseKey(u);
                } else {
                    unitHeap.update[u] += 1;
                }

                // Increase sibling count
                if (HnswGraphHelper.getOutDegree(faissHnswGraph, u) > 1) {
                    skipIterationForCommonNeighbor.set(false);
                    if (vb != -1) {
                        // We have a leaving node from the window
                        HnswGraphHelper.forAllOutgoingNodes(faissHnswGraph, u, 0, (w) -> {
                            if (w == vb) {
                                skipIterationForCommonNeighbor.set(true);
                                // Stop the loop
                                return false;
                            }
                            return true;
                        });
                    }

                    if (skipIterationForCommonNeighbor.get() == false) {
                        // If `popv` (e.g. v_b) and v (e.g. v_max) are NOT sibling, then do below:
                        HnswGraphHelper.forAllOutgoingNodes(faissHnswGraph, u, 0, (w) -> { unitHeap.update[w] += 1; });
                    } else {
                        popvExist.set(u);
                    }  // End if
                }
            }  // End for

            // For the vertexes pointed by vertex of `veryFirstVertex`
            HnswGraphHelper.forAllOutgoingNodes(faissHnswGraph, ve, 0, (w) -> {
                if (unitHeap.update[w] == 0) {
                    unitHeap.increaseKey(w);
                } else {
                    unitHeap.update[w] += 1;
                }
            });

            if (vb != -1) {
                // For incoming vertexes to `vb`
                // e.g.
                // A -> V_b
                // -> X
                // Then increase counter for A and X
                for (int i = incomingVertexes.startOffset(vb), limit1 = incomingVertexes.endOffset(vb); i < limit1; ++i) {
                    final int u = incomingVertexes.incomingVertices[i];
                    unitHeap.update[u] -= 1;

                    // Increase sibling count
                    if (HnswGraphHelper.getOutDegree(faissHnswGraph, u) > 1) {
                        if (popvExist.get(u) == false) {
                            HnswGraphHelper.forAllOutgoingNodes(faissHnswGraph, u, 1, (w) -> { unitHeap.update[w] -= 1; });
                        } else {
                            popvExist.set(u, false);
                        }
                    }
                }  // End for

                // For the vertexes pointed by vertex of `veryFirstVertex`
                HnswGraphHelper.forAllOutgoingNodes(faissHnswGraph, vb, 0, (w) -> { unitHeap.update[w] -= 1; });
            }  // End if

            if (orderIndex < untilOrderIndex) {
                final int vmax = unitHeap.extractMax();
                newPermutation[orderIndex++] = vmax;
            } else {
                break;
            }
        }  // End while

        // Insert isolated vertexes
        for (final int isolatedIndex : isolatedVertexes) {
            newPermutation[orderIndex++] = isolatedIndex;
        }

        final ReorderOrdMap reorderOrdMap = new ReorderOrdMap(newPermutation);

        // Transform the faiss index
        try (final IndexOutput indexOutput = directory.createOutput(targetFiles.faissIndexFileName + reorderSuffix, IOContext.DEFAULT)) {
            FaissIndexReorderTransformer.transform(idMapIndex, faissIndexInput, indexOutput, reorderOrdMap);
        }

        return reorderOrdMap;
    }

    public static void switchFiles(Path engineLuceneDirectory, String fileName, String suffix) throws IOException {

        Objects.requireNonNull(engineLuceneDirectory, "engineLuceneDirectory must not be null");
        Objects.requireNonNull(fileName, "fileName must not be null");
        Objects.requireNonNull(suffix, "suffix must not be null");

        Path original = engineLuceneDirectory.resolve(fileName);
        Path backup = engineLuceneDirectory.resolve(fileName + ".bak");
        Path suffixed = engineLuceneDirectory.resolve(fileName + suffix);

        if (!Files.exists(original)) {
            throw new IllegalStateException("File not found: " + original);
        }
        if (!Files.exists(suffixed)) {
            throw new IllegalStateException("File not found: " + suffixed);
        }
        if (Files.exists(backup)) {
            throw new IllegalStateException("Backup file already exists: " + backup);
        }

        Files.move(original, backup);
        Files.move(suffixed, original);
    }

    private static int[] getPermutation(FaissHNSW faissHNSW, IndexInput indexInput, int window) throws IOException {
        // Prepare
        final FaissHnswGraph faissHnswGraph = new FaissHnswGraph(faissHNSW, indexInput);
        final int numVectors = Math.toIntExact(faissHNSW.getTotalNumberOfVectors());

        // Construct incoming vertexes
        final IncomingVertexes incomingVertexes = IncomingVertexes.collectIncomingVertices(faissHnswGraph);

        // Isolated vectors
        List<Integer> isolatedVertexes = new ArrayList<>();

        // Heap
        final UnitHeap unitHeap = new UnitHeap(numVectors);

        // Bit set
        final BitSet popvExist = new BitSet(numVectors);

        // New permutation
        int orderIndex = 0;
        int[] newPermutation = new int[numVectors];

        // Find the first vertex with the largest indegree
        int veryFirstVertex = -1;
        int maxInDegreeFound = -1;
        for (int i = 0; i < numVectors; i++) {
            final int inDegree = incomingVertexes.getDegree(i);
            if (inDegree > maxInDegreeFound) {
                maxInDegreeFound = inDegree;
                veryFirstVertex = i;
            } else if (inDegree + HnswGraphHelper.getOutDegree(faissHnswGraph, i) == 0) {
                unitHeap.update[i] = Integer.MAX_VALUE / 2;
                isolatedVertexes.add(i);
                unitHeap.deleteElement(i);
            }
        }

        // Push the first vertex and remove it from the queue
        newPermutation[orderIndex++] = veryFirstVertex;
        unitHeap.deleteElement(veryFirstVertex);

        final AtomicReference<Boolean> skipIterationForCommonNeighbor = new AtomicReference<>(false);
        final int untilOrderIndex = (numVectors - isolatedVertexes.size());
        while (true) {
            final int ve = newPermutation[orderIndex - 1];
            final int vb = (orderIndex > (window + 1)) ? newPermutation[orderIndex - window - 1] : -1;

            // For incoming vertexes to `ve`
            // e.g.
            // A -> V_e -> P
            // -> X
            // Then increase counter for A, X and P.
            for (int i = incomingVertexes.startOffset(ve), limit1 = incomingVertexes.endOffset(ve); i < limit1; ++i) {
                // Increase incoming vertex count
                final int u = incomingVertexes.incomingVertices[i];
                if (unitHeap.update[u] == 0) {
                    unitHeap.increaseKey(u);
                } else {
                    unitHeap.update[u] += 1;
                }

                // Increase sibling count
                if (HnswGraphHelper.getOutDegree(faissHnswGraph, u) > 1) {
                    skipIterationForCommonNeighbor.set(false);
                    if (vb != -1) {
                        // We have a leaving node from the window
                        HnswGraphHelper.forAllOutgoingNodes(faissHnswGraph, u, 0, (w) -> {
                            if (w == vb) {
                                skipIterationForCommonNeighbor.set(true);
                                // Stop the loop
                                return false;
                            }
                            return true;
                        });
                    }

                    if (skipIterationForCommonNeighbor.get() == false) {
                        // If `popv` (e.g. v_b) and v (e.g. v_max) are NOT sibling, then do below:
                        HnswGraphHelper.forAllOutgoingNodes(faissHnswGraph, u, 0, (w) -> { unitHeap.update[w] += 1; });
                    } else {
                        popvExist.set(u);
                    }  // End if
                }
            }  // End for

            // For the vertexes pointed by vertex of `veryFirstVertex`
            HnswGraphHelper.forAllOutgoingNodes(faissHnswGraph, ve, 0, (w) -> {
                if (unitHeap.update[w] == 0) {
                    unitHeap.increaseKey(w);
                } else {
                    unitHeap.update[w] += 1;
                }
            });

            if (vb != -1) {
                // For incoming vertexes to `vb`
                // e.g.
                // A -> V_b
                // -> X
                // Then increase counter for A and X
                for (int i = incomingVertexes.startOffset(vb), limit1 = incomingVertexes.endOffset(vb); i < limit1; ++i) {
                    final int u = incomingVertexes.incomingVertices[i];
                    unitHeap.update[u] -= 1;

                    // Increase sibling count
                    if (HnswGraphHelper.getOutDegree(faissHnswGraph, u) > 1) {
                        if (popvExist.get(u) == false) {
                            HnswGraphHelper.forAllOutgoingNodes(faissHnswGraph, u, 1, (w) -> { unitHeap.update[w] -= 1; });
                        } else {
                            popvExist.set(u, false);
                        }
                    }
                }  // End for

                // For the vertexes pointed by vertex of `veryFirstVertex`
                HnswGraphHelper.forAllOutgoingNodes(faissHnswGraph, vb, 0, (w) -> { unitHeap.update[w] -= 1; });
            }  // End if

            if (orderIndex < untilOrderIndex) {
                final int vmax = unitHeap.extractMax();
                newPermutation[orderIndex++] = vmax;
            } else {
                break;
            }
        }  // End while

        // Insert isolated vertexes
        for (final int isolatedIndex : isolatedVertexes) {
            newPermutation[orderIndex++] = isolatedIndex;
        }

        return newPermutation;
    }
}
