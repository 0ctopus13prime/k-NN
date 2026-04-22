/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.lucene104.QuantizedByteVectorValues;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOUtils;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsReader;
import org.opensearch.knn.index.codec.nativeindex.AbstractNativeEnginesKnnVectorsReader;
import org.opensearch.knn.index.codec.nativeindex.ErrorResidualRefiner;
import org.opensearch.knn.index.codec.nativeindex.ResidualQuantizer;
import org.opensearch.knn.index.util.WarmupUtil;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Reader for Faiss 1040 scalar quantized vector fields. Extends {@link AbstractNativeEnginesKnnVectorsReader}
 * and always forces memory-optimized search regardless of the index-level setting.
 *
 * <p>Key differences from {@link NativeEngines990KnnVectorsReader}:
 * <ul>
 *   <li>Always forces memory-optimized search — not gated by index setting</li>
 *   <li>No quantization state cache (quantization is handled by Lucene, not k-NN's framework)</li>
 *   <li>No NativeMemoryCacheManager invalidation on close</li>
 *   <li>Byte vector search is not supported</li>
 *   <li>Implements {@link ErrorResidualRefiner} — can refine 1st-phase scores using 4-bit
 *       quantized error residuals from the {@code .ver} file</li>
 * </ul>
 *
 * <p>{@link #getFloatVectorValues(String)} delegates to Lucene's
 * {@code Lucene104ScalarQuantizedVectorsReader}, which returns a {@link FloatVectorValues}
 * with both {@code scorer()} (quantized) and {@code rescorer()} (full-precision) support.
 */
@Log4j2
public class Faiss1040ScalarQuantizedKnnVectorsReader extends AbstractNativeEnginesKnnVectorsReader implements ErrorResidualRefiner {

    /** Eagerly loaded .ver file reader, or null if the .ver file doesn't exist for this segment. */
    private final ErrorResidualReader errorResidualReader;

    Faiss1040ScalarQuantizedKnnVectorsReader(SegmentReadState state, FlatVectorsReader flatVectorsReader) {
        super(state, flatVectorsReader);
        this.errorResidualReader = tryLoadErrorResidualReader(state, flatVectorsReader);
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        throw new UnsupportedOperationException("Byte vector search is not supported for Faiss scalar quantized format");
    }

    /**
     * Always uses memory-optimized search — not gated by the index-level memory_optimized_search
     * setting. A null target triggers warmup initialization.
     * Throws IllegalStateException if the searcher cannot be loaded (e.g., no native file).
     */
    @Override
    public void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        final VectorSearcher memoryOptimizedSearcher = loadMemoryOptimizedSearcherIfRequired(fieldInfos.fieldInfo(field));

        if (memoryOptimizedSearcher == null) {
            throw new IllegalStateException(
                "Faiss scalar quantized format requires memory optimized search but searcher could not be loaded for field [" + field + "]"
            );
        }

        memoryOptimizedSearcher.search(target, knnCollector, acceptDocs);
    }

    @Override
    public void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        throw new UnsupportedOperationException("Byte vector search is not supported for Faiss scalar quantized format");
    }

    /**
     * Warms up the on-disk data for the given scalar-quantized field.
     * <p>
     * This warms up both the HNSW graph (via the memory-optimized searcher), quantized vectors and the
     * full-precision vectors. The full-precision vectors cannot be warmed up through
     * {@link WarmupUtil} because the {@link FloatVectorValues}
     * returned by the flat vectors reader is backed by quantized data. Instead, each vector
     * is read explicitly through the underlying
     * {@link ScalarQuantizedFloatVectorValues}.
     *
     * @param fieldName the name of the vector field to warm up
     * @throws IOException if an I/O error occurs while reading the underlying data
     */
    @Override
    public void warmUp(final String fieldName) throws IOException {
        // Warm up full-precision vectors
        // We cannot rely on WarmupUtil, which extracts the IndexInput from vector values and reads through it.
        // Because, the IndexInput returned by vector values is backed by quantized vectors.
        // Therefore, to warm up full-precision vectors, we need to load them explicitly as below.
        final ScalarQuantizedFloatVectorValues vectorValues = (ScalarQuantizedFloatVectorValues) flatVectorsReader.getFloatVectorValues(
            fieldName
        );
        for (int i = 0; i < vectorValues.size(); ++i) {
            vectorValues.vectorValue(i);
        }

        final VectorSearcher memoryOptimizedSearcher = loadMemoryOptimizedSearcherIfRequired(fieldInfos.fieldInfo(fieldName));
        if (memoryOptimizedSearcher != null) {
            // MOS is supported, warm up search parts
            memoryOptimizedSearcher.warmUp();
        } else {
            log.warn("Memory optimized search is not supported for {}", fieldName);
        }
    }

    /**
     * Refine 1st-phase scores using 4-bit quantized error residuals from the {@code .ver} file.
     *
     * <p>For each candidate document:
     * <ol>
     *   <li>Read the per-vector block from the .ver file (packed residual + metadata)</li>
     *   <li>Extract per-vector lower/upper bounds from the block metadata</li>
     *   <li>Dequantize the 4-bit residual and compute {@code <q', Q_4(r)>}</li>
     *   <li>Add the correction to the phase-1 score</li>
     * </ol>
     *
     * <p>{@code q' = query - centroid} is precomputed once per query.
     * The cloned {@link IndexInput} provides thread-safe reads for concurrent search.
     *
     * @param field        the vector field name
     * @param queryVector  the original query vector
     * @param docIds       segment-local document IDs (dense case: docId == ordinal)
     * @param phase1Scores corresponding 1st-phase scores from approximate search
     * @return ScoreDoc[] with refined scores for all input documents
     */
    @Override
    public ScoreDoc[] refine(String field, float[] queryVector, int[] docIds, float[] phase1Scores) throws IOException {
        if (errorResidualReader == null) {
            throw new IllegalStateException("Error residual reader not available for field: " + field);
        }

        final int dimension = errorResidualReader.getDimension();

        // Precompute q' = query - centroid once for this query
        final float[] qPrime = ResidualQuantizer.computeQPrime(queryVector, errorResidualReader.getCentroid());

        // Clone IndexInput for thread-safe reads (each search thread gets its own clone)
        try (IndexInput clonedInput = errorResidualReader.cloneInput()) {
            ScoreDoc[] result = new ScoreDoc[docIds.length];

            for (int i = 0; i < docIds.length; i++) {
                // Dense case: docId == ordinal
                byte[] block = errorResidualReader.readBlock(clonedInput, docIds[i]);

                // Extract per-vector lower/upper from the block metadata
                float lower = errorResidualReader.extractLower(block);
                float upper = errorResidualReader.extractUpper(block);

                // Compute corrected score: phase1Score + <q', Q_4(r)>
                float correctedScore = ResidualQuantizer.computeCorrectedScore(
                    qPrime, block, lower, upper, phase1Scores[i], dimension
                );

                result[i] = new ScoreDoc(docIds[i], correctedScore);
            }

            return result;
        }
    }

    @Override
    public void close() throws IOException {
        final List<Closeable> closeables = new ArrayList<>();
        closeables.add(flatVectorsReader);
        if (vectorSearcherHolder != null) {
            closeables.add(vectorSearcherHolder.getVectorSearcher());
        }
        if (errorResidualReader != null) {
            closeables.add(errorResidualReader);
        }
        IOUtils.close(closeables);
    }

    /**
     * Attempt to load the {@link ErrorResidualReader} for the first SQ field in this segment.
     *
     * <p>Extracts the centroid from {@link QuantizedByteVectorValues} (via reflection),
     * checks if the {@code .ver} file exists in the segment directory, and opens it eagerly.
     *
     * <p>Returns null if the .ver file doesn't exist or if an error occurs during loading.
     * In that case, the standard full-precision rescore path will be used as fallback.
     */
    private ErrorResidualReader tryLoadErrorResidualReader(SegmentReadState state, FlatVectorsReader flatVectorsReader) {
        try {
            for (FieldInfo fi : state.fieldInfos) {
                if (FieldInfoExtractor.isSQField(fi)) {
                    // Extract centroid from the quantized vector values via reflection
                    QuantizedByteVectorValues qbvv = KNN1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(
                        flatVectorsReader.getFloatVectorValues(fi.getName())
                    );
                    float[] centroid = qbvv.getCentroid();

                    String verFileName = state.segmentInfo.name + "_" + fi.getName() + ".ver";
                    if (Arrays.asList(state.directory.listAll()).contains(verFileName)) {
                        return new ErrorResidualReader(state.directory, state.segmentInfo.name, fi.getName(), centroid);
                    }
                }
            }
        } catch (IOException e) {
            log.warn("Failed to load error residual reader, falling back to standard rescore", e);
        }
        return null;
    }
}
