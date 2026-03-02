/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.bbq;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.internal.hppc.FloatArrayList;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.CloseableRandomVectorScorerSupplier;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import static org.apache.lucene.codecs.lucene102.Lucene102BinaryQuantizedVectorsFormat.BINARIZED_VECTOR_COMPONENT;
import static org.apache.lucene.codecs.lucene102.Lucene102BinaryQuantizedVectorsFormat.INDEX_BITS;
import static org.apache.lucene.codecs.lucene102.Lucene102BinaryQuantizedVectorsFormat.QUERY_BITS;
import static org.apache.lucene.index.VectorSimilarityFunction.COSINE;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.apache.lucene.util.RamUsageEstimator.shallowSizeOfInstance;
import static org.apache.lucene.util.quantization.OptimizedScalarQuantizer.discretize;
import static org.apache.lucene.util.quantization.OptimizedScalarQuantizer.packAsBinary;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

public class BBQWriter extends FlatVectorsWriter {
    static final int VERSION_START = 0;
    static final int VERSION_CURRENT = VERSION_START;
    static final String META_CODEC_NAME = "Lucene102BinaryQuantizedVectorsFormatMeta";
    static final String VECTOR_DATA_CODEC_NAME = "Lucene102BinaryQuantizedVectorsFormatData";
    static final String META_EXTENSION = "vemb";
    static final String VECTOR_DATA_EXTENSION = "veb";
    static final int DIRECT_MONOTONIC_BLOCK_SHIFT = 16;

    private static final long SHALLOW_RAM_BYTES_USED = shallowSizeOfInstance(BBQWriter.class);

    private final SegmentWriteState segmentWriteState;
    private final List<BBQWriter.FieldWriter> fields = new ArrayList<>();
    private final IndexOutput meta, binarizedVectorData;
    private final Lucene102BinaryFlatVectorsScorer vectorsScorer;
    private boolean finished;

    /**
     * Sole constructor
     *
     * @param vectorsScorer the scorer to use for scoring vectors
     */
    public BBQWriter(Lucene102BinaryFlatVectorsScorer vectorsScorer, SegmentWriteState state) throws IOException {
        super(vectorsScorer);
        this.vectorsScorer = vectorsScorer;
        this.segmentWriteState = state;
        String metaFileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, META_EXTENSION);

        String binarizedVectorDataFileName =
            IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, VECTOR_DATA_EXTENSION);
        boolean success = false;
        try {
            meta = state.directory.createOutput(metaFileName, state.context);
            binarizedVectorData = state.directory.createOutput(binarizedVectorDataFileName, state.context);

            CodecUtil.writeIndexHeader(meta, META_CODEC_NAME, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);
            CodecUtil.writeIndexHeader(
                binarizedVectorData,
                VECTOR_DATA_CODEC_NAME,
                VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
            success = true;
        } finally {
            if (success == false) {
                IOUtils.closeWhileHandlingException(this);
            }
        }
    }

    @Override
    public FlatFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
        @SuppressWarnings("unchecked")
        FieldWriter fieldWriter = new FieldWriter(fieldInfo);
        fields.add(fieldWriter);
        return fieldWriter;
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        for (FieldWriter field : fields) {
            // after raw vectors are written, normalize vectors for clustering and quantization
            final float[] clusterCenter;
            int vectorCount = field.vectors.size();
            clusterCenter = new float[field.dimensionSums.length];
            if (vectorCount > 0) {
                for (int i = 0; i < field.dimensionSums.length; i++) {
                    clusterCenter[i] = field.dimensionSums[i] / vectorCount;
                }
            }
            OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(field.fieldInfo.getVectorSimilarityFunction());
            if (sortMap == null) {
                writeField(field, clusterCenter, maxDoc, quantizer);
            } else {
                writeSortingField(field, clusterCenter, maxDoc, sortMap, quantizer);
            }
            field.finish();
        }
    }

    private void writeField(FieldWriter fieldData, float[] clusterCenter, int maxDoc, OptimizedScalarQuantizer quantizer)
        throws IOException {
        // write vector values
        long vectorDataOffset = binarizedVectorData.alignFilePointer(Float.BYTES);
        writeBinarizedVectors(fieldData, clusterCenter, quantizer);
        long vectorDataLength = binarizedVectorData.getFilePointer() - vectorDataOffset;
        float centroidDp = fieldData.getVectors().size() > 0 ? VectorUtil.dotProduct(clusterCenter, clusterCenter) : 0;

        writeMeta(
            fieldData.fieldInfo,
            maxDoc,
            vectorDataOffset,
            vectorDataLength,
            clusterCenter,
            centroidDp,
            fieldData.getDocsWithFieldSet()
        );
    }

    private void writeBinarizedVectors(FieldWriter fieldData, float[] clusterCenter, OptimizedScalarQuantizer scalarQuantizer)
        throws IOException {
        int discreteDims = discretize(fieldData.fieldInfo.getVectorDimension(), 64);
        byte[] quantizationScratch = new byte[discreteDims];
        byte[] vector = new byte[discreteDims / 8];
        for (int i = 0; i < fieldData.getVectors().size(); i++) {
            float[] v = fieldData.getVectors().get(i);
            OptimizedScalarQuantizer.QuantizationResult corrections =
                scalarQuantizer.scalarQuantize(v, quantizationScratch, INDEX_BITS, clusterCenter);
            packAsBinary(quantizationScratch, vector);
            binarizedVectorData.writeBytes(vector, vector.length);
            binarizedVectorData.writeInt(Float.floatToIntBits(corrections.lowerInterval()));
            binarizedVectorData.writeInt(Float.floatToIntBits(corrections.upperInterval()));
            binarizedVectorData.writeInt(Float.floatToIntBits(corrections.additionalCorrection()));
            assert corrections.quantizedComponentSum() >= 0 && corrections.quantizedComponentSum() <= 0xffff;
            binarizedVectorData.writeShort((short) corrections.quantizedComponentSum());
        }
    }

    private void writeSortingField(
        FieldWriter fieldData,
        float[] clusterCenter,
        int maxDoc,
        Sorter.DocMap sortMap,
        OptimizedScalarQuantizer scalarQuantizer
    ) throws IOException {
        throw new UnsupportedOperationException();
    }

    private void writeSortedBinarizedVectors(
        FieldWriter fieldData,
        float[] clusterCenter,
        int[] ordMap,
        OptimizedScalarQuantizer scalarQuantizer
    ) throws IOException {
        throw new UnsupportedOperationException();
    }

    private void writeMeta(
        FieldInfo field,
        int maxDoc,
        long vectorDataOffset,
        long vectorDataLength,
        float[] clusterCenter,
        float centroidDp,
        DocsWithFieldSet docsWithField
    ) throws IOException {
        meta.writeInt(field.number);
        meta.writeInt(field.getVectorEncoding().ordinal());
        meta.writeInt(field.getVectorSimilarityFunction().ordinal());
        meta.writeVInt(field.getVectorDimension());
        meta.writeVLong(vectorDataOffset);
        meta.writeVLong(vectorDataLength);
        int count = docsWithField.cardinality();
        meta.writeVInt(count);
        if (count > 0) {
            final ByteBuffer buffer = ByteBuffer.allocate(field.getVectorDimension() * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            buffer.asFloatBuffer().put(clusterCenter);
            meta.writeBytes(buffer.array(), buffer.array().length);
            meta.writeInt(Float.floatToIntBits(centroidDp));
        }
        OrdToDocDISIReaderConfiguration.writeStoredMeta(
            DIRECT_MONOTONIC_BLOCK_SHIFT,
            meta,
            binarizedVectorData,
            count,
            maxDoc,
            docsWithField
        );
    }

    @Override
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException("already finished");
        }
        finished = true;
        if (meta != null) {
            // write end of fields marker
            meta.writeInt(-1);
            CodecUtil.writeFooter(meta);
        }
        if (binarizedVectorData != null) {
            CodecUtil.writeFooter(binarizedVectorData);
        }
    }

    @Override
    public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) {
        throw new UnsupportedOperationException();
    }

    /**
     * 2-phase merge: reads vectors twice to avoid holding all fp32 vectors in memory.
     * Pass 1: iterate vectors to compute centroid (dimensionSums) and build docsWithFieldSet.
     * Pass 2: iterate vectors again to quantize against the centroid and write binarized data.
     */
    public void mergeOneField(FieldInfo fieldInfo, Supplier<KNNVectorValues<?>> knnVectorValuesSupplier) throws IOException {
        // Pass 1: compute centroid + build docsWithFieldSet
        FieldWriter fieldWriter = new FieldWriter(fieldInfo);
        KNNVectorValues<?> knnVectorValues = knnVectorValuesSupplier.get();
        initializeVectorValues(knnVectorValues);

        int maxDoc = 0;
        int vectorCount = 0;
        while (knnVectorValues.docId() != NO_MORE_DOCS) {
            maxDoc = knnVectorValues.docId();
            fieldWriter.addValueForMerge(knnVectorValues.docId(), (float[]) knnVectorValues.getVector());
            vectorCount++;
            knnVectorValues.nextDoc();
        }

        // Compute cluster center from accumulated sums
        float[] clusterCenter = new float[fieldInfo.getVectorDimension()];
        if (vectorCount > 0) {
            for (int i = 0; i < fieldWriter.dimensionSums.length; i++) {
                clusterCenter[i] = fieldWriter.dimensionSums[i] / vectorCount;
            }
        }

        OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction());

        // Pass 2: quantize and write
        writeFieldForMerge(fieldInfo, clusterCenter, maxDoc + 1, quantizer, knnVectorValuesSupplier, fieldWriter.getDocsWithFieldSet());
        fieldWriter.finish();
    }

    private void writeFieldForMerge(
        FieldInfo fieldInfo,
        float[] clusterCenter,
        int maxDoc,
        OptimizedScalarQuantizer quantizer,
        Supplier<KNNVectorValues<?>> knnVectorValuesSupplier,
        DocsWithFieldSet docsWithFieldSet
    ) throws IOException {
        long vectorDataOffset = binarizedVectorData.alignFilePointer(Float.BYTES);
        writeBinarizedVectorsForMerge(fieldInfo, clusterCenter, quantizer, knnVectorValuesSupplier);
        long vectorDataLength = binarizedVectorData.getFilePointer() - vectorDataOffset;
        float centroidDp = docsWithFieldSet.cardinality() > 0 ? VectorUtil.dotProduct(clusterCenter, clusterCenter) : 0;

        writeMeta(fieldInfo, maxDoc, vectorDataOffset, vectorDataLength, clusterCenter, centroidDp, docsWithFieldSet);
    }

    private void writeBinarizedVectorsForMerge(
        FieldInfo fieldInfo,
        float[] clusterCenter,
        OptimizedScalarQuantizer scalarQuantizer,
        Supplier<KNNVectorValues<?>> knnVectorValuesSupplier
    ) throws IOException {
        int discreteDims = discretize(fieldInfo.getVectorDimension(), 64);
        byte[] quantizationScratch = new byte[discreteDims];
        byte[] vector = new byte[discreteDims / 8];

        // Get a fresh iterator for the second pass
        KNNVectorValues<?> knnVectorValues = knnVectorValuesSupplier.get();
        initializeVectorValues(knnVectorValues);

        while (knnVectorValues.docId() != NO_MORE_DOCS) {
            float[] v = (float[]) knnVectorValues.getVector();
            OptimizedScalarQuantizer.QuantizationResult corrections = scalarQuantizer.scalarQuantize(
                v,
                quantizationScratch,
                INDEX_BITS,
                clusterCenter
            );
            packAsBinary(quantizationScratch, vector);
            binarizedVectorData.writeBytes(vector, vector.length);
            binarizedVectorData.writeInt(Float.floatToIntBits(corrections.lowerInterval()));
            binarizedVectorData.writeInt(Float.floatToIntBits(corrections.upperInterval()));
            binarizedVectorData.writeInt(Float.floatToIntBits(corrections.additionalCorrection()));
            assert corrections.quantizedComponentSum() >= 0 && corrections.quantizedComponentSum() <= 0xffff;
            binarizedVectorData.writeShort((short) corrections.quantizedComponentSum());
            knnVectorValues.nextDoc();
        }
    }

    @Override
    public CloseableRandomVectorScorerSupplier mergeOneFieldToIndex(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(meta, binarizedVectorData);
    }

    @Override
    public long ramBytesUsed() {
        long total = SHALLOW_RAM_BYTES_USED;
        for (FieldWriter field : fields) {
            // the field tracks the delegate field usage
            total += field.ramBytesUsed();
        }
        return total;
    }

    static class FieldWriter extends FlatFieldVectorsWriter<float[]> {
        private static final long SHALLOW_SIZE = shallowSizeOfInstance(FieldWriter.class);
        private final FieldInfo fieldInfo;
        private boolean finished;
        private final float[] dimensionSums;
        private final FloatArrayList magnitudes = new FloatArrayList();
        private final List<float[]> vectors;
        private final DocsWithFieldSet docsWithFieldSet;

        FieldWriter(FieldInfo fieldInfo) {
            this.fieldInfo = fieldInfo;
            this.dimensionSums = new float[fieldInfo.getVectorDimension()];
            this.vectors = new ArrayList<>();
            this.docsWithFieldSet = new DocsWithFieldSet();
        }

        @Override
        public List<float[]> getVectors() {
            return vectors;
        }

        @Override
        public DocsWithFieldSet getDocsWithFieldSet() {
            return docsWithFieldSet;
        }

        @Override
        public void finish() throws IOException {
            if (finished) {
                return;
            }
            finished = true;
        }

        @Override
        public boolean isFinished() {
            return finished;
        }

        @Override
        public void addValue(int docID, float[] vectorValue) throws IOException {
            vectors.add(vectorValue);
            docsWithFieldSet.add(docID);

            for (int i = 0; i < vectorValue.length; i++) {
                dimensionSums[i] += vectorValue[i];
            }
        }

        /**
         * Adds a value for the merge path: accumulates dimension sums and tracks docs,
         * but does NOT store the vector in memory.
         */
        public void addValueForMerge(int docID, float[] vectorValue) {
            docsWithFieldSet.add(docID);
            for (int i = 0; i < vectorValue.length; i++) {
                dimensionSums[i] += vectorValue[i];
            }
        }

        @Override
        public float[] copyValue(float[] vectorValue) {
            throw new UnsupportedOperationException();
        }

        @Override
        public long ramBytesUsed() {
            long size = SHALLOW_SIZE;
            size += magnitudes.ramBytesUsed();
            return size;
        }
    }

    // When accessing vectorValue method, targerOrd here means a row ordinal.
    static class OffHeapBinarizedQueryVectorValues {
        private final IndexInput slice;
        private final int dimension;
        private final int size;
        protected final byte[] binaryValue;
        protected final ByteBuffer byteBuffer;
        private final int byteSize;
        protected final float[] correctiveValues;
        private int lastOrd = -1;
        private int quantizedComponentSum;

        OffHeapBinarizedQueryVectorValues(IndexInput data, int dimension, int size) {
            this.slice = data;
            this.dimension = dimension;
            this.size = size;
            // 4x the quantized binary dimensions
            int binaryDimensions = (discretize(dimension, 64) / 8) * QUERY_BITS;
            this.byteBuffer = ByteBuffer.allocate(binaryDimensions);
            this.binaryValue = byteBuffer.array();
            // + 1 for the quantized sum
            this.correctiveValues = new float[3];
            this.byteSize = binaryDimensions + Float.BYTES * 3 + Short.BYTES;
        }

        public OptimizedScalarQuantizer.QuantizationResult getCorrectiveTerms(int targetOrd) throws IOException {
            if (lastOrd == targetOrd) {
                return new OptimizedScalarQuantizer.QuantizationResult(
                    correctiveValues[0],
                                                                       correctiveValues[1],
                                                                       correctiveValues[2],
                                                                       quantizedComponentSum
                );
            }
            vectorValue(targetOrd);
            return new OptimizedScalarQuantizer.QuantizationResult(
                correctiveValues[0],
                                                                   correctiveValues[1],
                                                                   correctiveValues[2],
                                                                   quantizedComponentSum
            );
        }

        public int size() {
            return size;
        }

        public int quantizedLength() {
            return binaryValue.length;
        }

        public int dimension() {
            return dimension;
        }

        public OffHeapBinarizedQueryVectorValues copy() throws IOException {
            return new OffHeapBinarizedQueryVectorValues(slice.clone(), dimension, size);
        }

        public IndexInput getSlice() {
            return slice;
        }

        public byte[] vectorValue(int targetOrd) throws IOException {
            if (lastOrd == targetOrd) {
                return binaryValue;
            }
            slice.seek((long) targetOrd * byteSize);
            slice.readBytes(binaryValue, 0, binaryValue.length);
            slice.readFloats(correctiveValues, 0, 3);
            quantizedComponentSum = Short.toUnsignedInt(slice.readShort());
            lastOrd = targetOrd;
            return binaryValue;
        }
    }
}
