/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;
import org.opensearch.knn.jni.FaissService;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.util.Map;
import java.util.function.Supplier;

/**
 * A factory class that provides various methods to create the {@link KNNVectorValues}.
 */
public final class KNNVectorValuesFactory {

    private static final Logger log = LogManager.getLogger(KNNVectorValuesFactory.class);

    /**
     * Returns a {@link KNNVectorValues} for the given {@link DocIdSetIterator} and {@link VectorDataType}
     *
     * @param vectorDataType {@link VectorDataType}
     * @param knnVectorValues {@link KnnVectorValues}
     * @return {@link KNNVectorValues}
     */
    public static <T> KNNVectorValues<T> getVectorValues(final VectorDataType vectorDataType, final KnnVectorValues knnVectorValues) {
        return getVectorValues(vectorDataType, new KNNVectorValuesIterator.DocIdsIteratorValues(knnVectorValues));
    }

    /**
     * Returns a {@link Supplier} for {@link #getVectorValues(VectorDataType, KnnVectorValues)}
     * Note: This class is public static so that it can be mocked for testing.
     *
     * @param vectorDataType {@link VectorDataType}
     * @param knnVectorValues {@link KnnVectorValues}
     * @return {@link KNNVectorValues}
     */
    public static <T> Supplier<KNNVectorValues<?>> getVectorValuesSupplier(
        final VectorDataType vectorDataType,
        final KnnVectorValues knnVectorValues
    ) {
        return () -> getVectorValues(vectorDataType, new KNNVectorValuesIterator.DocIdsIteratorValues(knnVectorValues));
    }

    public static <T> KNNVectorValues<T> getVectorValues(final VectorDataType vectorDataType, final DocIdSetIterator docIdSetIterator) {
        return getVectorValues(vectorDataType, new KNNVectorValuesIterator.DocIdsIteratorValues(docIdSetIterator));
    }

    /**
     * Returns a {@link KNNVectorValues} for the given {@link DocIdSetIterator} and a Map of docId and vectors.
     *
     * @param vectorDataType {@link VectorDataType}
     * @param docIdWithFieldSet {@link DocsWithFieldSet}
     * @return {@link KNNVectorValues}
     */
    public static <T> KNNVectorValues<T> getVectorValues(
        final VectorDataType vectorDataType,
        final DocsWithFieldSet docIdWithFieldSet,
        final Map<Integer, T> vectors
    ) {
        return getVectorValues(vectorDataType, new KNNVectorValuesIterator.FieldWriterIteratorValues<T>(docIdWithFieldSet, vectors));
    }

    /**
     * Returns a {@link Supplier} for {@link #getVectorValuesSupplier(VectorDataType, DocsWithFieldSet, Map)}.
     * Note: This class is public static so that it can be mocked for testing.
     *
     * @param vectorDataType {@link VectorDataType}
     * @param docIdWithFieldSet {@link DocsWithFieldSet}
     * @return {@link KNNVectorValues}
     */
    public static <T> Supplier<KNNVectorValues<?>> getVectorValuesSupplier(
        final VectorDataType vectorDataType,
        final DocsWithFieldSet docIdWithFieldSet,
        final Map<Integer, T> vectors
    ) {
        return () -> getVectorValues(vectorDataType, docIdWithFieldSet, vectors);
    }

    /**
     * Returns a {@link KNNVectorValues} for the given {@link FieldInfo} and {@link LeafReader}
     *
     * @param fieldInfo {@link FieldInfo}
     * @param leafReader {@link LeafReader}
     * @return {@link KNNVectorValues}
     */
    public static <T> KNNVectorValues<T> getVectorValues(final FieldInfo fieldInfo, final LeafReader leafReader) throws IOException {
        final DocIdSetIterator docIdSetIterator;
        if (!fieldInfo.hasVectorValues()) {
            docIdSetIterator = DocValues.getBinary(leafReader, fieldInfo.getName());
            final KNNVectorValuesIterator vectorValuesIterator = new KNNVectorValuesIterator.DocIdsIteratorValues(docIdSetIterator);
            return getVectorValues(FieldInfoExtractor.extractVectorDataType(fieldInfo), vectorValuesIterator);
        }
        if (fieldInfo.getVectorEncoding() == VectorEncoding.BYTE) {
            return getVectorValues(
                FieldInfoExtractor.extractVectorDataType(fieldInfo),
                new KNNVectorValuesIterator.DocIdsIteratorValues(leafReader.getByteVectorValues(fieldInfo.getName()))
            );
        } else if (fieldInfo.getVectorEncoding() == VectorEncoding.FLOAT32) {
            return getVectorValues(
                FieldInfoExtractor.extractVectorDataType(fieldInfo),
                new KNNVectorValuesIterator.DocIdsIteratorValues(leafReader.getFloatVectorValues(fieldInfo.getName()))
            );
        } else {
            throw new IllegalArgumentException("Invalid Vector encoding provided, hence cannot return VectorValues");
        }
    }

    public static <T> KNNVectorValues<T> getFloatVectorValuesForPoc(final FieldInfo fieldInfo, final NativeMemoryAllocation indexAllocation)
        throws IOException {
        System.out.println("______________ getFloatVectorValuesForPoc");

        // It's never null! We're doing POC!
        assert (indexAllocation != null);

        // Returned info
        // - info[0] = Flat vector pointer
        // - info[1] = Dimension
        // - info[2] = Total #vectors

        final long[] info = new long[3];
        FaissService.getFlatVectorValues(indexAllocation.getMemoryAddress(), info);
        final long flatVectorAddress = info[0];
        final int dimension = (int) info[1];
        final int totalNumberOfVectors = (int) info[2];
        final long oneVectorSize = 4L * dimension;
        final long flatVectorBlockSize = oneVectorSize * totalNumberOfVectors;
        final MemorySegment flatVectors = MemorySegment.ofAddress(flatVectorAddress).reinterpret(flatVectorBlockSize);

        final FloatVectorValues floatVectorValues = new FloatVectorValues() {
            final float[] buffer = new float[dimension];

            @Override
            public DocIndexIterator iterator() {
                return new DocIndexIterator() {
                    int docId = -1;

                    @Override
                    public int docID() {
                        System.out.println("______________ docID(), docId=" + docId);
                        return docId;
                    }

                    @Override
                    public int nextDoc() throws IOException {
                        System.out.println("_______________nextDoc, docId=" + docId);
                        ++docId;
                        if (docId < totalNumberOfVectors) {
                            return docId;
                        }
                        return docId = NO_MORE_DOCS;
                    }

                    @Override
                    public int advance(int i) throws IOException {
                        System.out.println("______________ advance, i=" + i + ", docId=" + docId);
                        docId = i;
                        if (docId < totalNumberOfVectors) {
                            return docId;
                        }
                        return docId = NO_MORE_DOCS;
                    }

                    @Override
                    public long cost() {
                        return totalNumberOfVectors;
                    }

                    @Override
                    public int index() {
                        // It's dense case, where docId == ord
                        return docId;
                    }
                };
            }

            @Override
            public float[] vectorValue(int internalVectorId) throws IOException {
                System.out.println("________________ vectorValue, internalVectorId=" + internalVectorId);
                flatVectors.asSlice(internalVectorId * oneVectorSize).asByteBuffer().asFloatBuffer().get(buffer);
                return buffer;
            }

            @Override
            public FloatVectorValues copy() throws IOException {
                throw new UnsupportedOperationException();
            }

            @Override
            public int dimension() {
                return dimension;
            }

            @Override
            public int size() {
                return totalNumberOfVectors;
            }
        };

        return getVectorValues(
            FieldInfoExtractor.extractVectorDataType(fieldInfo),
            new KNNVectorValuesIterator.DocIdsIteratorValues(floatVectorValues)
        );
    }

    /**
     * Returns a {@link KNNVectorValues} for the given {@link FieldInfo} and {@link LeafReader}
     *
     * @param fieldInfo {@link FieldInfo}
     * @param docValuesProducer {@link DocValuesProducer}
     * @param knnVectorsReader {@link KnnVectorsReader}
     * @return {@link KNNVectorValues}
     */
    public static <T> KNNVectorValues<T> getVectorValues(
        final FieldInfo fieldInfo,
        final DocValuesProducer docValuesProducer,
        final KnnVectorsReader knnVectorsReader
    ) throws IOException {
        if (fieldInfo.hasVectorValues() && knnVectorsReader != null) {
            final KnnVectorValues knnVectorValues;
            if (fieldInfo.getVectorEncoding() == VectorEncoding.BYTE) {
                knnVectorValues = knnVectorsReader.getByteVectorValues(fieldInfo.getName());
            } else if (fieldInfo.getVectorEncoding() == VectorEncoding.FLOAT32) {
                knnVectorValues = knnVectorsReader.getFloatVectorValues(fieldInfo.getName());
            } else {
                throw new IllegalArgumentException("Invalid Vector encoding provided, hence cannot return VectorValues");
            }
            return getVectorValues(
                FieldInfoExtractor.extractVectorDataType(fieldInfo),
                new KNNVectorValuesIterator.DocIdsIteratorValues(knnVectorValues)
            );
        } else if (docValuesProducer != null) {
            return getVectorValues(
                FieldInfoExtractor.extractVectorDataType(fieldInfo),
                new KNNVectorValuesIterator.DocIdsIteratorValues(docValuesProducer.getBinary(fieldInfo))
            );
        } else {
            throw new IllegalArgumentException("Field does not have vector values and DocValues");
        }
    }

    @SuppressWarnings("unchecked")
    private static <T> KNNVectorValues<T> getVectorValues(
        final VectorDataType vectorDataType,
        final KNNVectorValuesIterator knnVectorValuesIterator
    ) {
        switch (vectorDataType) {
            case FLOAT:
                return (KNNVectorValues<T>) new KNNFloatVectorValues(knnVectorValuesIterator);
            case BYTE:
                return (KNNVectorValues<T>) new KNNByteVectorValues(knnVectorValuesIterator);
            case BINARY:
                return (KNNVectorValues<T>) new KNNBinaryVectorValues(knnVectorValuesIterator);
        }
        throw new IllegalArgumentException("Invalid Vector data type provided, hence cannot return VectorValues");
    }

    /**
     * Retrieves the {@link KNNVectorValues} for a specific field during a merge operation, based on the vector data type.
     *
     * @param vectorDataType The {@link VectorDataType} representing the type of vectors stored.
     * @param fieldInfo      The {@link FieldInfo} object containing metadata about the field.
     * @param mergeState     The {@link org.apache.lucene.index.MergeState} representing the state of the merge operation.
     * @param <T>            The type of vectors being processed.
     * @return The {@link KNNVectorValues} associated with the field during the merge.
     * @throws IOException If an I/O error occurs during the retrieval.
     */
    private static <T> KNNVectorValues<T> getKNNVectorValuesForMerge(
        final VectorDataType vectorDataType,
        final FieldInfo fieldInfo,
        final MergeState mergeState
    ) {
        try {
            switch (fieldInfo.getVectorEncoding()) {
                case FLOAT32:
                    FloatVectorValues mergedFloats = KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
                    return getVectorValues(vectorDataType, mergedFloats);
                case BYTE:
                    ByteVectorValues mergedBytes = KnnVectorsWriter.MergedVectorValues.mergeByteVectorValues(fieldInfo, mergeState);
                    return getVectorValues(vectorDataType, mergedBytes);
                default:
                    throw new IllegalStateException("Unsupported vector encoding [" + fieldInfo.getVectorEncoding() + "]");
            }
        } catch (final IOException e) {
            log.error("Unable to merge vectors for field [{}]", fieldInfo.getName(), e);
            throw new IllegalStateException("Unable to merge vectors for field [" + fieldInfo.getName() + "]", e);
        }
    }

    /**
     * Returns a {@link Supplier} for {@link #getKNNVectorValuesForMerge(VectorDataType, FieldInfo, MergeState)}.
     * Note: This class is public static so that it can be mocked for testing.
     *
     * @param vectorDataType
     * @param fieldInfo
     * @param mergeState
     * @return
     */
    public static Supplier<KNNVectorValues<?>> getKNNVectorValuesSupplierForMerge(
        final VectorDataType vectorDataType,
        final FieldInfo fieldInfo,
        final MergeState mergeState
    ) {
        return () -> getKNNVectorValuesForMerge(vectorDataType, fieldInfo, mergeState);
    }
}
