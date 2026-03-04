/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.bbq;

import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene90.IndexedDISI;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

import java.io.IOException;
import java.nio.ByteBuffer;

import static org.apache.lucene.util.VectorUtil.scaleMaxInnerProductScore;
import static org.apache.lucene.util.quantization.OptimizedScalarQuantizer.discretize;

public abstract class OffHeapBinarizedVectorValues extends BinarizedByteVectorValues {
    final int dimension;
    final int size;
    final int numBytes;
    final VectorSimilarityFunction similarityFunction;
    final FlatVectorsScorer vectorsScorer;

    final IndexInput slice;
    final byte[] binaryValue;
    final ByteBuffer byteBuffer;
    final int byteSize;
    private int lastOrd = -1;
    final float[] correctiveValues;
    int quantizedComponentSum;
    final OptimizedScalarQuantizer binaryQuantizer;
    final float[] centroid;
    final float centroidDp;
    private final int discretizedDimensions;
    // Rerank support
    private float[] centeredQueryVector;
    final int errorResidualBits;
    final int residualNumBytes;
    final int primaryBlockSize;

    OffHeapBinarizedVectorValues(
        int dimension,
        int size,
        float[] centroid,
        float centroidDp,
        OptimizedScalarQuantizer quantizer,
        VectorSimilarityFunction similarityFunction,
        FlatVectorsScorer vectorsScorer,
        int errorResidualBits,
        IndexInput slice
    ) {
        this.dimension = dimension;
        this.size = size;
        this.similarityFunction = similarityFunction;
        this.vectorsScorer = vectorsScorer;
        this.slice = slice;
        this.centroid = centroid;
        this.centroidDp = centroidDp;
        this.numBytes = discretize(dimension, 64) / 8;
        this.correctiveValues = new float[3];
        this.primaryBlockSize = numBytes + (Float.BYTES * 3) + Short.BYTES;
        this.errorResidualBits = errorResidualBits;
        this.residualNumBytes = (errorResidualBits > 0)
            ? discretize(dimension * errorResidualBits, 64) / 8
            : 0;
        int residualBlockSize = (errorResidualBits > 0)
            ? residualNumBytes + (Float.BYTES * 3) + Short.BYTES
            : 0;
        this.byteSize = primaryBlockSize + residualBlockSize;
        this.byteBuffer = ByteBuffer.allocate(numBytes);
        this.binaryValue = byteBuffer.array();
        this.binaryQuantizer = quantizer;
        this.discretizedDimensions = discretize(dimension, 64);
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public byte[] vectorValue(int targetOrd) throws IOException {
        if (lastOrd == targetOrd) {
            return binaryValue;
        }
        slice.seek((long) targetOrd * byteSize);
        slice.readBytes(byteBuffer.array(), byteBuffer.arrayOffset(), numBytes);
        slice.readFloats(correctiveValues, 0, 3);
        quantizedComponentSum = Short.toUnsignedInt(slice.readShort());
        lastOrd = targetOrd;
        return binaryValue;
    }

    @Override
    public int discretizedDimensions() {
        return discretizedDimensions;
    }

    @Override
    public float getCentroidDP() {
        return centroidDp;
    }

    @Override
    public OptimizedScalarQuantizer.QuantizationResult getCorrectiveTerms(int targetOrd) throws IOException {
        if (lastOrd == targetOrd) {
            return new OptimizedScalarQuantizer.QuantizationResult(
                correctiveValues[0],
                correctiveValues[1],
                correctiveValues[2],
                quantizedComponentSum
            );
        }
        slice.seek(((long) targetOrd * byteSize) + numBytes);
        slice.readFloats(correctiveValues, 0, 3);
        quantizedComponentSum = Short.toUnsignedInt(slice.readShort());
        return new OptimizedScalarQuantizer.QuantizationResult(
            correctiveValues[0],
            correctiveValues[1],
            correctiveValues[2],
            quantizedComponentSum
        );
    }

    @Override
    public OptimizedScalarQuantizer getQuantizer() {
        return binaryQuantizer;
    }

    @Override
    public float[] getCentroid() {
        return centroid;
    }

    @Override
    public int getVectorByteLength() {
        return numBytes;
    }

    /**
     * Pre-compute query-centroid dot product for reranking.
     *
     * @param queryVector the query vector
     */
    public void initRerank(float[] queryVector) {
        this.centeredQueryVector = queryVector;
        for (int i = 0 ; i < this.centeredQueryVector.length ; ++i) {
            this.centeredQueryVector[i] -= centroid[i];
        }
    }

    /**
     * Read the quantized error residual for the given vector ordinal.
     * Seeks to the residual block which starts at targetOrd * byteSize + byteSize / 2.
     *
     * @param targetOrd the vector ordinal
     * @return the quantized error residual with its correction factors
     * @throws IOException if an I/O error occurs
     */
    public BinarizedByteVectorValues.QuantizedErrorResidual getQuantizedErrorResidual(int targetOrd) throws IOException {
        // Jump to the residual block: after the primary block
        slice.seek((long) targetOrd * byteSize + primaryBlockSize);
        byte[] residualBytes = new byte[residualNumBytes];
        slice.readBytes(residualBytes, 0, residualNumBytes);
        float[] resCorrectiveValues = new float[3];
        slice.readFloats(resCorrectiveValues, 0, 3);
        int resQuantizedComponentSum = Short.toUnsignedInt(slice.readShort());
        return new BinarizedByteVectorValues.QuantizedErrorResidual(
            residualBytes,
                                                                    resCorrectiveValues[0],
                                                                    resCorrectiveValues[1],
                                                                    resCorrectiveValues[2],
                                                                    resQuantizedComponentSum
        );
    }

    /**
     * Rerank a candidate by computing a refined score using the error residual.
     * Score = &lt;q, centroid&gt; + &lt;q, Q(x)&gt; + &lt;q, Q_res(r)&gt;
     * where &lt;q, centroid&gt; is pre-computed, &lt;q, Q(x)&gt; is the approximated score from first pass,
     * and &lt;q, Q_res(r)&gt; is computed from the quantized error residual.
     *
     * @param targetOrd            the vector ordinal
     * @param approximatedMipScore the max-inner-product scaled score from the first pass
     * @return the refined max-inner-product scaled score
     * @throws IOException if an I/O error occurs
     */
    public float rerank(int targetOrd, float approximatedMipScore) throws IOException {
        // Reverse scaleMaxInnerProductScore to get raw dot product
        float rawApproxDot = unscaleMaxInnerProductScore(approximatedMipScore);

        BinarizedByteVectorValues.QuantizedErrorResidual residual = getQuantizedErrorResidual(targetOrd);
        float rLow = residual.lowerInterval();
        float rHigh = residual.upperInterval();
        byte[] packedResidual = residual.quantizedErrorResidual();

        // Compute <q, Q_res(r)> by dequantizing the residual and dotting with query
        float qDotResidual = 0f;
        int nSteps = (1 << errorResidualBits) - 1;
        float step = (nSteps > 0) ? (rHigh - rLow) / nSteps : 0f;

        if (errorResidualBits == 1) {
            // Fast path for 1-bit: each dimension is 1 bit, packed 8 per byte
            for (int d = 0; d < dimension; d++) {
                int bit = (packedResidual[d >> 3] >> (7 - (d & 7))) & 1;
                float dequantized = (bit == 0) ? rLow : rHigh;
                qDotResidual += centeredQueryVector[d] * dequantized;
            }
        } else {
            // General path for multi-bit: extract B-bit codes from packed bytes
            for (int d = 0; d < dimension; d++) {
                int bitOffset = d * errorResidualBits;
                int code = 0;
                for (int b = 0; b < errorResidualBits; b++) {
                    int globalBit = bitOffset + b;
                    int byteIdx = globalBit / 8;
                    int bitIdx = 7 - (globalBit % 8);
                    code = (code << 1) | ((packedResidual[byteIdx] >> bitIdx) & 1);
                }
                float dequantized = rLow + code * step;
                qDotResidual += centeredQueryVector[d] * dequantized;
            }
        }

        // rawApproxDot already includes <q, centroid> from the first-pass scorer
        float rawTotal = rawApproxDot + qDotResidual;
        return scaleMaxInnerProductScore(rawTotal);
    }
    /**
     * Reverse of {@link org.apache.lucene.util.VectorUtil#scaleMaxInnerProductScore(float)}.
     * Recovers the raw dot product from a max-inner-product scaled score.
     *
     * @param score the scaled MIP score
     * @return the raw dot product value
     */
    static float unscaleMaxInnerProductScore(float score) {
        if (score <= 1.0f) {
            // Was negative dot product: score = 1 / (1 + (-dp)) => dp = -(1/score - 1)
            return -(1.0f / score - 1.0f);
        }
        // Was non-negative dot product: score = dp + 1 => dp = score - 1
        return score - 1.0f;
    }

    static OffHeapBinarizedVectorValues load(
        OrdToDocDISIReaderConfiguration configuration,
        int dimension,
        int size,
        OptimizedScalarQuantizer binaryQuantizer,
        VectorSimilarityFunction similarityFunction,
        FlatVectorsScorer vectorsScorer,
        float[] centroid,
        float centroidDp,
        int errorResidualBits,
        long quantizedVectorDataOffset,
        long quantizedVectorDataLength,
        IndexInput vectorData
    ) throws IOException {
        if (configuration.isEmpty()) {
            return new EmptyOffHeapVectorValues(dimension, similarityFunction, vectorsScorer);
        }
        assert centroid != null;
        IndexInput bytesSlice = vectorData.slice("quantized-vector-data", quantizedVectorDataOffset, quantizedVectorDataLength);
        if (configuration.isDense()) {
            return new DenseOffHeapVectorValues(
                dimension, size, centroid, centroidDp, binaryQuantizer,
                similarityFunction, vectorsScorer, errorResidualBits, bytesSlice
            );
        } else {
            return new SparseOffHeapVectorValues(
                configuration, dimension, size, centroid, centroidDp, binaryQuantizer,
                vectorData, similarityFunction, vectorsScorer, errorResidualBits, bytesSlice
            );
        }
    }

    /** Dense off-heap binarized vector values */
    static class DenseOffHeapVectorValues extends OffHeapBinarizedVectorValues {
        DenseOffHeapVectorValues(
            int dimension, int size, float[] centroid, float centroidDp,
            OptimizedScalarQuantizer binaryQuantizer, VectorSimilarityFunction similarityFunction,
            FlatVectorsScorer vectorsScorer, int errorResidualBits, IndexInput slice
        ) {
            super(dimension, size, centroid, centroidDp, binaryQuantizer, similarityFunction, vectorsScorer, errorResidualBits, slice);
        }

        @Override
        public DenseOffHeapVectorValues copy() throws IOException {
            return new DenseOffHeapVectorValues(
                dimension, size, centroid, centroidDp, binaryQuantizer,
                similarityFunction, vectorsScorer, errorResidualBits, slice.clone()
            );
        }

        @Override
        public Bits getAcceptOrds(Bits acceptDocs) {
            return acceptDocs;
        }

        @Override
        public VectorScorer scorer(float[] target) throws IOException {
            DenseOffHeapVectorValues copy = copy();
            DocIndexIterator iterator = copy.iterator();
            RandomVectorScorer scorer = vectorsScorer.getRandomVectorScorer(similarityFunction, copy, target);
            return new VectorScorer() {
                @Override
                public float score() throws IOException {
                    return scorer.score(iterator.index());
                }

                @Override
                public DocIdSetIterator iterator() {
                    return iterator;
                }
            };
        }

        @Override
        public DocIndexIterator iterator() {
            return createDenseIterator();
        }
    }

    /** Sparse off-heap binarized vector values */
    private static class SparseOffHeapVectorValues extends OffHeapBinarizedVectorValues {
        private final DirectMonotonicReader ordToDoc;
        private final IndexedDISI disi;
        // dataIn was used to init a new IndexedDIS for #randomAccess()
        private final IndexInput dataIn;
        private final OrdToDocDISIReaderConfiguration configuration;

        SparseOffHeapVectorValues(
            OrdToDocDISIReaderConfiguration configuration,
            int dimension, int size, float[] centroid, float centroidDp,
            OptimizedScalarQuantizer binaryQuantizer, IndexInput dataIn,
            VectorSimilarityFunction similarityFunction, FlatVectorsScorer vectorsScorer,
            int errorResidualBits, IndexInput slice
        ) throws IOException {
            super(dimension, size, centroid, centroidDp, binaryQuantizer, similarityFunction, vectorsScorer, errorResidualBits, slice);
            this.configuration = configuration;
            this.dataIn = dataIn;
            this.ordToDoc = configuration.getDirectMonotonicReader(dataIn);
            this.disi = configuration.getIndexedDISI(dataIn);
        }

        @Override
        public SparseOffHeapVectorValues copy() throws IOException {
            return new SparseOffHeapVectorValues(
                configuration, dimension, size, centroid, centroidDp, binaryQuantizer,
                dataIn, similarityFunction, vectorsScorer, errorResidualBits, slice.clone()
            );
        }

        @Override
        public int ordToDoc(int ord) {
            return (int) ordToDoc.get(ord);
        }

        @Override
        public Bits getAcceptOrds(Bits acceptDocs) {
            if (acceptDocs == null) {
                return null;
            }
            return new Bits() {
                @Override
                public boolean get(int index) {
                    return acceptDocs.get(ordToDoc(index));
                }

                @Override
                public int length() {
                    return size;
                }
            };
        }

        @Override
        public DocIndexIterator iterator() {
            return IndexedDISI.asDocIndexIterator(disi);
        }

        @Override
        public VectorScorer scorer(float[] target) throws IOException {
            SparseOffHeapVectorValues copy = copy();
            DocIndexIterator iterator = copy.iterator();
            RandomVectorScorer scorer = vectorsScorer.getRandomVectorScorer(similarityFunction, copy, target);
            return new VectorScorer() {
                @Override
                public float score() throws IOException {
                    return scorer.score(iterator.index());
                }

                @Override
                public DocIdSetIterator iterator() {
                    return iterator;
                }
            };
        }
    }

    private static class EmptyOffHeapVectorValues extends OffHeapBinarizedVectorValues {
        EmptyOffHeapVectorValues(int dimension, VectorSimilarityFunction similarityFunction, FlatVectorsScorer vectorsScorer) {
            super(dimension, 0, null, Float.NaN, null, similarityFunction, vectorsScorer, 0, null);
        }

        @Override
        public DocIndexIterator iterator() {
            return createDenseIterator();
        }

        @Override
        public DenseOffHeapVectorValues copy() {
            throw new UnsupportedOperationException();
        }

        @Override
        public Bits getAcceptOrds(Bits acceptDocs) {
            return null;
        }

        @Override
        public VectorScorer scorer(float[] target) {
            return null;
        }
    }
}
