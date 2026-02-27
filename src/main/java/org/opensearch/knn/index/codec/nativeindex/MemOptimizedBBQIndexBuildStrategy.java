/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import org.apache.lucene.codecs.lucene102.Lucene102BinaryFlatVectorsScorer;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.index.codec.nativeindex.bbq.BBQReader;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.jni.FaissService;

import java.io.IOException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Map;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

public class MemOptimizedBBQIndexBuildStrategy implements NativeIndexBuildStrategy {

    @Override
    public void buildAndWriteIndex(BuildIndexParams indexInfo) throws IOException {
        final KNNVectorValues<?> knnVectorValues = indexInfo.getKnnVectorValuesSupplier().get();
        // Needed to make sure we don't get 0 dimensions while initializing index
        initializeVectorValues(knnVectorValues);
        Map<String, Object> indexParameters = indexInfo.getParameters();
        IndexBuildSetup indexBuildSetup = QuantizationIndexUtils.prepareIndexBuild(knnVectorValues, indexInfo);

        // Load bbq reader
        final SegmentWriteState writeState = indexInfo.getSegmentWriteState();
        final SegmentReadState readState = new SegmentReadState(
            writeState.directory,
            writeState.segmentInfo,
            new FieldInfos(new FieldInfo[] { indexInfo.getFieldInfo() }),
            writeState.context,
            indexInfo.getFieldName()
        );
        final BBQReader bbqReader = new BBQReader(readState, new Lucene102BinaryFlatVectorsScorer(null));
        final FloatVectorValues floatVectorValues = bbqReader.getFloatVectorValues(indexInfo.getFieldName());
        final BBQReader.BinarizedVectorValues binarizedVectorValues = (BBQReader.BinarizedVectorValues) floatVectorValues;
        final int quantizedVecBytes = binarizedVectorValues.quantizedVectorValues.vectorValue(0).length;
        final float centroidDp = binarizedVectorValues.quantizedVectorValues.getCentroidDP();

        // Initialize the index
        final long indexMemoryAddress =
            AccessController.doPrivileged((PrivilegedAction<Long>) () -> FaissService.initBBQIndex(
                indexInfo.getTotalLiveDocs(),
                indexBuildSetup.getDimensions(),
                indexParameters,
                centroidDp,
                quantizedVecBytes
            ));

        // Pass quantized vectors + correction factors
        passQuantizedVectorsCorrectionFactors(indexMemoryAddress, binarizedVectorValues);

        // Transfer document ids
        final int batchSize = 500;
        final int[] docIds = new int[batchSize];
        int numAdded = 0;
        while (knnVectorValues.docId() != NO_MORE_DOCS) {
            int i = 0;
            while (i < batchSize && knnVectorValues.docId() != NO_MORE_DOCS) {
                docIds[i++] = knnVectorValues.docId();
                knnVectorValues.nextDoc();
            }

            FaissService.addDocsToBBQIndex(indexMemoryAddress, docIds, i, numAdded);
            numAdded += i;
        }

        // Write index without flat vectors
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            try {
                FaissService.writeBBQIndex(indexInfo.getIndexOutputWithBuffer(), indexMemoryAddress, indexParameters);
            } catch (Exception e) {
                e.printStackTrace();
                System.out.println("_________________ !!!!!!!!!! " + e.getMessage());
            }
            return null;
        });
    }

    private void passQuantizedVectorsCorrectionFactors(
        final long indexMemoryAddress,
        final BBQReader.BinarizedVectorValues binarizedVectorValues
    ) throws IOException {
        final int batchSize = 500;
        byte[] buffer = null;
        for (int i = 0; i < binarizedVectorValues.size(); ) {
            final int loopSize = Math.min(binarizedVectorValues.size() - i, batchSize);
            for (int j = 0, o = 0; j < loopSize; ++j) {
                final byte[] binaryVector = binarizedVectorValues.quantizedVectorValues.vectorValue(i);
                if (buffer == null) {
                    // [Quantized Vector | lowerInterval (float) | upperInterval (float) | additionalCorrection (float) | quantizedComponentSum (int)]
                    buffer = new byte[(binaryVector.length + Integer.BYTES * 4) * batchSize];
                }
                final OptimizedScalarQuantizer.QuantizationResult quantizationResult =
                    binarizedVectorValues.quantizedVectorValues.getCorrectiveTerms(i);

                // Copy quantized vector
                System.arraycopy(binaryVector, 0, buffer, o, binaryVector.length);
                o += binaryVector.length;

                // Copy correction factors
                int bits = Float.floatToRawIntBits(quantizationResult.lowerInterval());
                buffer[o++] = (byte) (bits);
                buffer[o++] = (byte) (bits >>> 8);
                buffer[o++] = (byte) (bits >>> 16);
                buffer[o++] = (byte) (bits >>> 24);

                bits = Float.floatToRawIntBits(quantizationResult.upperInterval());
                buffer[o++] = (byte) (bits);
                buffer[o++] = (byte) (bits >>> 8);
                buffer[o++] = (byte) (bits >>> 16);
                buffer[o++] = (byte) (bits >>> 24);

                bits = Float.floatToRawIntBits(quantizationResult.additionalCorrection());
                buffer[o++] = (byte) (bits);
                buffer[o++] = (byte) (bits >>> 8);
                buffer[o++] = (byte) (bits >>> 16);
                buffer[o++] = (byte) (bits >>> 24);

                bits = quantizationResult.quantizedComponentSum();
                buffer[o++] = (byte) (bits);
                buffer[o++] = (byte) (bits >>> 8);
                buffer[o++] = (byte) (bits >>> 16);
                buffer[o++] = (byte) (bits >>> 24);
            }

            FaissService.passBBQVectors(indexMemoryAddress, buffer, loopSize);

            i += loopSize;
        }
    }
}
