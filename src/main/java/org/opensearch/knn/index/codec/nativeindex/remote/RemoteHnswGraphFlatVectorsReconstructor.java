/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import lombok.Getter;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.memoryoptsearch.faiss.FaissEmptyFlatVectors;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndexLoadUtils;
import org.opensearch.knn.memoryoptsearch.faiss.UnsupportedFaissIndexException;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryIndex;

import java.io.IOException;

import static org.opensearch.knn.memoryoptsearch.faiss.FaissIdMapIndex.IXMP;

public final class RemoteHnswGraphFlatVectorsReconstructor {
    public static void reconstruct(final IndexInput graphInput, final IndexInput flatVectorsInput, final IndexOutput output)
        throws IOException {
        /*
        Output sections
        1. IndexIDMap metadata
        2. IndexHNSW metadata
        3. HNSW graph structure
        4. IndexFlat metadata
        5. Flat Vectors
        6. id_map mappings
         */

        final String indexType = FaissIndexLoadUtils.readIndexType(graphInput);
        final FaissIdMapIndexParser graphOffsetParser = new FaissIdMapIndexParser(indexType);
        graphOffsetParser.calculateOffset(graphInput);

        byte[] buffer = new byte[16 * 1024];

        // Copy Section 1-3
        writeBytesFromIndexInput(graphInput, output, 0, graphOffsetParser.getHnswGraphEndOffset(), buffer);

        // Copy Flat vectors
        writeBytesFromIndexInput(flatVectorsInput, output, 0, flatVectorsInput.length(), buffer);

        // Copy id-mapping
        writeBytesFromIndexInput(graphInput, output, graphOffsetParser.getIdMappingStartOffset(), graphInput.length(), buffer);
    }

    private static void writeBytesFromIndexInput(
        final IndexInput input,
        final IndexOutput output,
        long startOffset,
        long endOffset,
        byte[] buffer
    ) throws IOException {
        input.seek(startOffset);
        long remaining = endOffset - startOffset;
        while (remaining > 0) {
            int toRead = (int) Math.min(buffer.length, remaining);
            input.readBytes(buffer, 0, toRead, false);
            output.writeBytes(buffer, 0, toRead);
            remaining -= toRead;
        }
    }

    @Getter
    private static class FaissIdMapIndexParser extends FaissBinaryIndex {
        private long hnswGraphEndOffset;
        private long idMappingStartOffset;

        public FaissIdMapIndexParser(final String indexType) {
            super(indexType);
        }

        public void calculateOffset(final IndexInput input) throws IOException {
            // Read id-map type.
            if (indexType.equals(IXMP)) {
                readCommonHeader(input);
            } else {
                readBinaryCommonHeader(input);
            }

            // Load nested HNSW + null flat vectors
            FaissIndex.load(input);

            // Get rid of "null" string
            hnswGraphEndOffset = input.getFilePointer() - FaissEmptyFlatVectors.NULL.length();

            // Save starting offset of id mapping table
            idMappingStartOffset = input.getFilePointer();
        }

        @Override
        protected void doLoad(IndexInput input) {
            throw new UnsupportedFaissIndexException("NO!");
        }

        @Override
        public VectorEncoding getVectorEncoding() {
            throw new UnsupportedFaissIndexException("NO!");
        }

        @Override
        public FloatVectorValues getFloatValues(IndexInput indexInput) {
            throw new UnsupportedFaissIndexException("NO!");
        }

        @Override
        public ByteVectorValues getByteValues(IndexInput indexInput) {
            throw new UnsupportedFaissIndexException("NO!");
        }
    }
}
