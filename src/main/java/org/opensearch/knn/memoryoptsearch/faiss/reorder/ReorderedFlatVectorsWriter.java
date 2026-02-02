/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ReorderedFlatVectorsWriter implements Closeable {
    private final IndexOutput meta, vectorData;
    private final List<ReorderedFlatFieldVectorsWriter> fields = new ArrayList<>();
    private final int numDocs;
    private final int numLevel = 4;
    private final int numDocsForGrouping = 256;
    private final int groupFactor = 4;

    public ReorderedFlatVectorsWriter(
        final Directory directory,
        final String metaFileName,
        final String flatVectorFileName,
        final int numDocs
    ) throws IOException {
        this.numDocs = numDocs;
        meta = directory.createOutput(metaFileName, IOContext.DEFAULT);
        vectorData = directory.createOutput(flatVectorFileName, IOContext.DEFAULT);
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(meta, vectorData);
    }

    public ReorderedFlatFieldVectorsWriter<?> addField(final FieldInfo fieldInfo) throws IOException {
        // TODO : Byte index (dense + sparse), Float index (sparse)
        fields.add(new ReorderedDenseFloatFlatFieldVectorsWriter(fieldInfo));
        return fields.getLast();
    }

    private class ReorderedDenseFloatFlatFieldVectorsWriter extends ReorderedFlatFieldVectorsWriter<float[]> {
        private Long[] docAndOrds;
        private int docIdx;
        private int ord;
        private long vecStartOffset;
        private final ByteBuffer buffer;

        public ReorderedDenseFloatFlatFieldVectorsWriter(final FieldInfo fieldInfo) {
            this.docAndOrds = new Long[numDocs];
            this.docIdx = 0;
            this.ord = 0;
            this.vecStartOffset = vectorData.getFilePointer();
            this.buffer = ByteBuffer.allocate(fieldInfo.getVectorDimension() * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        }

        @Override
        public void addValue(final int docID, final float[] vector) throws IOException {
            docAndOrds[docIdx++] = ((long) docID) << 32 | ord;
            buffer.asFloatBuffer().put(vector);
            vectorData.writeBytes(buffer.array(), buffer.array().length);
        }

        @Override
        public void finish() throws IOException {
            // Sort by doc ids in asc
            Arrays.sort(
                docAndOrds, (a, b) -> {
                    final int docA = (int) (a >>> 32); // extract upper 32 bits
                    final int docB = (int) (b >>> 32);
                    return Integer.compare(docA, docB);
                }
            );

            // Create skip list
            final DocIdOrdSkipListIndexBuilder skipListIndexBuilder =
                new DocIdOrdSkipListIndexBuilder(
                    true,
                    numLevel,
                    numDocsForGrouping,
                    groupFactor,
                    meta
                );
            // Is dense = true
            meta.writeByte((byte) 1);
            // Max doc id
            meta.writeInt((int) (docAndOrds[docAndOrds.length - 1] >>> 32));
            // Flush a skip list
            for (final long docAndOrd : docAndOrds) {
                skipListIndexBuilder.add((int) (docAndOrd >>> 32), (int) docAndOrd);
            }
            skipListIndexBuilder.finish();
        }
    }

    public static abstract class ReorderedFlatFieldVectorsWriter<T> {
        public abstract void addValue(int docID, T vectorValue) throws IOException;

        public abstract void finish() throws IOException;
    }
}
