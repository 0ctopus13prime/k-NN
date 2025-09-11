/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.Getter;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;

public class MMapByteVectorValues extends ByteVectorValues {
    private final IndexInput indexInput;
    @Getter
    private final long oneVectorByteSize;
    @Getter
    private final long baseOffset;
    private final int dimension;
    private final int totalNumberOfVectors;
    private byte[] buffer;
    @Getter
    private MemorySegment[] memorySegments;

    public MMapByteVectorValues(
        final IndexInput indexInput,
        final long oneVectorByteSize,
        final long baseOffset,
        final int dimension,
        final int totalNumberOfVectors
    ) {
        this.indexInput = indexInput;
        this.oneVectorByteSize = oneVectorByteSize;
        this.baseOffset = baseOffset;
        this.dimension = dimension;
        this.totalNumberOfVectors = totalNumberOfVectors;

        try {
            Field f = indexInput.getClass().getSuperclass().getDeclaredField("segments");
            f.setAccessible(true);
            this.memorySegments = (MemorySegment[]) f.get(indexInput);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public byte[] vectorValue(int internalVectorId) throws IOException {
        indexInput.seek(baseOffset + internalVectorId * oneVectorByteSize);
        if (buffer == null) {
            buffer = new byte[(int) oneVectorByteSize];
        }
        indexInput.readBytes(buffer, 0, buffer.length);
        return buffer;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public int size() {
        return totalNumberOfVectors;
    }

    @Override
    public ByteVectorValues copy() throws IOException {
        return new MMapByteVectorValues(indexInput.clone(), oneVectorByteSize, baseOffset, dimension, totalNumberOfVectors);
    }
}
