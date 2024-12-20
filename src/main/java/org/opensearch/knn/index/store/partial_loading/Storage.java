/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store.partial_loading;

import org.apache.lucene.store.IndexInput;

import java.io.IOException;

public class Storage {
    public long baseOffset;
    public long regionSize;
    public int elementSize = 1;

    public void loadBlock(IndexInput readStream) throws IOException {
        this.regionSize = readStream.readLong() * elementSize;
        this.baseOffset = readStream.getFilePointer();
        readStream.seek(baseOffset + regionSize);
    }

    public void readBytes(IndexInput indexInput, long offset, byte[] bytesBuffer) throws IOException {
        indexInput.seek(baseOffset + offset);
        indexInput.readBytes(bytesBuffer, 0, bytesBuffer.length);
    }
}
