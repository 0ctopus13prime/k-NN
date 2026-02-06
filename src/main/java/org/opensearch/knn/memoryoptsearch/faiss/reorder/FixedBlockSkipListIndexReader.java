/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.store.IndexInput;

import java.io.IOException;

public class FixedBlockSkipListIndexReader {
    private final long[] blocks;
    public final int maxDoc;
    private final int numBytesPerValue;
    private int doc;

    public FixedBlockSkipListIndexReader(final IndexInput metaInput, int maxDoc) throws IOException {
        final long numBytes = Integer.BYTES - (Integer.numberOfLeadingZeros(maxDoc) / Byte.SIZE);
        this.maxDoc = metaInput.readInt();
        this.numBytesPerValue = metaInput.readInt();

        final long tmp = (numBytes * (maxDoc + 1));
        long blockSizes = (tmp / Long.BYTES) + ((tmp % Long.BYTES) > 0 ? 1 : 0);
        this.blocks = new long[Math.toIntExact(blockSizes)];
        metaInput.readLongs(blocks, 0, blocks.length);
    }

    public int skipTo(int doc) {
        return this.doc = doc;
    }

    public int getOrd() {
        long bitPos = doc * 24L;          // 3 bytes * 8 bits
        int word = (int) (bitPos >>> 6);  // / 64
        long shift = bitPos & 63;         // % 64

        long v = blocks[word] >>> shift;

        // straddles two longs
        if (shift > 40) {
            v |= blocks[word + 1] << (64 - shift);
        }

        return (int) (v & 0xFFFFFF);
    }
}
