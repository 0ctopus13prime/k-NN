package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.store.IndexInput;
import org.opensearch.common.lucene.store.ByteArrayIndexInput;

import java.io.IOException;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

public class DocIdOrdSkipListIndexReader {
    private final IndexInput metaInput;
    private final boolean isDense;
    private final int numLevel;
    private final int numDocsForGrouping;
    private final int groupFactor;
    private NonLeafSkipList[] nonLeafLevelSkipList;
    private long level0StartOffset;
    private long skipListStartOffset;
    private final int maxDoc;
    private int currDoc;
    private Lucene101PForUtil pforUtil;
    private int[] ords;
    private Lucene101PostingDecodingUtil decodingUtil;
    private final byte[] skipListBytes;

    public DocIdOrdSkipListIndexReader(
        final IndexInput metaInput,
        boolean isDense,
        int numLevel,
        int numDocsForGrouping,
        int groupFactor,
        long[] offsets,
        int maxDoc,
        long skipListStartOffset
    ) throws IOException {
        this.metaInput = metaInput;
        this.isDense = isDense;
        this.numLevel = numLevel;
        this.numDocsForGrouping = numDocsForGrouping;
        this.groupFactor = groupFactor;
        this.maxDoc = maxDoc;
        this.pforUtil = new Lucene101PForUtil(new Lucene101ForUtil());
        this.ords = new int[numDocsForGrouping];
        this.decodingUtil = new Lucene101PostingDecodingUtil(metaInput);

        // Get the level-0 starting offset
        level0StartOffset = metaInput.getFilePointer();
        this.skipListStartOffset = skipListStartOffset;

        // Skip to the skip list
        metaInput.seek(skipListStartOffset);

        // Load non-leaf level
        nonLeafLevelSkipList = new NonLeafSkipList[numLevel];
        int skipListBytesLen = 0;
        for (int level = 1; level < numLevel; ++level) {
            final int skipListSize = Math.toIntExact(offsets[level + 1] - offsets[level]);
            skipListBytesLen += skipListSize;
        }
        skipListBytes = new byte[skipListBytesLen];

        for (int level = 1, offset = 0; level < numLevel; ++level) {
            final int skipListSize = Math.toIntExact(offsets[level + 1] - offsets[level]);
            metaInput.readBytes(skipListBytes, offset, skipListSize);
            if (level != 1) {
                nonLeafLevelSkipList[level] = new NonLeafSkipList(level, offset, skipListSize);
            } else {
                nonLeafLevelSkipList[level] = new Level1SkipList(offset, skipListSize);
            }
            offset += skipListSize;
        }
    }

    public int skipTo(int doc) {
        // All exhausted
        if (doc > maxDoc) {
            return currDoc = NO_MORE_DOCS;
        }
        // Fast track : No need to advance
        if (doc < nonLeafLevelSkipList[1].maxDocUpperBound) {
            return currDoc = doc;
        }

        for (int level = numLevel - 1; level >= 1; --level) {
            if (nonLeafLevelSkipList[level].skipTo(doc)) {
                if (level > 1) {
                    nonLeafLevelSkipList[level - 1].prepare(
                        nonLeafLevelSkipList[level].lowerLevelStartOffset,
                        nonLeafLevelSkipList[level].numSkippedDocs
                    );
                }
            }
        }

        // It's a dense case, we always has `doc`.
        return currDoc = doc;
    }

    public int getOrd() throws IOException {
        final Level1SkipList level1Skipper = (Level1SkipList) nonLeafLevelSkipList[1];
        return level1Skipper.findOrd();
    }

    private class NonLeafSkipList {
        public IndexInput skipInput;
        public long lowerLevelStartOffset;
        public int numSkippedDocs;
        public int maxDocUpperBound;
        protected final int level;
        protected final int numDocsToSkip;
        protected final int readUntil;
        protected boolean isLastBlock;

        private NonLeafSkipList(final int level, final int skipListOffset, final int skipListSize) {
            this.skipInput = new ByteArrayIndexInput("NonLeafSkipList", skipListBytes, skipListOffset, skipListSize);
            this.level = level;
            // D * G^(level) -> #docs to skip at this level
            // Ex: #docs_in_block=256, group_factor=4 then level-1 = 256 * 4 docs, level-2 = 256 * 4 * 4 docs
            this.numDocsToSkip = numDocsForGrouping * ((int) Math.pow(groupFactor, level));
            this.lowerLevelStartOffset = 0;
            this.numSkippedDocs = 0;
            this.readUntil = skipListSize;
            this.maxDocUpperBound = 0;
            this.isLastBlock = false;
        }

        public boolean skipTo(int doc) {
            if (maxDocUpperBound > 0 && doc < maxDocUpperBound) {
                return false;
            }

            int readCount = 0;
            while (skipInput.getFilePointer() < readUntil) {
                try {
                    lowerLevelStartOffset = skipInput.readVLong();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                ++readCount;
                maxDocUpperBound += numDocsToSkip;
                numSkippedDocs = maxDocUpperBound - numDocsToSkip;
                isLastBlock = maxDocUpperBound > maxDoc;
                if (doc < maxDocUpperBound) {
                    break;
                }
            }

            return readCount > 0;
        }

        public void prepare(final long lowerLevelStartOffset, final int skippedDoc) {
            try {
                skipInput.seek(lowerLevelStartOffset);
                this.numSkippedDocs = skippedDoc;
                this.maxDocUpperBound = skippedDoc;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    private class Level1SkipList extends NonLeafSkipList {
        private final int maxJumpTableSize = groupFactor - 1;
        private boolean jumpTableLoaded = false;
        private int[] jumpTable;
        private int jumpTableSize;
        private int leafBlockIndexLoaded;

        private Level1SkipList(final int skipListOffset, final int skipListSize) {
            super(1, skipListOffset, skipListSize);
            jumpTable = new int[groupFactor];
            jumpTableSize = 0;
        }

        @Override
        public boolean skipTo(int doc) {
            if (maxDocUpperBound > 0 && doc < maxDocUpperBound) {
                return false;
            }

            while (skipInput.getFilePointer() < readUntil) {
                // Load starting offset
                try {
                    lowerLevelStartOffset = skipInput.readVLong();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }

                // Doc range update
                maxDocUpperBound += numDocsToSkip;
                numSkippedDocs = maxDocUpperBound - numDocsToSkip;
                isLastBlock = maxDocUpperBound > maxDoc;

                // Jump table is not loaded ye
                jumpTableLoaded = false;
                leafBlockIndexLoaded = -1;

                // Did we find the block?
                if (doc < maxDocUpperBound) {
                    return true;
                }

                // Skip a jump table. We don't skip for the last block.
                if (isLastBlock == false) {
                    try {
                        skipInput.seek(skipInput.getFilePointer() + Short.BYTES * maxJumpTableSize);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                }
            }

            return false;
        }

        public int findOrd() throws IOException {
            // Load jump table
            loadJumpTable();

            // Which leaf block?
            final int leafBlockIndex = getLeafBlockIndex();

            // If we can find ord within the loaded ords, then we can save one re-load.
            if (leafBlockIndex != leafBlockIndexLoaded) {
                // Get the actual leaf block offset
                final long leafBlockOffset;
                if (leafBlockIndex == 0) {
                    leafBlockOffset = level0StartOffset + lowerLevelStartOffset;
                } else {
                    leafBlockOffset = level0StartOffset + lowerLevelStartOffset + jumpTable[leafBlockIndex - 1];
                }

                // Seek to the leaf block
                metaInput.seek(leafBlockOffset);
                if (isLastBlock == false) {
                    // SIMD 256 decoding ords
                    pforUtil.decode(decodingUtil, ords);
                } else {
                    final int vIntEncodedDocs = ((maxDoc + 1) % numDocsForGrouping);

                    if (leafBlockIndex != jumpTableSize) {
                        // SIMD 256 decoding ords
                        pforUtil.decode(decodingUtil, ords);
                    } else {
                        if (vIntEncodedDocs <= 0) {
                            // SIMD 256 decoding ords
                            pforUtil.decode(decodingUtil, ords);
                        } else {
                            // Remaining ords are vInt encoded.
                            for (int i = 0; i < vIntEncodedDocs; ++i) {
                                ords[i] = metaInput.readVInt();
                            }
                        }
                    }
                }

                leafBlockIndexLoaded = leafBlockIndex;
            }

            return ords[currDoc - (numSkippedDocs + numDocsForGrouping * leafBlockIndex)];
        }

        private void loadJumpTable() {
            if (jumpTableLoaded) {
                return;
            }

            try {
                if (skipInput.getFilePointer() == readUntil) {
                    // Edge case: There's only one single remaining block.
                    // In this case, we don't have a jump table.
                    // For example, #docs = 1074 = 256 * 4 + 50. (#docs_in_group = 256, group_factor = 4)
                    // Then, there will be two level-1 blocks, and the last one will not have the jump table.
                    // Why? Since we don't record the starting offset of the first block in the group,
                    // we've already known it as `lowerLevelStartOffset`.
                    jumpTableSize = 0;
                    jumpTableLoaded = true;
                    return;
                }

                // We have more than one block in the group
                final int firstBlockSizeAndNumSubBlocks = skipInput.readShort();

                // It's valid only when isLastBlock == true
                final int jumpTableSizeInLastBlock = firstBlockSizeAndNumSubBlocks & 0b1111;
                jumpTable[0] = firstBlockSizeAndNumSubBlocks >> 4;

                if (isLastBlock == false) {
                    jumpTableSize = groupFactor - 1;
                } else {
                    jumpTableSize = jumpTableSizeInLastBlock;
                }

                for (int i = 1; i < jumpTableSize; ++i) {
                    jumpTable[i] = skipInput.readShort();
                }

                jumpTableLoaded = true;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        private int getLeafBlockIndex() {
            return (currDoc - numSkippedDocs) / numDocsForGrouping;
        }
    }
}
