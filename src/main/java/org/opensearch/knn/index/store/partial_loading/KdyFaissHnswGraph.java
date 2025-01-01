/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store.partial_loading;

import java.io.IOException;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.hnsw.HnswGraph;

public class KdyFaissHnswGraph extends HnswGraph {
    public FaissHNSW hnsw;
    private long begin;
    private long end;
    private long currOffset;
    private IndexInput indexInput;

    public KdyFaissHnswGraph(KdyHNSW kdyHNSW, IndexInput indexInput) {
        this.hnsw = kdyHNSW.hnswFlatIndex.hnsw;
        this.indexInput = indexInput;
    }

    @Override public void seek(int level, int target) throws IOException {
        long o = hnsw.offsets[target];
        begin = o + hnsw.cumNumberNeighborPerLevel[level];
        end = o + hnsw.cumNumberNeighborPerLevel[level + 1];
        currOffset = begin;
    }

    @Override public int size() {
        return 1000_0000;
    }

    @Override public int nextNeighbor() throws IOException {
        if (currOffset < end) {
            final int id = hnsw.neighbors.readInt(indexInput, currOffset);
            ++currOffset;
            if (id >= 0) {
                return id;
            }
        }

        return Integer.MAX_VALUE;
    }

    @Override public int numLevels() throws IOException {
        return hnsw.maxLevel;
    }

    @Override public int entryNode() throws IOException {
        return hnsw.entryPoint;
    }

    @Override public HnswGraph.NodesIterator getNodesOnLevel(int level) throws IOException {
        throw new UnsupportedOperationException();
    }
}
