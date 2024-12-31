/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store.partial_loading;

import org.apache.lucene.util.hnsw.NeighborQueue;
import lombok.AllArgsConstructor;
import lombok.ToString;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.SparseFixedBitSet;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class FaissHNSW {
    public double[] assignProbas;
    public int[] cumNumberNeighborPerLevel;
    public IntStorage levels = new IntStorage();
    // public LongStorage offsets = new LongStorage();
    public long[] offsets = null;
    public IntStorage neighbors = new IntStorage();
    public int entryPoint;
    public int maxLevel = -1;
    public int efConstruction = 40;
    public int efSearch = 16;

    public FaissHNSW() {
        this(32);
    }

    public FaissHNSW(int M) {
        // ??? set_default_probas(M, 1.0 / log(M));
    }

    @AllArgsConstructor @ToString public static class IdAndDistance {
        public int id;
        public float distance;
    }

    public DistanceMaxHeap hnswSearch(
        IndexInput indexInput, SearchParametersHNSW parametersHNSW, DistanceComputer distanceComputer
    ) throws IOException {
        IdAndDistance nearest = new IdAndDistance(entryPoint, distanceComputer.compute(indexInput, entryPoint));
        final SparseFixedBitSet bitSet = new SparseFixedBitSet(1000100);
        for (int level = maxLevel; level >= 1; --level) {
            greedyUpdateNearest(indexInput, distanceComputer, level, nearest);
        }
        final int ef = Math.max(parametersHNSW.efSearch, parametersHNSW.k);
        DistanceMaxHeap resultMaxHeap = new DistanceMaxHeap(parametersHNSW.k);
        DistanceMaxHeap candidates = new DistanceMaxHeap(ef);
        candidates.insertWithOverflow(nearest.id, nearest.distance);
        searchFromCandidatesFaiss(indexInput, distanceComputer, resultMaxHeap, candidates, bitSet);
        return resultMaxHeap;
    }

    private static float addToMaxHeaps(
        int id, float distance, DistanceMaxHeap resultMaxHeap, DistanceMaxHeap candidates
    ) {
        resultMaxHeap.insertWithOverflow(id, distance);
        candidates.insertWithOverflow(id, distance);
        return resultMaxHeap.top().distance;
    }

    private static float getMaxAcceptableDistance(DistanceMaxHeap resultMaxHeap) {
        if (resultMaxHeap.isFull()) {
            return resultMaxHeap.top().distance;
        }
        return Float.MAX_VALUE;
    }

    private void searchFromCandidatesFaiss(
        IndexInput indexInput,
        DistanceComputer distanceComputer,
        DistanceMaxHeap resultMaxHeap,
        DistanceMaxHeap candidates,
        SparseFixedBitSet visited
    ) throws IOException {
        if (candidates.top().distance < getMaxAcceptableDistance(resultMaxHeap)) {
            resultMaxHeap.insertWithOverflow(candidates.top().id, candidates.top().distance);
        }
        visited.set(candidates.top().id);

        IdAndDistance minIad = new IdAndDistance(0, 0);
        while (!candidates.isEmpty()) {
            candidates.popMin(minIad);
            final long o = offsets[minIad.id];
            final long begin = o + cumNumberNeighborPerLevel[0];
            final long end = o + cumNumberNeighborPerLevel[1];

            for (long offset = begin ; offset < end ; offset++) {
                final int neighborId = neighbors.readInt(indexInput, offset);
                if (neighborId < 0) {
                    break;
                }
                if (visited.getAndSet(neighborId)) {
                    continue;
                }
                final float dist = distanceComputer.compute(indexInput, neighborId);
                if (dist < getMaxAcceptableDistance(resultMaxHeap)) {
                    resultMaxHeap.insertWithOverflow(neighborId, dist);
                }
                candidates.insertWithOverflow(neighborId, dist);
            }
        }
    }

    private void greedyUpdateNearest(
        IndexInput indexInput,
        DistanceComputer distanceComputer,
        int level,
        IdAndDistance nearest) throws IOException {

        while (true) {
            final int prevNearest = nearest.id;

            // Neighbor range

            final long o = offsets[nearest.id];
            final long begin = o + cumNumberNeighborPerLevel[level];
            final long end = o + cumNumberNeighborPerLevel[level + 1];

            // System.out.println(" +++++++++++++++++++++++++ greedyUpdateNearest, begin="
            //    + begin + ", end=" + end + ", prevNearest=" + prevNearest);

            for (long j = begin; j < end; j++) {
                final int neighborId = neighbors.readInt(indexInput, j);
                if (neighborId < 0) {
                    break;
                }
                final float distance = distanceComputer.compute(indexInput, neighborId);
                if (distance < nearest.distance) {
                    nearest.id = neighborId;
                    nearest.distance = distance;
                }
            }  // End for

            // Process leftovers

            if (nearest.id == prevNearest) {
                return;
            }
        }  // End while
    }
}
