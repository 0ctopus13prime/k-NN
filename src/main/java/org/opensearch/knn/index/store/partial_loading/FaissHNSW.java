/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store.partial_loading;

import lombok.AllArgsConstructor;
import lombok.ToString;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

public class FaissHNSW {
    public double[] assignProbas;
    public int[] cumNumberNeighborPerLevel;
    public IntStorage levels = new IntStorage();
    public LongStorage offsets = new LongStorage();
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
        // System.out.println(" ++++++++++++++++++= hnswSearch: entryPoint=" + entryPoint + ", dis=" + nearest.distance);
        for (int level = maxLevel; level >= 1; --level) {
            // System.out.println(" +++++++++++++ next greedyUpdateNearest, level=" + level);
            greedyUpdateNearest(indexInput, distanceComputer, level, nearest);
        }

        final int ef = Math.max(parametersHNSW.efSearch, parametersHNSW.k);
        // // System.out.println(" +++++++++++++++++ hnswSearch, ef="
        // //     + ef + ", k=" + parametersHNSW.k + ", efSearch=" + parametersHNSW.efSearch
        // //     + ", nearest.id=" + nearest.id
        // //     + ", nearest.distance=" + nearest.distance);
        DistanceMaxHeap resultMaxHeap = new DistanceMaxHeap(parametersHNSW.k);
        DistanceMaxHeap candidates = new DistanceMaxHeap(ef);
        candidates.insertWithOverflow(nearest.id, nearest.distance);
        searchFromCandidates(indexInput, distanceComputer, resultMaxHeap, candidates, 0);
        return resultMaxHeap;
    }

    private static float addToMaxHeaps(
        int id, float distance, float threshold, DistanceMaxHeap resultMaxHeap, DistanceMaxHeap candidates
    ) {
        if (distance < threshold) {
            resultMaxHeap.insertWithOverflow(id, distance);
        }
        candidates.insertWithOverflow(id, distance);
        return resultMaxHeap.isEmpty() ? Float.MAX_VALUE : resultMaxHeap.top().distance;
    }

    private void searchFromCandidates(
        IndexInput indexInput, DistanceComputer distanceComputer, DistanceMaxHeap resultMaxHeap, DistanceMaxHeap candidates, int level
    ) throws IOException {
        final Set<Integer> idSet = new HashSet<>();
        float threshold = resultMaxHeap.isEmpty() ? Float.MAX_VALUE : resultMaxHeap.top().distance;
        for (final IdAndDistance candidate : candidates) {
            if (candidate.distance < threshold) {
                resultMaxHeap.insertWithOverflow(candidate.id, candidate.distance);
                threshold = resultMaxHeap.top().distance;
            }
            idSet.add(candidate.id);
        }

        // System.out.println(" ++++++++++++++++++ searchFromCandidates, threshold=" + threshold + ", resultSize=" + resultMaxHeap.size());

        int[] savedNeighborIds = new int[4];
        float[] distances = new float[4];
        final IdAndDistance currMin = new IdAndDistance(0, 0);

        while (!candidates.isEmpty()) {
            candidates.popMin(currMin);

            long begin, end;
            final long o = offsets.readLong(indexInput, currMin.id);
            begin = o + cumNumberNeighborPerLevel[level];
            end = o + cumNumberNeighborPerLevel[level + 1];

            // System.out.println(" ++++++++++++++++++++++ currMin.id="
            //    + currMin.id + ", currMin.dis=" + currMin.distance + ", begin=" + begin + ", end=" + end);

            long maxNeighborIdx = begin;
            while (maxNeighborIdx < end) {
                final int neighborId = neighbors.readInt(indexInput, maxNeighborIdx);
                if (neighborId < 0) {
                    break;
                }
                ++maxNeighborIdx;
            }

            // System.out.println(" ++++++++++++++++++++ maxNeighborIdx=" + maxNeighborIdx);

            int count = 0;

            for (long neighborIdx = begin; neighborIdx < maxNeighborIdx; ++neighborIdx) {
                final int neighborId = neighbors.readInt(indexInput, neighborIdx);
                if (!idSet.contains(neighborId)) {
                    idSet.add(neighborId);
                    savedNeighborIds[count] = neighborId;
                    ++count;
                }

                if (count == 4) {
                    distanceComputer.computeBatch4(indexInput, savedNeighborIds, distances);

                    threshold = addToMaxHeaps(savedNeighborIds[0], distances[0], threshold, resultMaxHeap, candidates);
                    threshold = addToMaxHeaps(savedNeighborIds[1], distances[1], threshold, resultMaxHeap, candidates);
                    threshold = addToMaxHeaps(savedNeighborIds[2], distances[2], threshold, resultMaxHeap, candidates);
                    threshold = addToMaxHeaps(savedNeighborIds[3], distances[3], threshold, resultMaxHeap, candidates);

                    count = 0;
                }  // End if
            }  // End for

            for (int i = 0; i < count; ++i) {
                final float distance = distanceComputer.compute(indexInput, savedNeighborIds[i]);
                threshold = addToMaxHeaps(savedNeighborIds[i], distance, threshold, resultMaxHeap, candidates);
            }
        }  // End while
    }

    private void greedyUpdateNearest(IndexInput indexInput, DistanceComputer distanceComputer, int level, IdAndDistance nearest)
        throws IOException {
        int[] bufferedIds = new int[4];
        float[] distances = new float[4];

        while (true) {
            final int prevNearest = nearest.id;

            // Neighbor range
            long begin, end;
            final long o = offsets.readLong(indexInput, nearest.id);
            begin = o + cumNumberNeighborPerLevel[level];
            end = o + cumNumberNeighborPerLevel[level + 1];

            // System.out.println(" +++++++++++++++++++++++++ greedyUpdateNearest, begin="
            //    + begin + ", end=" + end + ", prevNearest=" + prevNearest);

            int nBuffered = 0;

            for (long j = begin; j < end; j++) {
                int neighborId = neighbors.readInt(indexInput, j);
                if (neighborId < 0) {
                    break;
                }
                bufferedIds[nBuffered++] = neighborId;

                if (nBuffered == 4) {
                    distanceComputer.computeBatch4(indexInput, bufferedIds, distances);

                    if (distances[0] < nearest.distance) {
                        nearest.id = bufferedIds[0];
                        nearest.distance = distances[0];
                    }
                    if (distances[1] < nearest.distance) {
                        nearest.id = bufferedIds[1];
                        nearest.distance = distances[1];
                    }
                    if (distances[2] < nearest.distance) {
                        nearest.id = bufferedIds[2];
                        nearest.distance = distances[2];
                    }
                    if (distances[3] < nearest.distance) {
                        nearest.id = bufferedIds[3];
                        nearest.distance = distances[3];
                    }
                    nBuffered = 0;
                }
            }  // End for

            // Process leftovers
            for (int i = 0; i < nBuffered; ++i) {
                final float distance = distanceComputer.compute(indexInput, bufferedIds[i]);
                if (distance < nearest.distance) {
                    nearest.id = bufferedIds[i];
                    nearest.distance = distance;
                }
            }

            if (nearest.id == prevNearest) {
                return;
            }
        }  // End while
    }
}
