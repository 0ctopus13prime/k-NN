/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store.partial_loading;

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
        long __s = System.nanoTime();
        for (int level = maxLevel; level >= 1; --level) {
            // System.out.println(" +++++++++++++ next greedyUpdateNearest, level=" + level);
            greedyUpdateNearest(indexInput, distanceComputer, level, nearest);
        }
        long __e = System.nanoTime();
        long __tg = __e - __s;

        __s = System.nanoTime();
        final int ef = Math.max(parametersHNSW.efSearch, parametersHNSW.k);
        // System.out.println(" +++++++++++++++++ hnswSearch, ef="
        //     + ef + ", k=" + parametersHNSW.k + ", efSearch=" + parametersHNSW.efSearch
        //     + ", nearest.id=" + nearest.id
        //     + ", nearest.distance=" + nearest.distance);
        DistanceMaxHeap resultMaxHeap = new DistanceMaxHeap(parametersHNSW.k);
        DistanceMaxHeap candidates = new DistanceMaxHeap(ef);
        candidates.insertWithOverflow(nearest.id, nearest.distance);
        searchFromCandidates(indexInput, distanceComputer, resultMaxHeap, candidates, 0);
        __e = System.nanoTime();
        long __tsfc = __e - __s;
        System.out.println("Time for greedy: " + (__tg / 1000) + ", search_from_candidate: " + (__tsfc / 1000));
        return resultMaxHeap;
    }

    private static float addToMaxHeaps(
        int id, float distance, DistanceMaxHeap resultMaxHeap, DistanceMaxHeap candidates
    ) {
        resultMaxHeap.insertWithOverflow(id, distance);
        // System.out.println("++++++++++ Adding to candidate | id=" + id + ", dis=" + distance
        //                    + ", heap_size=" + candidates.size()
        //                    + ", top=" + (candidates.isEmpty() ? -1f : candidates.top().distance));
        candidates.insertWithOverflow(id, distance);
        return resultMaxHeap.top().distance;
    }

    private void searchFromCandidates(
        IndexInput indexInput,
        DistanceComputer distanceComputer,
        DistanceMaxHeap resultMaxHeap,
        DistanceMaxHeap candidates,
        int level
    ) throws IOException {
        // final Set<Integer> idSet = new HashSet<>(128, 0.65f);
        final SparseFixedBitSet bitSet = new SparseFixedBitSet(1000100);
        for (final IdAndDistance candidate : candidates) {
            resultMaxHeap.insertWithOverflow(candidate.id, candidate.distance);
            bitSet.set(candidate.id);
            // idSet.add(candidate.id);
        }

        // System.out.println(" ++++++++++++++++++ searchFromCandidates, threshold="
        //      + resultMaxHeap.top().distance + ", resultSize=" + resultMaxHeap.size());

        int[] neighbor4Ids = new int[4];
        float[] distances4 = new float[4];
        final IdAndDistance currMin = new IdAndDistance(0, 0);

        while (!candidates.isEmpty()) {
            candidates.popMin(currMin);

            long begin, end;
            final long o = offsets.readLong(indexInput, currMin.id);
            begin = o + cumNumberNeighborPerLevel[level];
            end = o + cumNumberNeighborPerLevel[level + 1];

            // System.out.println(" ++++++++++++++++++++++ currMin.id="
            //    + currMin.id + ", currMin.dis=" + currMin.distance
            //     + ", begin=" + begin + ", end=" + end
            //     + ", threshold=" + (resultMaxHeap.isEmpty() ? Float.MAX_VALUE : resultMaxHeap.top().distance)
            //     + ", currMax.id=" + (resultMaxHeap.isEmpty() ? -1: resultMaxHeap.top().id)
            //     + ", len(candidate)=" + candidates.size());

            int numBuffered = 0;

            for (long offset = begin ; offset < end ; offset++) {
                final int neighborId = neighbors.readInt(indexInput, offset);
                if (neighborId >= 0) {
                    if (!bitSet.getAndSet(neighborId)) {
                        neighbor4Ids[numBuffered++] = neighborId;

                        if (numBuffered == 4) {
                            distanceComputer.computeBatch4(indexInput, neighbor4Ids, distances4);

                            addToMaxHeaps(neighbor4Ids[0], distances4[0], resultMaxHeap, candidates);
                            addToMaxHeaps(neighbor4Ids[1], distances4[1], resultMaxHeap, candidates);
                            addToMaxHeaps(neighbor4Ids[2], distances4[2], resultMaxHeap, candidates);
                            addToMaxHeaps(neighbor4Ids[3], distances4[3], resultMaxHeap, candidates);

                            numBuffered = 0;
                        }  // End if
                    }
                    continue;
                }
                break;
            }

            for (int i = 0; i < numBuffered; ++i) {
                final float distance = distanceComputer.compute(indexInput, neighbor4Ids[i]);

                addToMaxHeaps(neighbor4Ids[i], distance, resultMaxHeap, candidates);
            }

            // System.out.println(" ++++++++++++++++++++ len(neighborIds)=" + neighborIds.size());

//            for (int neighborId : neighborIds) {
//                if (!bitSet.getAndSet(neighborId)) {
//                    neighbor4Ids[numBuffered++] = neighborId;
//
//                    if (numBuffered == 4) {
//                        distanceComputer.computeBatch4(indexInput, neighbor4Ids, distances4);
//
//                        addToMaxHeaps(neighbor4Ids[0], distances4[0], resultMaxHeap, candidates);
//                        addToMaxHeaps(neighbor4Ids[1], distances4[1], resultMaxHeap, candidates);
//                        addToMaxHeaps(neighbor4Ids[2], distances4[2], resultMaxHeap, candidates);
//                        addToMaxHeaps(neighbor4Ids[3], distances4[3], resultMaxHeap, candidates);
//
//                        numBuffered = 0;
//                    }  // End if
//                }
//                if (!idSet.contains(neighborId)) {
//                    idSet.add(neighborId);
//                    neighbor4Ids[numBuffered++] = neighborId;
//
//                    if (numBuffered == 4) {
//                        distanceComputer.computeBatch4(indexInput, neighbor4Ids, distances4);
//
//                        addToMaxHeaps(neighbor4Ids[0], distances4[0], resultMaxHeap, candidates);
//                        addToMaxHeaps(neighbor4Ids[1], distances4[1], resultMaxHeap, candidates);
//                        addToMaxHeaps(neighbor4Ids[2], distances4[2], resultMaxHeap, candidates);
//                        addToMaxHeaps(neighbor4Ids[3], distances4[3], resultMaxHeap, candidates);
//
//                        numBuffered = 0;
//                    }  // End if
//                }
//            }  // End for
        }  // End while
    }

    private void greedyUpdateNearest(IndexInput indexInput,
        DistanceComputer distanceComputer,
        int level,
        IdAndDistance nearest) throws IOException {
        int[] bufferedIds = new int[4];
        float[] distances = new float[4];
        final Set<Integer> idSet = new HashSet<>(128, 0.65f);

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

//            for (long j = begin; j < end; j++) {
//                final int neighborId = neighbors.readInt(indexInput, j);
//                if (neighborId < 0) {
//                    break;
//                }
//                final float distance = distanceComputer.compute(indexInput, neighborId);
//                if (distance < nearest.distance) {
//                    nearest.id = neighborId;
//                    nearest.distance = distance;
//                }
//            }  // End for

            for (long j = begin; j < end; j++) {
                final int neighborId = neighbors.readInt(indexInput, j);
                if (neighborId < 0) {
                    break;
                }
                if (idSet.contains(neighborId)) {
                    continue;
                }
                idSet.add(neighborId);
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

/*
  TODO : Read bulk neighbor list.
  Scoring function ... -> JNI?
  where does extra 1-2 ms come from..???
   e.g. graph searching -> 2-3ms, result -> 5ms
 */
