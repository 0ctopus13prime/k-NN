/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store.partial_loading;

import java.util.Iterator;
import java.util.NoSuchElementException;

public class DistanceMaxHeap implements Iterable<FaissHNSW.IdAndDistance> {
    private int k = 0;  // Pointing the next last leaf element.
    private int numValidElems = 0;
    private final int maxSize;
    private final FaissHNSW.IdAndDistance[] heap;
    private int[] invalidIndices;
    private int invalidIndicesUpto;

    public DistanceMaxHeap(int maxSize) {
        final int heapSize;

        if (maxSize == 0) {
            // We allocate 1 extra to avoid if statement in top()
            heapSize = 2;
        } else {
            // NOTE: we add +1 because all access to heap is
            // 1-based not 0-based.  heap[0] is unused.
            heapSize = maxSize + 1;
        }

        // T is an unbounded type, so this unchecked cast works always.
        this.heap = new FaissHNSW.IdAndDistance[heapSize];
        this.maxSize = maxSize;

        for (int i = 1; i < heapSize; i++) {
            heap[i] = new FaissHNSW.IdAndDistance(0, Float.MAX_VALUE);
        }

        invalidIndices = new int[2];
        invalidIndicesUpto = 0;
    }

    private FaissHNSW.IdAndDistance add(int id, float distance) {
        // don't modify size until we know heap access didn't throw AIOOB.
        int index = k + 1;
        heap[index].id = id;
        heap[index].distance = distance;
        k = index;
        upHeap(index);
        return heap[1];
    }

    private int findLastValidIndex() {
        float minDistance = Float.MAX_VALUE;
        int minIdx = -1;
        for (int i = k; i > 0; --i) {
            if (heap[i].id != -1 && heap[i].distance < minDistance) {
                minIdx = i;
                minDistance = heap[i].distance;
            }
        }

        return minIdx;
    }

    public final void popMin(FaissHNSW.IdAndDistance minIad) {
        final int minIdx = findLastValidIndex();
        if (invalidIndicesUpto >= invalidIndices.length) {
            int[] newInvalidIndices = new int[2 * invalidIndices.length];
            System.arraycopy(invalidIndices, 0, newInvalidIndices, 0, invalidIndices.length);
            invalidIndices = newInvalidIndices;
        }
        invalidIndices[invalidIndicesUpto++] = minIdx;
        minIad.id = heap[minIdx].id;
        minIad.distance = heap[minIdx].distance;
        // Mark it invalid.
        heap[minIdx].id = -1;
        heap[minIdx].distance = Float.MIN_VALUE;
        --numValidElems;
    }

    public void insertWithOverflow(int id, float distance) {
        if (numValidElems < maxSize) {
            if (invalidIndicesUpto <= 0) {
                add(id, distance);
            } else {
                // Find minimum invalid index.
                int minIdxIdx = 0;
                int minIdx = Integer.MAX_VALUE;
                for (int i = 0; i < invalidIndicesUpto; ++i) {
                    if (invalidIndices[i] < minIdx) {
                        minIdx = invalidIndices[i];
                        minIdxIdx = i;
                    }
                }
                if (minIdxIdx != (invalidIndicesUpto - 1)) {
                    for (int i = minIdxIdx + 1; i < invalidIndicesUpto; ++i) {
                        invalidIndices[i - 1] = invalidIndices[i];
                    }
                }
                --invalidIndicesUpto;

                // System.out.println("minIdx=" + minIdx + ", invalidIndicesUpto=" + invalidIndicesUpto);

                heap[minIdx].id = id;
                heap[minIdx].distance = distance;
                upHeap(minIdx);
            }
            ++numValidElems;
        } else if (greaterThan(heap[1].distance, distance)) {
            heap[1].id = id;
            heap[1].distance = distance;
            updateTop();
        }
    }

    private static boolean greaterThan(float a, float b) {
        return a > b;
    }

    public final FaissHNSW.IdAndDistance top() {
        return heap[1];
    }

    public final FaissHNSW.IdAndDistance pop() {
        if (k > 0) {
            FaissHNSW.IdAndDistance result = heap[1]; // save first value
            heap[1] = heap[k]; // move last to first
            k--;
            downHeap(1); // adjust heap
            --numValidElems;
            return result;
        } else {
            return null;
        }
    }

    private FaissHNSW.IdAndDistance updateTop() {
        downHeap(1);
        return heap[1];
    }

    public final int size() {
        return numValidElems;
    }

    public boolean isEmpty() {
        return numValidElems <= 0;
    }

    private boolean upHeap(int origPos) {
        int i = origPos;
        FaissHNSW.IdAndDistance node = heap[i]; // save bottom node
        int j = i >>> 1;
        while (j > 0 && greaterThan(node.distance, heap[j].distance)) {
            heap[i] = heap[j]; // shift parents down
            i = j;
            j = j >>> 1;
        }
        heap[i] = node; // install saved node
        return i != origPos;
    }

    private void downHeap(int i) {
        FaissHNSW.IdAndDistance node = heap[i]; // save top node
        int j = i << 1; // find smaller child
        int k = (i << 1) + 1;
        if (k <= this.k && greaterThan(heap[k].distance, heap[j].distance)) {
            j = k;
        }
        while (j <= this.k && greaterThan(heap[j].distance, node.distance)) {
            heap[i] = heap[j]; // shift up child
            i = j;
            j = i << 1;
            k = j + 1;
            if (k <= this.k && greaterThan(heap[k].distance, heap[j].distance)) {
                j = k;
            }
        }
        heap[i] = node; // install saved node
    }

    public FaissHNSW.IdAndDistance[] getHeapArray() {
        return heap;
    }

    @Override public Iterator<FaissHNSW.IdAndDistance> iterator() {
        return new Iterator<>() {
            int i = 1;

            @Override public boolean hasNext() {
                return i <= k;
            }

            @Override public FaissHNSW.IdAndDistance next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                return heap[i++];
            }
        };
    }
}
