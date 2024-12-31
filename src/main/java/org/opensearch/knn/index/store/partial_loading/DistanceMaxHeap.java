/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store.partial_loading;

import java.util.Iterator;
import java.util.NoSuchElementException;

public class DistanceMaxHeap implements Iterable<FaissHNSW.IdAndDistance> {
    private int k;
    private int numValidElems;
    private final int maxK;
    private final FaissHNSW.IdAndDistance[] heap;

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
        this.maxK = heapSize - 1;

        for (int i = 1; i < heapSize; i++) {
            heap[i] = new FaissHNSW.IdAndDistance(0, Float.MAX_VALUE);
        }

        this.k = 0;
        this.numValidElems = 0;
    }

    private void add(int id, float distance) {
        // don't modify size until we know heap access didn't throw AIOOB.
        final int index = k + 1;
        heap[index].id = id;
        heap[index].distance = distance;
        k = index;
        upHeap(index);
    }

    private int findMinimumIndex() {
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
        final int minIdx = findMinimumIndex();
        minIad.id = heap[minIdx].id;
        minIad.distance = heap[minIdx].distance;
        heap[minIdx].id = -1;
        --numValidElems;
    }

    public void insertWithOverflow(int id, float distance) {
        if (k == maxK) {
            if (distance >= heap[1].distance) {
                return;
            }
            if (heap[1].id == -1) {
                ++numValidElems;
            }

            heap[1].id = id;
            heap[1].distance = distance;
            updateTop();
        } else {
            add(id, distance);
            ++numValidElems;
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

    public boolean isFull() {
        return k == maxK;
    }

    private void upHeap(int origPos) {
        int i = origPos;
        FaissHNSW.IdAndDistance node = heap[i]; // save bottom node
        int j = i >>> 1;
        while (j > 0 && greaterThan(node.distance, heap[j].distance)) {
            heap[i] = heap[j]; // shift parents down
            i = j;
            j = j >>> 1;
        }
        heap[i] = node; // install saved node
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
