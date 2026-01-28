/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common.reorder;

import java.util.Arrays;

public class UnitHeap {
    public int[] update;
    public ListElement[] linkedList;
    private HeadEnd[] header;
    private int top;
    private int heapSize;

    public UnitHeap(int heapSize) {
        this.heapSize = heapSize;
        this.header = new HeadEnd[Math.max(4, heapSize >> 4)];
        this.linkedList = new ListElement[heapSize];
        this.update = new int[heapSize];

        // Create a linked list and connecting each other
        for (int i = 0; i < heapSize; ++i) {
            linkedList[i] = new ListElement();
            linkedList[i].prev = i - 1;
            linkedList[i].next = i + 1;
            linkedList[i].key = 0;
            update[i] = 0;
        }
        linkedList[heapSize - 1].next = -1;

        // Since all nodes have the same value initially, make header point to entire region.
        for (int i = 0; i < header.length; i++) {
            header[i] = new HeadEnd();
        }
        header[0].first = 0;
        header[0].second = heapSize - 1;
        top = 0;
    }

    public void deleteElement(int index) {
        final int prev = linkedList[index].prev;
        final int next = linkedList[index].next;
        final int key = linkedList[index].key;

        // Connect prev and next node element.
        if (prev >= 0) {
            linkedList[prev].next = next;
        }
        if (next >= 0) {
            linkedList[next].prev = prev;
        }

        // Adjust header table
        if (header[key].first == header[key].second) {
            header[key].first = header[key].second = -1;
        } else if (header[key].first == index) {
            header[key].first = next;
        } else if (header[key].second == index) {
            header[key].second = prev;
        }

        // Adjust top
        if (top == index) {
            top = linkedList[top].next;
        }

        // Invalidate prev, next pointer of `index` node.
        linkedList[index].prev = linkedList[index].next = -1;
    }

    public void reconstruct() {
        // Index numbers
        Integer[] tmp = new Integer[heapSize];
        for (int i = 0; i < heapSize; i++) {
            tmp[i] = i;
        }

        // Sort the index according to the key in descending order
        Arrays.sort(tmp, (a, b) -> Integer.compare(linkedList[b].key, linkedList[a].key));

        // The first element e.g. node with the max key
        int key = linkedList[tmp[0]].key;
        linkedList[tmp[0]].next = tmp[1];
        linkedList[tmp[0]].prev = -1;

        // The last element e.g. node with the min key
        linkedList[tmp[tmp.length - 1]].next = -1;
        linkedList[tmp[tmp.length - 1]].prev = tmp[tmp.length - 2];

        // Making a header
        header = new HeadEnd[Math.max(key + 1, header.length)];
        for (int i = 0 ; i < header.length ; ++i) {
            header[i] = new HeadEnd();
        }
        header[key].first = tmp[0];
        for (int i = 1; i < tmp.length - 1; i++) {
            // Make list element point to prev, next
            // X <-> Y <-> Z
            int prev = tmp[i - 1];
            int v = tmp[i];
            int next = tmp[i + 1];
            linkedList[v].prev = prev;
            linkedList[v].next = next;

            // Adjust the header table
            int tmpkey = linkedList[tmp[i]].key;
            if (tmpkey != key) {
                header[key].second = tmp[i - 1];
                header[tmpkey].first = tmp[i];
                key = tmpkey;
            }
        }

        // Adjust the header table for the last element.
        if (key == linkedList[tmp[tmp.length - 1]].key) header[key].second = tmp[tmp.length - 1];
        else {
            header[key].second = tmp[tmp.length - 2];
            int lastone = tmp[tmp.length - 1];
            int lastkey = linkedList[lastone].key;
            header[lastkey].first = header[lastkey].second = lastone;
        }

        // Make top
        top = tmp[0];
    }

    public void decreaseKey(int index) {
        update[index] -= 1;
    }

    public void increaseKey(int index) {
        int key = linkedList[index].key;
        final int prev = linkedList[index].prev;
        final int next = linkedList[index].next;
        final int firstVertexInCurrRegion = header[key].first;

        // So head.start and end are pointing to the nodes having the same `key` value, and we're trying to increase the value.
        // Therefore, we should take the node from the linked list,
        // and put it before the range to keep the linked list's sorting order.
        // Ex: | the region having `key` + 1 | | the current region |
        //                          --------^ we want to put the node in here
        if (firstVertexInCurrRegion != index) {
            // Make previous element point to the next one.
            linkedList[prev].next = next;
            if (next >= 0) {
                linkedList[next].prev = prev;
            }

            // Append the node before the range (e.g. key value == `key`)
            // Now the node is in a previous range (e.g. key value == `key` + 1)
            final int lastVertexInTargetRegion = linkedList[firstVertexInCurrRegion].prev;
            linkedList[index].prev = lastVertexInTargetRegion;
            linkedList[index].next = firstVertexInCurrRegion;
            linkedList[firstVertexInCurrRegion].prev = index;
            if (lastVertexInTargetRegion >= 0) {
                linkedList[lastVertexInTargetRegion].next = index;
            }
        }

        // Increase the key value
        linkedList[index].key++;

        // Adjust header table for `key`
        if (header[key].first == header[key].second) {
            header[key].first = header[key].second = -1;
        } else if (header[key].first == index) {
            header[key].first = next;
        } else if (header[key].second == index) {
            header[key].second = prev;
        }

        // Now adjust header table for `key` + 1
        ++key;
        header[key].second = index;  // We appended the node the next region, make `second` point to that.
        if (header[key].first < 0) {
            header[key].first = index;
        }

        // Update top
        if (linkedList[top].key < key) {
            top = index;
        }

        // Expand header if necessary
        if ((key + 4) >= header.length) {
            final int oldLength = header.length;
            header = java.util.Arrays.copyOf(header, (int) (header.length * 1.5));
            for (int i = oldLength ; i < header.length; i++) {
                header[i] = new HeadEnd();
            }
        }
    }

    public int extractMax() {
        int oldTop;
        do {
            oldTop = top;
            if (update[top] < 0) {
                decreaseTop();
            }
        } while (top != oldTop);

        deleteElement(oldTop);
        return oldTop;
    }

    private void decreaseTop() {
        final int oldTop = top;
        final int key = linkedList[oldTop].key;
        final int next = linkedList[oldTop].next;
        int minKeyGENewKey = key;
        final int newKey = key + update[oldTop] - (update[oldTop] / 2);

        if (newKey < linkedList[next].key) {
            // Find the region where the node belongs to after decreased key
            // Iterating region in desc order
            // Ex: ... | region key >= newKey | | region key < newKey | ...
            //      ----^ Goal is to find this region
            int firstVertexIdInRegion = linkedList[header[minKeyGENewKey].second].next;
            while (firstVertexIdInRegion >= 0 && linkedList[firstVertexIdInRegion].key >= newKey) {
                minKeyGENewKey = linkedList[firstVertexIdInRegion].key;
                firstVertexIdInRegion = linkedList[header[minKeyGENewKey].second].next;
            }

            // Make a new top
            linkedList[next].prev = -1;

            // Insert the node into the corresponding region
            // Note that `minKeyGENewKey` is the minimum number that >= newKey
            final int lastVertexInRegion = header[minKeyGENewKey].second;

            // The first node of the region that the node will be inserted into
            final int firstVertexInNextRegion = linkedList[lastVertexInRegion].next;

            // Append the node into the region
            // | previous region (key >= newKey) | | the region that it belongs |
            //                            ------^ appending the node in here
            linkedList[top].prev = lastVertexInRegion;
            linkedList[top].next = firstVertexInNextRegion;
            if (firstVertexInNextRegion >= 0) {
                linkedList[firstVertexInNextRegion].prev = oldTop;
            }

            // Make the last node in the next region point to the node
            linkedList[lastVertexInRegion].next = oldTop;

            // Update `top`
            top = next;

            // Adjust header table
            if (header[key].first == header[key].second) {
                header[key].first = header[key].second = -1;
            } else {
                header[key].first = next;
            }

            // Assign the new key
            linkedList[oldTop].key = newKey;

            // We only added half of `update`
            update[oldTop] /= 2;

            // Since we appended the node, we should make header table point it
            header[newKey].second = oldTop;
            if (header[newKey].first < 0) {
                header[newKey].first = oldTop;
            }
        }
    }

    private static class HeadEnd {
        public int first = -1;
        public int second = -1;
    }

    public static class ListElement {
        public int key;
        public int prev;
        public int next;
    }
}
