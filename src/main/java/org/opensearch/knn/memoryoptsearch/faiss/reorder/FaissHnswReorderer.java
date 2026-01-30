/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHNSW;

import java.io.IOException;

public class FaissHnswReorderer {
    public static void transform(
        final FaissHNSW faissHNSW,
        final IndexInput indexInput,
        final IndexOutput indexOutput,
        final ReorderOrdMap reorderOrdMap
    ) throws IOException {
        final int totalNumberOfVectors = Math.toIntExact(faissHNSW.getTotalNumberOfVectors());

        // Copy assignProbas
        indexInput.seek(faissHNSW.getAssignProbas().getBaseOffset());
        for (long i = 0; i < faissHNSW.getAssignProbas().getSectionSize(); ++i) {
            indexOutput.writeByte(indexInput.readByte());
        }

        // Copy accumulated number of neighbors
        indexOutput.writeLong(faissHNSW.getCumNumberNeighborPerLevel().length);
        for (int numNeighbors : faissHNSW.getCumNumberNeighborPerLevel()) {
            indexOutput.writeInt(numNeighbors);
        }

        // Copy levels
        indexOutput.writeLong(faissHNSW.getLevels().getSectionSize());
        for (int ord = 0; ord < totalNumberOfVectors; ++ord) {
            // Ex
            // Original : [0, 1, 2]
            // New : [2, 0, 1]
            // Then `ord` = 0, `oldOrd` = 2
            final int oldOrd = reorderOrdMap.newOrd2Old[ord];

            // Get level of oldOrd. With the above example, it's getting `oldOrd`(==2)'s level
            indexInput.seek(faissHNSW.getLevels().getBaseOffset() + Integer.BYTES * oldOrd);
            indexOutput.writeInt(indexInput.readInt());
        }

        // Reorder offsets
        final DirectMonotonicReader offsetReader = faissHNSW.getOffsetsReader();
        long offsetSoFar = 0;
        for (int oldOrd : reorderOrdMap.newOrd2Old) {
            indexOutput.writeLong(offsetSoFar);
            if (oldOrd != (totalNumberOfVectors - 1)) {
                final int neighborsSize = Math.toIntExact(offsetReader.get(oldOrd + 1) - offsetReader.get(oldOrd));
                offsetSoFar += neighborsSize;
            }
        }

        // Reorder neighbors
        final int maxLevel = faissHNSW.getMaxLevel();
        for (int oldOrd : reorderOrdMap.newOrd2Old) {
            final long offset = offsetReader.get(oldOrd);
            for (int level = 0; level < maxLevel; ++level) {
                // Read neighbors
                final long begin = offset + faissHNSW.getCumNumberNeighborPerLevel()[level];
                final long end = offset + faissHNSW.getCumNumberNeighborPerLevel()[level + 1];
                indexInput.seek(faissHNSW.getNeighbors().getBaseOffset() + Integer.BYTES * begin);

                for (long i = begin; i < end; i++) {
                    final int neighborId = indexInput.readInt();
                    // The idea is that a vector does not always have a complete list of neighbor vectors.
                    // FAISS assigns a fixed size to the neighbor list and uses -1 to indicate missing entries.
                    // Therefore, we can safely stop once hit -1.
                    // For example, if the neighbor list size is 16 and a vector has only 8 neighbors, the list would appear as:
                    // [1, 4, 6, 8, 13, 17, 60, 88, -1, -1, ..., -1].
                    if (neighborId >= 0) {
                        // Convert old ord neighbors to new ords
                        indexOutput.writeInt(reorderOrdMap.oldOrd2New[neighborId]);
                    } else {
                        indexOutput.writeInt(neighborId);
                    }
                }
            }
        }

        // Entry point
        indexOutput.writeInt(faissHNSW.getEntryPoint());

        // Max level
        indexOutput.writeInt(faissHNSW.getMaxLevel());

        // EF construct parameter
        indexOutput.writeInt(faissHNSW.getEfConstruct());

        // EF search parameter
        indexOutput.writeInt(faissHNSW.getEfSearch());

        // Dummy field
        indexOutput.writeInt(0);
    }
}
