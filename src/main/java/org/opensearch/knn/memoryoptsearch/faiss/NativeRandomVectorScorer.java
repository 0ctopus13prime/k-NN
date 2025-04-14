/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.IOException;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

public class NativeRandomVectorScorer implements RandomVectorScorer {
    private final int maxOrd;
    private final MemorySegment query;
    private final long flatVectorsManagerAddress;
    private final int dimension;

    public NativeRandomVectorScorer(
        final int maxOrd,
        final long flatVectorsManagerAddress,
        final int dimension,
        final MemorySegment query
    ) {
        this.maxOrd = maxOrd;
        this.flatVectorsManagerAddress = flatVectorsManagerAddress;
        this.dimension = dimension;
        this.query = query;
    }

    @Override
    public float score(int internalVectorId) throws IOException {
        // Entry point
        try {
            final float singleScore = (float) singleScoreMethodHandle.invoke(
                flatVectorsManagerAddress, query, internalVectorId, dimension);
            System.out.println("__________ single score=" + singleScore + ", vid=" + internalVectorId);
            return singleScore;
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public int maxOrd() {
        return maxOrd;
    }

    static final MethodHandle singleScoreMethodHandle;

    static {
        final Linker linker = Linker.nativeLinker();
        final SymbolLookup lookup = SymbolLookup.loaderLookup();

        singleScoreMethodHandle = linker.downcallHandle(
            lookup.find("Java_org_opensearch_knn_jni_FaissService_singleScoring").orElseThrow(),
            FunctionDescriptor.of(
                ValueLayout.JAVA_FLOAT,  // return float score
                ValueLayout.JAVA_LONG,  // manager address
                ValueLayout.ADDRESS,  // query
                ValueLayout.JAVA_INT, // vector id
                ValueLayout.JAVA_INT  // dimension
            )
        );
    }
}
