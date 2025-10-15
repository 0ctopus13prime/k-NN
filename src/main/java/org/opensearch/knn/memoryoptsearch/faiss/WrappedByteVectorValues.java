/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.RequiredArgsConstructor;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.KnnVectorValues;

/**
 * An abstract wrapper around a {@link ByteVectorValues} instance, providing a consistent
 * interface for layered or decorated vector value implementations.
 * <p>
 * This class is typically used when multiple wrappers are stacked around a base
 * {@link ByteVectorValues}, and utility methods are needed to access the innermost
 * (bottom-level) implementation.
 */
@RequiredArgsConstructor
public abstract class WrappedByteVectorValues extends ByteVectorValues {

    // The wrapped (nested) {@link ByteVectorValues} instance.
    protected final ByteVectorValues byteVectorValues;

    /**
     * Extracts the bottom-level {@link ByteVectorValues} from a possibly wrapped
     * {@link KnnVectorValues} instance.
     * <p>
     * If the provided {@code knnVectorValues} is not an instance of
     * {@link ByteVectorValues}, this method returns {@code null}. Otherwise, it unwraps
     * any nested {@link WrappedByteVectorValues} layers until it reaches the base
     * {@link ByteVectorValues}.
     *
     * @param knnVectorValues the {@link KnnVectorValues} to unwrap
     * @return the innermost {@link ByteVectorValues}, or {@code null} if not applicable
     */
    public static ByteVectorValues getBottomByteVectorValues(final KnnVectorValues knnVectorValues) {
        if (knnVectorValues instanceof ByteVectorValues byteVectorValues) {
            if (byteVectorValues instanceof WrappedByteVectorValues wrappedByteVectorValues) {
                return wrappedByteVectorValues.byteVectorValues;
            }
            return byteVectorValues;
        }

        return null;
    }
}
