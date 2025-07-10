/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

public class FaissEmptyFlatVectors extends FaissIndex {
    final public static String NULL = "null";

    public FaissEmptyFlatVectors() {
        super(NULL);
    }

    @Override
    protected void doLoad(IndexInput input) throws IOException {
        // Do nothing
    }

    @Override
    public VectorEncoding getVectorEncoding() {
        // TODO
        return null;
    }

    @Override
    public FloatVectorValues getFloatValues(IndexInput indexInput) throws IOException {
        // TODO
        return null;
    }

    @Override
    public ByteVectorValues getByteValues(IndexInput indexInput) throws IOException {
        // TODO
        return null;
    }
}
