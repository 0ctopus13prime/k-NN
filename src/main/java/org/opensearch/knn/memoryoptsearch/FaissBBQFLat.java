/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.Getter;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.codec.nativeindex.bbq.BBQReader;
import org.opensearch.knn.index.codec.nativeindex.bbq.Lucene102BinaryFlatVectorsScorer;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;

import java.io.IOException;

public class FaissBBQFLat extends FaissIndex {
    @Getter
    private FlatVectorsReader bbqFlatReader;
    @Getter
    private String fieldName;

    public FaissBBQFLat(final String indexType) {
        super(indexType);
    }

    @Override
    protected void doLoad(IndexInput input) throws IOException {
        // No-op
        System.out.println("FaissBBQFLat::doLoad is called");
    }

    @Override
    public VectorEncoding getVectorEncoding() {
        return VectorEncoding.FLOAT32;
    }

    @Override
    public FloatVectorValues getFloatValues(IndexInput indexInput) throws IOException {
        return bbqFlatReader.getFloatVectorValues(fieldName);
    }

    @Override
    public ByteVectorValues getByteValues(IndexInput indexInput) throws IOException {
        throw new UnsupportedOperationException();
    }

    public void tempDoLoadManually(final SegmentReadState bbqReadState, final String fieldName) throws IOException {
        bbqFlatReader = new BBQReader(bbqReadState, new Lucene102BinaryFlatVectorsScorer());
        this.fieldName = fieldName;
    }
}
