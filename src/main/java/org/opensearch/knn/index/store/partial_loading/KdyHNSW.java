/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store.partial_loading;

import org.apache.lucene.store.IndexInput;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

public class KdyHNSW {
    private static final byte[] IXMP = new byte[] { 'I', 'x', 'M', 'p' };
    private static final byte[] IHNF = new byte[] { 'I', 'H', 'N', 'f' };
    private static final byte[] IXF2 = new byte[] { 'I', 'x', 'F', '2' };

    public FaissIdMapIndex idMapIndex;
    public FaissHNSWFlatIndex hnswFlatIndex;
    public FaissIndexFlatL2 indexFlatL2;

    public KdyHNSW(final IndexInput readStream) {
        try {
            // IxMp
            System.out.println("IxMP 0");
            byte[] indexType = new byte[4];
            readStream.readBytes(indexType, 0, indexType.length);
            assertIndexType(indexType, IXMP);
            idMapIndex = new FaissIdMapIndex();
            readHeader(readStream, idMapIndex);
            System.out.println("IxMP 1");

            // IHNf
            System.out.println("IHNf 0");
            readStream.readBytes(indexType, 0, indexType.length);
            assertIndexType(indexType, IHNF);
            hnswFlatIndex = new FaissHNSWFlatIndex();
            idMapIndex.index = hnswFlatIndex;
            readHeader(readStream, hnswFlatIndex);
            readHNSW(readStream, hnswFlatIndex.hnsw);
            System.out.println("IHNf 1");

            // IXF2
            System.out.println("IXF2 0");
            readStream.readBytes(indexType, 0, indexType.length);
            assertIndexType(indexType, IXF2);
            indexFlatL2 = new FaissIndexFlatL2();
            hnswFlatIndex.index = indexFlatL2;
            readHeader(readStream, indexFlatL2);

            indexFlatL2.oneVectorByteSize = indexFlatL2.d * 4L;
            System.out.println("indexFlatL2.oneVectorByteSize = " + indexFlatL2.oneVectorByteSize);

            final long codeSize = readStream.readLong() * 4L;
            indexFlatL2.codes.baseOffset = readStream.getFilePointer();
            indexFlatL2.codes.regionSize = codeSize;
            System.out.println("indexFlatL2.codes.baseOffset = " + indexFlatL2.codes.baseOffset);
            System.out.println("indexFlatL2.codes.regionSize = " + indexFlatL2.codes.regionSize);
            if (indexFlatL2.codes.regionSize != (indexFlatL2.nTotal * indexFlatL2.oneVectorByteSize)) {
                throw new RuntimeException(
                    "[KDY] CCCCCCCCCCCCCC " + indexFlatL2.codes.regionSize + " != " + (indexFlatL2.nTotal * indexFlatL2.oneVectorByteSize));
            }
            readStream.seek(indexFlatL2.codes.baseOffset + indexFlatL2.codes.regionSize);
            System.out.println("IXF2 1");

            // IHNf - post processing
            System.out.println("IHNf 2");
            hnswFlatIndex.index = indexFlatL2;
            System.out.println("IHNf 3");

            // IxMp - post processing
            System.out.println("IxMP 2, " + readStream.getFilePointer());
            idMapIndex.idMap.loadBlock(readStream);
            System.out.println("idMapIndex.idMap.baseOffset = " + idMapIndex.idMap.baseOffset);
            System.out.println("idMapIndex.idMap.regionSize = " + idMapIndex.idMap.regionSize);
            System.out.println("IxMP 3");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static void readHNSW(IndexInput readStream, FaissHNSW hnsw) throws IOException {
        long size = readStream.readLong();
        hnsw.assignProbas = new double[(int) size];
        for (int i = 0; i < size; i++) {
            byte[] doubleBytes = new byte[8];
            readStream.readBytes(doubleBytes, 0, 8);
            hnsw.assignProbas[i] = ByteBuffer.wrap(doubleBytes).order(ByteOrder.LITTLE_ENDIAN).getDouble();
            System.out.println("hnsw.assignProbas[i]=" + hnsw.assignProbas[i] + ", " + Arrays.toString(doubleBytes));
        }
        System.out.println("len(hnsw.assignProbas) == " + hnsw.assignProbas.length + ", offset=" + readStream.getFilePointer());

        size = readStream.readLong();
        hnsw.cumNumberNeighborPerLevel = new int[(int) size];
        if (size > 0) {
            readStream.readInts(hnsw.cumNumberNeighborPerLevel, 0, (int) size);
            for (int i = 0; i < size; ++i) {
                System.out.println("hnsw.cumNumberNeighborPerLevel[i] = " + hnsw.cumNumberNeighborPerLevel[i]);
            }
        }
        System.out.println("len(hnsw.cumNumberNeighborPerLevel) == " + hnsw.cumNumberNeighborPerLevel.length);

        hnsw.levels.loadBlock(readStream);
        System.out.println("hnsw.levels.baseOffset = " + hnsw.levels.baseOffset);
        System.out.println("hnsw.levels.regionSize = " + hnsw.levels.regionSize);

        hnsw.offsets = new long[(int) readStream.readLong()];
        readStream.readLongs(hnsw.offsets, 0, hnsw.offsets.length);
        System.out.println("len(hnsw.offsets)= " + hnsw.offsets.length);

        hnsw.neighbors.loadBlock(readStream);
        System.out.println("hnsw.neighbors.baseOffset = " + hnsw.neighbors.baseOffset);
        System.out.println("hnsw.neighbors.regionSize = " + hnsw.neighbors.regionSize);

        hnsw.entryPoint = readStream.readInt();
        System.out.println("hnsw.entryPoint = " + hnsw.entryPoint);

        hnsw.maxLevel = readStream.readInt();
        System.out.println("hnsw.maxLevel = " + hnsw.maxLevel);

        hnsw.efConstruction = readStream.readInt();
        System.out.println("hnsw.efConstruction = " + hnsw.efConstruction);

        hnsw.efSearch = readStream.readInt();
        System.out.println("hnsw.efSearch = " + hnsw.efSearch);

        // dummy read
        readStream.readInt();
    }

    private static void assertIndexType(byte[] indexType, byte[] expected) {
        if (!Arrays.equals(indexType, expected)) {
            throw new RuntimeException("[KDY] NOOOOOOOOOOOOOOOOOOOOO! " + Arrays.toString(indexType) + " != " + Arrays.toString(expected));
        }
    }

    private static void readHeader(IndexInput readStream, FaissIndex index) throws IOException {
        index.d = readStream.readInt();
        System.out.println("index.d = " + index.d);
        index.nTotal = readStream.readLong();
        System.out.println("index.nTotal = " + index.nTotal);
        // 2 dummies
        readStream.readLong();
        readStream.readLong();
        index.isTrained = readStream.readByte() == 1;
        System.out.println("index.isTrained = " + index.isTrained);

        final int metricTypeIndex = readStream.readInt();
        System.out.println("metricTypeIndex = " + metricTypeIndex);
        if (metricTypeIndex != 1) {
            throw new RuntimeException("[KDY] MMMMMMMMMMMMMM metricTypeIndex=" + metricTypeIndex);
        }
        final FaissIndex.MetricType metricType = FaissIndex.MetricType.values()[metricTypeIndex];
        index.metricType = metricType;
    }
}
