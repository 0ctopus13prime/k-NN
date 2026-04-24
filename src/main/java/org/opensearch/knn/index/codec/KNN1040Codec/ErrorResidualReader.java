/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.Getter;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.StringHelper;
import org.opensearch.knn.index.codec.nativeindex.ResidualQuantizer;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Reads and provides access to the {@code .ver} (vector error residual) file for a single field.
 *
 * <p>Opened eagerly when {@code Faiss1040ScalarQuantizedKnnVectorsReader} is constructed.
 * Holds the {@link IndexInput} handle, parsed header fields, and the centroid vector from
 * the 1-bit SQ quantization.
 *
 * <h2>File format</h2>
 * <pre>
 * [Lucene CodecUtil index header]
 * [dimension]      4B int
 * [numVectors]     4B int
 * [bytesPerBlock]  4B int
 * [per-vector blocks × numVectors]
 *   [packed 4-bit residual]   packedResidualBytes
 *   [lowerInterval]           4B float (LE)
 *   [upperInterval]           4B float (LE)
 *   [additionalCorrection]    4B float (LE)
 *   [componentSum]            4B int (LE)
 * [Lucene CodecUtil footer]
 * </pre>
 *
 * <h2>Thread safety</h2>
 * The underlying {@link IndexInput} is not thread-safe. For concurrent search, each thread
 * must call {@link #cloneInput()} to obtain an independent clone that can seek and read
 * without interfering with other threads.
 *
 * @see ResidualQuantizer — writes the {@code .ver} file during index build
 */
public class ErrorResidualReader implements Closeable {

    /** Vector dimensionality, read from the .ver file header. */
    @Getter
    private final int dimension;

    /** Total number of vectors in this segment, read from the .ver file header. */
    @Getter
    private final int numVectors;

    /**
     * Total bytes per vector block (packed residual + 16B metadata).
     * Used to compute seek offsets: {@code dataStartOffset + ordinal * bytesPerBlock}.
     */
    @Getter
    private final int bytesPerBlock;

    /**
     * Bytes occupied by the packed 4-bit residual within each block: {@code (dimension + 1) / 2}.
     * The remaining 16 bytes in each block are per-vector metadata (lower, upper, correction, sum).
     */
    @Getter
    private final int packedResidualBytes;

    /**
     * File offset where the first per-vector block begins (after codec header + our 3 int fields).
     * All ordinal-based seeks are relative to this offset.
     */
    private final long dataStartOffset;

    /**
     * Centroid vector (mean of all vectors in the field), passed in from
     * {@code QuantizedByteVectorValues.getCentroid()} at construction time.
     * Used at search time to compute {@code q' = query - centroid}.
     */
    @Getter
    private final float[] centroid;

    /** The underlying file handle. Cloned per search thread via {@link #cloneInput()}. */
    private final IndexInput verInput;

    /**
     * Open and parse the {@code .ver} file for the given field.
     *
     * <p>Validates the Lucene codec header and footer for data integrity, then reads
     * the format-specific fields (dimension, numVectors, bytesPerBlock) and records the
     * data start offset for ordinal-based seeks.
     *
     * @param directory   segment directory containing the .ver file
     * @param segmentName segment name (e.g., "_0")
     * @param fieldName   the knn vector field name
     * @param centroid    centroid vector from QuantizedByteVectorValues
     * @param ioContext   IO context for opening the file (caller should provide random access hints)
     * @throws IOException if the file cannot be opened or the header/footer is invalid
     */
    public ErrorResidualReader(Directory directory, String segmentName, String fieldName, float[] centroid, IOContext ioContext)
        throws IOException {
        final String fileName = segmentName + "_" + fieldName + ".ver";
        this.verInput = directory.openInput(fileName, ioContext.withHints(FileTypeHint.DATA, FileDataHint.KNN_VECTORS, DataAccessHint.RANDOM));

        // Validate the codec header (magic, codec name, version) without checking segment ID.
        // checkHeader consumes: magic(4) + codec name length(2) + codec name bytes + version(4).
        // After this call, the file pointer is positioned right before the segment ID.
        CodecUtil.checkHeader(
            verInput,
            ResidualQuantizer.CODEC_NAME,
            ResidualQuantizer.VERSION_CURRENT,
            ResidualQuantizer.VERSION_CURRENT
        );
        // Skip segment ID (16 bytes) written by writeIndexHeader
        verInput.skipBytes(StringHelper.ID_LENGTH);
        // Skip suffix: 1 byte length + that many suffix bytes (handles any suffix length)
        int suffixLength = Byte.toUnsignedInt(verInput.readByte());
        verInput.skipBytes(suffixLength);

        // Read format-specific header fields
        this.dimension = verInput.readInt();
        this.numVectors = verInput.readInt();
        this.bytesPerBlock = verInput.readInt();
        this.packedResidualBytes = (dimension + 1) / 2;

        // Record where per-vector data begins
        this.dataStartOffset = verInput.getFilePointer();

        this.centroid = centroid;

        // Validate footer for data integrity (checksum)
        CodecUtil.checksumEntireFile(verInput);
    }

    /**
     * Clone the {@link IndexInput} for thread-safe concurrent reads.
     *
     * <p>Each search thread should call this method and use the returned clone for all
     * {@link #readBlock} calls. The clone has independent seek position.
     *
     * @return an independent clone of the underlying IndexInput
     */
    public IndexInput cloneInput() {
        return verInput.clone();
    }

    /**
     * Read the full per-vector block (packed residual + metadata) for the given ordinal.
     *
     * <p>Seeks to {@code dataStartOffset + ordinal * bytesPerBlock} and reads
     * {@code bytesPerBlock} bytes. The caller should use a cloned IndexInput
     * obtained from {@link #cloneInput()} for thread safety.
     *
     * <p>Block layout:
     * <pre>
     * [0 .. packedResidualBytes-1]            packed 4-bit residual nibbles
     * [packedResidualBytes .. +3]             lowerInterval (float, LE)
     * [packedResidualBytes+4 .. +7]           upperInterval (float, LE)
     * [packedResidualBytes+8 .. +11]          additionalCorrection (float, LE)
     * [packedResidualBytes+12 .. +15]         componentSum (int, LE)
     * </pre>
     *
     * @param clonedInput a cloned IndexInput (from {@link #cloneInput()})
     * @param ordinal     vector ordinal (dense case: ordinal == docId)
     * @return byte array of length {@code bytesPerBlock} containing the full block
     * @throws IOException if the read fails
     */
    public byte[] readBlock(IndexInput clonedInput, int ordinal) throws IOException {
        long offset = dataStartOffset + (long) ordinal * bytesPerBlock;
        clonedInput.seek(offset);
        byte[] block = new byte[bytesPerBlock];
        clonedInput.readBytes(block, 0, bytesPerBlock);
        return block;
    }

    public void prefetch(IndexInput input, int ordinal) {
        final long offset = dataStartOffset + (long) ordinal * bytesPerBlock;
        try {
            input.prefetch(offset, bytesPerBlock);
        } catch (IOException e) {
            // Prefetch is a hint — failure is not fatal
        }
    }

    /**
     * Extract the per-vector lowerInterval from a block read by {@link #readBlock}.
     * The lower interval is stored as a little-endian float at offset {@code packedResidualBytes}.
     */
    public float extractLower(byte[] block) {
        return readFloatLE(block, packedResidualBytes);
    }

    /**
     * Extract the per-vector upperInterval from a block read by {@link #readBlock}.
     * The upper interval is stored as a little-endian float at offset {@code packedResidualBytes + 4}.
     */
    public float extractUpper(byte[] block) {
        return readFloatLE(block, packedResidualBytes + 4);
    }

    @Override
    public void close() throws IOException {
        verInput.close();
    }

    /**
     * Read a little-endian float from a byte array at the given offset.
     */
    private static float readFloatLE(byte[] buffer, int offset) {
        int bits = (buffer[offset] & 0xFF)
            | ((buffer[offset + 1] & 0xFF) << 8)
            | ((buffer[offset + 2] & 0xFF) << 16)
            | ((buffer[offset + 3] & 0xFF) << 24);
        return Float.intBitsToFloat(bits);
    }
}
