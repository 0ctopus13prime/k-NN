/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.kdy;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.StringHelper;

import java.io.IOException;

public class SegmentIdExtractor {
    public final byte[] segmentId;
    public final String segmentSuffix;

    public SegmentIdExtractor(
        final IndexInput metaInput
    ) throws IOException {
        // Seek to the 0th
        metaInput.seek(0);
        // Header
        final int headerMagicNumber = CodecUtil.readBEInt(metaInput);
        // Codec name
        final String codec = metaInput.readString();
        // Version
        final int version = CodecUtil.readBEInt(metaInput);
        // Segment id
        segmentId = new byte[StringHelper.ID_LENGTH];
        metaInput.readBytes(segmentId, 0, segmentId.length);
        final int suffixLength = metaInput.readByte() & 0xFF;
        final byte[] suffixBytes = new byte[suffixLength];
        metaInput.readBytes(suffixBytes, 0, suffixLength);
        segmentSuffix = new String(suffixBytes);
    }
}
