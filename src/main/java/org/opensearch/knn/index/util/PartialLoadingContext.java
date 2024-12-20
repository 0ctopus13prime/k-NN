/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.MMapDirectory;
import org.opensearch.knn.index.store.IndexInputThreadLocalGetter;
import org.opensearch.knn.index.store.partial_loading.KdyHNSW;

import java.nio.file.Paths;

public class PartialLoadingContext {
    public final IndexInputThreadLocalGetter indexInputThreadLocalGetter;
    public final KdyHNSW kdyHNSW;

    public PartialLoadingContext(IndexInputThreadLocalGetter indexInputThreadLocalGetter, KdyHNSW kdyHNSW) {
        this.indexInputThreadLocalGetter = indexInputThreadLocalGetter;
        this.kdyHNSW = kdyHNSW;
    }


    public String getMMapFilePathIfAvailable() {
        final Directory directory = indexInputThreadLocalGetter.getDirectory();
        final Directory innerDirectory = FilterDirectory.unwrap(directory);
        System.out.println(
            "___________________________ innerDirectory instanceof MMapDirectory: " + (innerDirectory instanceof MMapDirectory)
        );
        System.out.println("___________________________ innerDirectory=" + innerDirectory);
        if (innerDirectory instanceof MMapDirectory) {
            final MMapDirectory mmapDirectory = (MMapDirectory) innerDirectory;
            final String filePath = mmapDirectory.getDirectory()
                .resolve(Paths.get(indexInputThreadLocalGetter.getVectorFileName()))
                .toString();
            return filePath;
        }

        return null;
    }
}
