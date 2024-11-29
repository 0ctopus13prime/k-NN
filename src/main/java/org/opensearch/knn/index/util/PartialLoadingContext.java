/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import lombok.AllArgsConstructor;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.MMapDirectory;
import org.opensearch.knn.index.store.IndexInputThreadLocalGetter;

import java.nio.file.Paths;

@AllArgsConstructor
public class PartialLoadingContext {
    private IndexInputThreadLocalGetter indexInputThreadLocalGetter;

    public String getMMapFilePathIfAvailable() {
        final Directory directory = indexInputThreadLocalGetter.getDirectory();
        final Directory innerDirectory = FilterDirectory.unwrap(directory);
        if (innerDirectory instanceof MMapDirectory) {
            final MMapDirectory mmapDirectory = (MMapDirectory) innerDirectory;
            final String filePath =
                mmapDirectory.getDirectory().resolve(Paths.get(indexInputThreadLocalGetter.getVectorFileName())).toString();
            return filePath;
        }

        return null;
    }
}
