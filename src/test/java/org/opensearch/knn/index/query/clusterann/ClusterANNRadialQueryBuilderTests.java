/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clusterann;

import com.carrotsearch.randomizedtesting.ThreadFilter;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import com.google.common.collect.ImmutableMap;
import org.apache.lucene.search.Query;
import org.apache.lucene.tests.util.LuceneTestCase;

import org.opensearch.core.index.Index;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.index.query.KNNQueryBuilder;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.METHOD_CLUSTER;

/**
 * {@code KNNQueryBuilder} on a ClusterANN field: radial parameters build a {@link ClusterANNRadialQuery} with the radius
 * converted by the Lucene engine's formulas, and the top-k path is untouched.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { ClusterANNRadialQueryBuilderTests.JnaCleanerThreadFilter.class })
public class ClusterANNRadialQueryBuilderTests extends LuceneTestCase {

    /** Building a knn query loads the JNA-backed native helpers, whose cleaner thread outlives the suite. */
    public static class JnaCleanerThreadFilter implements ThreadFilter {
        @Override
        public boolean reject(final Thread t) {
            return t.getName().contains("JNA Cleaner");
        }
    }

    private static final String FIELD = "cluster_vec";
    private static final float[] VECTOR = { 1f, 2f, 3f, 4f };

    private static QueryShardContext clusterAnnContext(final SpaceType spaceType) {
        final QueryShardContext context = mock(QueryShardContext.class);
        final KNNVectorFieldType field = mock(KNNVectorFieldType.class);
        when(context.index()).thenReturn(new Index("dummy", "dummy"));
        when(context.fieldMapper(anyString())).thenReturn(field);
        when(context.getFieldType(anyString())).thenReturn(field);
        when(context.getShardId()).thenReturn(3);
        when(field.isClusterANN()).thenReturn(true);
        when(field.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(field.transformQueryVector(any())).thenReturn(VECTOR);
        final KNNMethodContext method = new KNNMethodContext(
            KNNEngine.UNDEFINED,
            spaceType,
            new MethodComponentContext(METHOD_CLUSTER, ImmutableMap.of())
        );
        when(field.getKnnMappingConfig()).thenReturn(KNNTestCase.getMappingConfigForMethodMapping(method, VECTOR.length));
        return context;
    }

    public void testMaxDistance_l2_buildsRadialQueryWithLuceneScoreThreshold() throws Exception {
        final KNNQueryBuilder builder = KNNQueryBuilder.builder().fieldName(FIELD).vector(VECTOR).maxDistance(10f).build();
        final Query query = builder.toQuery(clusterAnnContext(SpaceType.L2));
        assertTrue(query.toString(), query instanceof ClusterANNRadialQuery);
        final ClusterANNRadialQuery radial = (ClusterANNRadialQuery) query;
        assertEquals(FIELD, radial.getField());
        // Lucene L2 score = 1 / (1 + distance)
        assertEquals(1f / 11f, radial.getResultSimilarity(), 1e-6f);
        assertNull(radial.getFilter());
    }

    public void testMinScore_l2_isUsedAsIs() throws Exception {
        final KNNQueryBuilder builder = KNNQueryBuilder.builder().fieldName(FIELD).vector(VECTOR).minScore(0.5f).build();
        final ClusterANNRadialQuery radial = (ClusterANNRadialQuery) builder.toQuery(clusterAnnContext(SpaceType.L2));
        assertEquals(0.5f, radial.getResultSimilarity(), 0f);
    }

    public void testMinScore_innerProduct_isUsedAsIs() throws Exception {
        final KNNQueryBuilder builder = KNNQueryBuilder.builder().fieldName(FIELD).vector(VECTOR).minScore(2.0f).build();
        final ClusterANNRadialQuery radial = (ClusterANNRadialQuery) builder.toQuery(clusterAnnContext(SpaceType.INNER_PRODUCT));
        assertEquals(2.0f, radial.getResultSimilarity(), 0f);
    }

    public void testMaxDistance_innerProduct_convertsLikeLuceneEngine() throws Exception {
        final KNNQueryBuilder builder = KNNQueryBuilder.builder().fieldName(FIELD).vector(VECTOR).maxDistance(-2f).build();
        final ClusterANNRadialQuery radial = (ClusterANNRadialQuery) builder.toQuery(clusterAnnContext(SpaceType.INNER_PRODUCT));
        assertEquals(KNNEngine.LUCENE.distanceToRadialThreshold(-2f, SpaceType.INNER_PRODUCT), radial.getResultSimilarity(), 1e-6f);
    }

    public void testTopK_stillBuildsTheClusterAnnQuery() throws Exception {
        final KNNQueryBuilder builder = KNNQueryBuilder.builder().fieldName(FIELD).vector(VECTOR).k(10).build();
        final Query query = builder.toQuery(clusterAnnContext(SpaceType.L2));
        assertTrue(query.toString(), query instanceof ClusterANNRescoreQuery);
    }
}
