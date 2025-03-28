/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.memory;

import lombok.Getter;
import lombok.Setter;
import org.apache.lucene.index.LeafReaderContext;
import org.opensearch.knn.common.featureflags.KNNFeatureFlags;
import org.opensearch.common.concurrent.RefCountedReleasable;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.query.KNNWeight;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Semaphore;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Represents a persistent allocation made in native memory. In this case, persistent means that the allocation is made
 * and not freed in the same call to the JNI. Therefore, in order to prevent memory leaks, we need to ensure that each
 * allocation is properly freed
 */
public interface NativeMemoryAllocation {

    /**
     * Closes the native memory allocation. It should deallocate all native memory associated with this allocation.
     */
    void close();

    /**
     * Check if the allocation has been closed.
     *
     * @return true if allocation has been closed; false otherwise
     */
    boolean isClosed();

    /**
     * Get the native memory address associated with the native memory allocation.
     *
     * @return memory address of native memory allocation
     */
    long getMemoryAddress();

    /**
     * Locks allocation for read. Multiple threads can obtain this lock assuming that no threads have the write lock.
     */
    void readLock();

    /**
     * Locks allocation for write. Only one thread can obtain this lock and no threads can have a read lock.
     */
    void writeLock();

    /**
     * Unlocks allocation for read.
     */
    void readUnlock();

    /**
     * Unlocks allocation for write.
     */
    void writeUnlock();

    /**
     * Get the size of the native memory allocation in kilobytes.
     *
     * @return size of native memory allocation
     */
    int getSizeInKB();

    /**
     * Increments the refCount of this instance.
     *
     * @see #decRef
     * @throws IllegalStateException iff the reference counter can not be incremented.
     */
    default void incRef() {}

    /**
     * Decreases the refCount of this  instance. If the refCount drops to 0, then this
     * instance is considered as closed and should not be used anymore.
     *
     * @see #incRef
     *
     * @return returns {@code true} if the ref count dropped to 0 as a result of calling this method
     */
    default boolean decRef() {
        return true;
    }

    /**
     * Represents native indices loaded into memory. Because these indices are backed by files, they should be
     * freed when file is deleted.
     */
    class IndexAllocation implements NativeMemoryAllocation {

        private final ExecutorService executor;
        private final long memoryAddress;
        private final int sizeKb;
        private volatile boolean closed;
        @Getter
        private final KNNEngine knnEngine;
        @Getter
        private final String vectorFileName;
        @Getter
        private final String openSearchIndexName;
        private final ReadWriteLock readWriteLock;
        private final SharedIndexState sharedIndexState;
        @Getter
        private final boolean isBinaryIndex;
        private final RefCountedReleasable<IndexAllocation> refCounted;

        /**
         * Constructor
         *
         * @param executorService Executor service used to close the allocation
         * @param memoryAddress Pointer in memory to the index
         * @param sizeKb Size this index consumes in kilobytes
         * @param knnEngine KNNEngine associated with the index allocation
         * @param vectorFileName Vector file name. Ex: _0_165_my_field.faiss
         * @param openSearchIndexName Name of OpenSearch index this index is associated with
         */
        IndexAllocation(
            ExecutorService executorService,
            long memoryAddress,
            int sizeKb,
            KNNEngine knnEngine,
            String vectorFileName,
            String openSearchIndexName
        ) {
            this(executorService, memoryAddress, sizeKb, knnEngine, vectorFileName, openSearchIndexName, null, false);
        }

        /**
         * Constructor
         *
         * @param executorService Executor service used to close the allocation
         * @param memoryAddress Pointer in memory to the index
         * @param sizeKb Size this index consumes in kilobytes
         * @param knnEngine KNNEngine associated with the index allocation
         * @param vectorFileName Vector file name. Ex: _0_165_my_field.faiss
         * @param openSearchIndexName Name of OpenSearch index this index is associated with
         * @param sharedIndexState Shared index state. If not shared state present, pass null.
         */
        IndexAllocation(
            ExecutorService executorService,
            long memoryAddress,
            int sizeKb,
            KNNEngine knnEngine,
            String vectorFileName,
            String openSearchIndexName,
            SharedIndexState sharedIndexState,
            boolean isBinaryIndex
        ) {
            this.executor = executorService;
            this.closed = false;
            this.knnEngine = knnEngine;
            this.vectorFileName = vectorFileName;
            this.openSearchIndexName = openSearchIndexName;
            this.memoryAddress = memoryAddress;
            this.readWriteLock = new ReentrantReadWriteLock();
            this.sizeKb = sizeKb;
            this.sharedIndexState = sharedIndexState;
            this.isBinaryIndex = isBinaryIndex;
            this.refCounted = new RefCountedReleasable<>("IndexAllocation-Reference", this, this::closeInternal);
        }

        protected void closeInternal() {
            Runnable onClose = () -> {
                writeLock();
                try {
                    cleanup();
                } finally {
                    writeUnlock();
                }
            };

            // The close operation needs to be blocking to prevent overflow
            // This blocks any entry until the close has completed, preventing creation before close scenarios
            if (KNNFeatureFlags.isForceEvictCacheEnabled()) {
                onClose.run();
            } else {
                executor.execute(onClose);
            }
        }

        @Override
        public void close() {
            if (!closed && refCounted.refCount() > 0) {
                refCounted.close();
            }
        }

        private void cleanup() {
            if (this.closed) {
                return;
            }

            this.closed = true;

            // memoryAddress is sometimes initialized to 0. If this is ever the case, freeing will surely fail.
            if (memoryAddress != 0) {
                JNIService.free(memoryAddress, knnEngine, isBinaryIndex);
            }

            if (sharedIndexState != null) {
                SharedIndexStateManager.getInstance().release(sharedIndexState);
            }
        }

        @Override
        public boolean isClosed() {
            return closed;
        }

        @Override
        public long getMemoryAddress() {
            return memoryAddress;
        }

        /**
         * The read lock will be obtained in the
         * {@link KNNWeight#scorer(LeafReaderContext context) scorer} when a native index needs
         * to be queried.
         */
        @Override
        public void readLock() {
            readWriteLock.readLock().lock();
        }

        /**
         * The write lock will be obtained in the
         * {@link NativeMemoryCacheManager NativeMemoryManager's} onRemoval function when the Index Allocation is
         * evicted from the cache. This prevents memory from being deallocated when it is being actively searched.
         */
        @Override
        public void writeLock() {
            readWriteLock.writeLock().lock();
        }

        @Override
        public void readUnlock() {
            readWriteLock.readLock().unlock();
        }

        @Override
        public void writeUnlock() {
            readWriteLock.writeLock().unlock();
        }

        @Override
        public int getSizeInKB() {
            return sizeKb;
        }

        @Override
        public void incRef() {
            refCounted.incRef();
        }

        @Override
        public boolean decRef() {
            return refCounted.decRef();
        }
    }

    /**
     * Represents training data that has been allocated in native memory.
     */
    class TrainingDataAllocation implements NativeMemoryAllocation {

        private final ExecutorService executor;

        private volatile boolean closed;
        @Setter
        private long memoryAddress;
        private final int sizeKb;
        @Getter
        @Setter
        private QuantizationConfig quantizationConfig = QuantizationConfig.EMPTY;

        // Implement reader/writer with semaphores to deal with passing lock conditions between threads
        private int readCount;
        private final Semaphore readSemaphore;
        private final Semaphore writeSemaphore;
        private final VectorDataType vectorDataType;

        /**
         * Constructor
         *
         * @param executor Executor used for allocation close
         * @param memoryAddress pointer in memory to the training data allocation
         * @param sizeKb amount memory needed for allocation in kilobytes
         */
        public TrainingDataAllocation(ExecutorService executor, long memoryAddress, int sizeKb, VectorDataType vectorDataType) {
            this.executor = executor;
            this.closed = false;
            this.memoryAddress = memoryAddress;
            this.sizeKb = sizeKb;

            this.readCount = 0;
            this.readSemaphore = new Semaphore(1);
            this.writeSemaphore = new Semaphore(1);
            this.vectorDataType = vectorDataType;
        }

        @Override
        public void close() {
            executor.execute(() -> {
                writeLock();
                try {
                    cleanup();
                } finally {
                    writeUnlock();
                }
            });
        }

        /**
         * Unsafe close operation. This method assumes that the calling thread already has the writeLock. Thus,
         * the executor can go ahead and cleanup the allocation and then release the write lock. Use with caution.
         */
        public void closeUnsafe() {
            executor.execute(() -> {
                cleanup();
                writeUnlock();
            });
        }

        private void cleanup() {
            if (closed) {
                return;
            }

            closed = true;

            if (this.memoryAddress != 0) {
                vectorDataType.freeNativeMemory(this.memoryAddress);
            }
        }

        @Override
        public boolean isClosed() {
            return closed;
        }

        @Override
        public long getMemoryAddress() {
            return memoryAddress;
        }

        /**
         * A read lock will be obtained when a training job needs access to the TrainingDataAllocation.
         * In the future, we may want to switch to tryAcquire functionality.
         */
        @Override
        public void readLock() {
            try {
                readSemaphore.acquire();
            } catch (InterruptedException ex) {
                throw new RuntimeException(ex);
            }

            // If the read count is 0, we need to grab the permit for the write lock. This is so that the write permit
            // cannot be grabbed when there are read locks in use. In readUnlock, if the readCount goes to 0, we
            // release the writeLock
            if (readCount == 0) {
                try {
                    writeLock();
                } catch (RuntimeException e) {
                    readSemaphore.release();
                    throw e;
                }
            }

            readCount++;
            readSemaphore.release();
        }

        /**
         * A write lock will be obtained either on eviction from {@link NativeMemoryCacheManager NativeMemoryManager's}
         * or when training data is actually being loaded. A semaphore is used because collecting training data
         * happens asynchronously, so the thread that obtains the lock will not be the same thread that releases the
         * lock.
         */
        @Override
        public void writeLock() {
            try {
                writeSemaphore.acquire();
            } catch (InterruptedException ex) {
                throw new RuntimeException(ex);
            }
        }

        @Override
        public void readUnlock() {
            try {
                readSemaphore.acquire();
            } catch (InterruptedException ex) {
                throw new RuntimeException(ex);
            }

            readCount--;

            // The read count should never be less than 0, but add <= here just to be on the safe side.
            if (readCount <= 0) {
                writeUnlock();
            }

            readSemaphore.release();
        }

        @Override
        public void writeUnlock() {
            writeSemaphore.release();
        }

        @Override
        public int getSizeInKB() {
            return sizeKb;
        }
    }

    /**
     * An anonymous allocation is used to reserve space in the native memory cache. It does not have a
     * memory address. This allocation type should be used when a function allocates a large portion of memory in the
     * function, runs for a while, and then frees it.
     */
    class AnonymousAllocation implements NativeMemoryAllocation {

        private final ExecutorService executor;
        private volatile boolean closed;
        private final int size;
        private final ReadWriteLock readWriteLock;

        AnonymousAllocation(ExecutorService executor, int size) {
            this.executor = executor;
            this.closed = false;
            this.size = size;
            this.readWriteLock = new ReentrantReadWriteLock();
        }

        @Override
        public void close() {
            if (isClosed()) {
                return;
            }

            executor.execute(() -> {
                writeLock();
                closed = true;
                writeUnlock();
            });
        }

        @Override
        public boolean isClosed() {
            return closed;
        }

        @Override
        public long getMemoryAddress() {
            throw new UnsupportedOperationException("Cannot get memory address for an AnonymousAllocation.");
        }

        @Override
        public void readLock() {
            readWriteLock.readLock().lock();
        }

        @Override
        public void writeLock() {
            readWriteLock.writeLock().lock();
        }

        @Override
        public void readUnlock() {
            readWriteLock.readLock().unlock();
        }

        @Override
        public void writeUnlock() {
            readWriteLock.writeLock().unlock();
        }

        @Override
        public int getSizeInKB() {
            return size;
        }
    }
}
