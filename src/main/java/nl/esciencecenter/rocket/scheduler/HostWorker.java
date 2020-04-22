package nl.esciencecenter.rocket.scheduler;

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import nl.esciencecenter.rocket.cache.DistributedCache;
import nl.esciencecenter.rocket.cache.FileCache;
import nl.esciencecenter.rocket.cache.HostCache;
import nl.esciencecenter.rocket.cubaapi.CudaPinned;
import nl.esciencecenter.rocket.profiling.Profiler;
import nl.esciencecenter.rocket.util.Future;
import nl.esciencecenter.rocket.util.Util;
import nl.esciencecenter.xenon.filesystems.FileSystem;
import nl.esciencecenter.xenon.filesystems.Path;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.HashSet;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

public class HostWorker<K> {
    protected static final Logger logger = LogManager.getLogger();

    private FileSystem fs;
    private Profiler profiler;
    private Optional<FileCache> fileCache;
    private Optional<DistributedCache> distributedCache;
    private Optional<HostCache> hostCache;
    private ThreadPoolExecutor localIoExecutor;
    private ThreadPoolExecutor remoteIoExecutor;
    private ThreadPoolExecutor generalExecutor;
    private Set<K> keysSeen;

    static ThreadPoolExecutor newFixedThreadPool(String nameFormat, int size) {
        ThreadFactory factory = new ThreadFactoryBuilder().setNameFormat(nameFormat).build();
        return new ThreadPoolExecutor(size, size, 0, TimeUnit.SECONDS, new LinkedBlockingQueue<>(), factory);
    }

    public HostWorker(
            FileSystem fs,
            Profiler profiler,
            Optional<HostCache> hostCache,
            Optional<FileCache> fileCache,
            Optional<DistributedCache> distributedCache,
            int hostThreads) {
        if (hostCache.map(c -> c.getNumEntries() < 2).orElse(false)) {
            throw new IllegalArgumentException("cache requires at least two entries");
        }

        this.fs = fs;
        this.profiler = profiler;
        this.hostCache = hostCache;
        this.fileCache = fileCache;
        this.distributedCache = distributedCache;
        this.keysSeen = new HashSet<>();
        this.generalExecutor = newFixedThreadPool("host-%d", hostThreads);
        this.remoteIoExecutor = newFixedThreadPool("io-%d", 1);
        this.localIoExecutor = newFixedThreadPool("file-%d", 1);
    }

    public Future<ByteBuffer[]> loadFilesAsync(Path[] paths) {
        return Future.runAsync(
                remoteIoExecutor,
                () -> {
                    try (Profiler.Record record = profiler.trace("copy_input")) {
                        ByteBuffer[] buffers = new ByteBuffer[paths.length];
                        long total = 0;

                        for (int i = 0; i < paths.length; i++) {
                            Path path = paths[i];
                            try (InputStream s = Util.readFile(fs, path)) {
                                byte[] buffer = s.readAllBytes();

                                buffers[i] = ByteBuffer.wrap(buffer);
                                total += buffer.length;
                            }
                        }

                        profiler.report("bytes transferred IO", total);
                        profiler.report("files transferred IO", paths.length);
                        return buffers;
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }
        );
    }

    public <T> Future<T> submitHostTask(String name, Supplier<T> task) {
        return Future.runAsync(
                generalExecutor,
                () ->  {
                    try (Profiler.Record record = profiler.trace(name)) {
                        return task.get();
                    }
                }
        );
    }

    public interface CacheHitCallback<K> {
        public Future<Void> call(CudaPinned src);
    }

    public interface CacheMissCallback<K> {
        public Future<Long> call(CudaPinned dst);
    }

    private Future<Void> loadInput(
            K key,
            HostCache.Entry hostMem,
            CacheMissCallback<K> miss
    ) {
        // Call miss
        return miss.call(hostMem.write())

                // Complete host cache entry
                .thenRun(hostMem::downgradeToReader)

                .andVoid();
    }

    private Future<Void> loadInputFromDistributedCacheAsync(
            K key,
            HostCache.Entry hostMem,
            CacheHitCallback<K> hit,
            CacheMissCallback<K> miss
    ) {
        if (!distributedCache.isPresent()) {
            return loadInput(key, hostMem, miss);
        }

        logger.trace("loading input from distributed cache {}", key);
        ByteBuffer buffer = hostMem.write().asByteBuffer();
        Future<Optional<Long>> fut = distributedCache.get().request(key.toString(), buffer);

        return fut.then(sizeOpt -> {
            // Successfull requested buffer from distributed cache
            if (sizeOpt.isPresent()) {
                long size = sizeOpt.get();

                logger.trace("distributed cache hit for {}", key);
                profiler.report("host-general","distributed cache hits");
                profiler.report("host-general","bytes received from remote node", size);

                return Future.runAsync(generalExecutor, () -> {
                    hostMem.downgradeToReader(size);
                    return hit.call(hostMem.read().sliceBytes(0, size));
                }).then(x -> x);
            }

            // Failed to request item from distributed cache, load item ourselves
            else {
                logger.trace("distributed cache miss for {}", key);
                profiler.report("host-general","distributed cache misses");
                return loadInput(key, hostMem, miss);
            }
        });
    }

    private Future<Void> loadFromFileCacheAsync(
            K f,
            HostCache.Entry hostMem,
            CacheHitCallback<K> hit,
            CacheMissCallback<K> miss
    ) {
        if (fileCache.isEmpty()) {
            return loadInputFromDistributedCacheAsync(f, hostMem, hit, miss);
        }

        // Acquire slot in file cache
        Future<FileCache.Entry> fut = fileCache.get().acquireAsync(f.toString());

        logger.trace("loading input from file cache {}", f);
        return fut.then(entry -> {

            // Cache miss
            if (entry.isWriter()) {
                logger.trace("file cache miss for {}", f);
                profiler.report("file cache misses");

                // Attempt to load from distributed cache.
                return loadInputFromDistributedCacheAsync(f, hostMem, hit, miss)

                        // Copy data from memory to local file
                        .thenRunAsync(localIoExecutor, __ -> {
                            profiler.run("write_slot", () -> {
                                long size = hostMem.size();

                                try {
                                    entry.write().writeSlot(hostMem.read().asByteBuffer(), size);
                                    entry.downgradeToReader(size);
                                } catch (IOException e) {
                                    throw new RuntimeException(e);
                                }
                            });
                        })

                        // Always release entry
                        .onComplete((r, e) -> entry.release());

            }

            // Cache hit
            else {
                logger.trace("file cache hit for {}", f);
                profiler.report("file cache hits");

                // Copy data from local file to host memory
                return Future.runAsync(localIoExecutor, () -> {
                            return profiler.run("read_slot", () -> {
                                long size = entry.size();

                                try {
                                    entry.read().readSlot(hostMem.write().asByteBuffer(), size);
                                    hostMem.downgradeToReader(size);
                                } catch (IOException e) {
                                    throw new RuntimeException(e);
                                }

                                return size;
                            });
                        })

                        // Always release entry
                        .onComplete((r, e) -> entry.release())

                        // call hit callback
                        .then(size -> hit.call(hostMem.read().sliceBytes(0, size)));


            }
        });
    }

    public Future<Void> loadFromHostCacheAsync(
            K key,
            CacheHitCallback<K> hit,
            CacheMissCallback<K> miss
    ) {
        if (keysSeen.add(key)) {
            profiler.report("first-time cache requests");
        }

        Future<HostCache.Entry> entryFut = hostCache.get().acquireAsync(key.toString());

        return entryFut.then(entry -> {

                    // Cache miss, try file cache
                    if (entry.isWriter()) {
                        logger.trace("host cache miss for {}", key);
                        profiler.report("host cache misses");

                        return loadFromFileCacheAsync(key, entry, hit, miss);
                    }

                    // Cache hit
                    else {
                        logger.trace("host cache hit for {}", key);
                        profiler.report("host cache hits");

                        return hit.call(entry.read().sliceBytes(0, entry.size()));
                    }
                })

                .onComplete((r, e) -> entryFut.thenRun(HostCache.Entry::release));
    }

    public boolean hasHostCache() {
        return hostCache.isPresent();
    }

    public void cleanup() {
        this.distributedCache.ifPresent(c -> c.shutdown());

        this.remoteIoExecutor.shutdown();
        this.localIoExecutor.shutdown();
        this.generalExecutor.shutdown();

        this.fileCache.ifPresent(c -> c.cleanup());
        this.hostCache.ifPresent(c -> c.cleanup());
    }
}
