package nl.esciencecenter.rocket;

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import nl.esciencecenter.rocket.cache.DeviceCache;
import nl.esciencecenter.rocket.cache.DistributedCache;
import nl.esciencecenter.rocket.cache.FileCache;
import nl.esciencecenter.rocket.cache.HostCache;
import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaMem;
import nl.esciencecenter.rocket.cubaapi.CudaMemByte;
import nl.esciencecenter.rocket.cubaapi.CudaPinned;
import nl.esciencecenter.rocket.cubaapi.CudaStream;
import nl.esciencecenter.rocket.profiling.Profiler;
import nl.esciencecenter.rocket.types.ApplicationContext;
import nl.esciencecenter.rocket.types.HashableKey;
import nl.esciencecenter.rocket.types.InputTask;
import nl.esciencecenter.rocket.types.LeafTask;
import nl.esciencecenter.rocket.util.Future;
import nl.esciencecenter.rocket.util.FutureQueue;
import nl.esciencecenter.rocket.util.PriorityRunnable;
import nl.esciencecenter.rocket.util.Util;
import nl.esciencecenter.xenon.filesystems.FileSystem;
import nl.esciencecenter.xenon.filesystems.Path;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

public class worker {
    public static class DeviceWorker<R> {
        protected static final Logger logger = LogManager.getLogger();

        private class ScratchMem {
            CudaMemByte dmem;
            CudaPinned hmem;

            private ScratchMem(CudaContext ctx, long size) {
                dmem = ctx.allocBytes(size);
                hmem = ctx.allocHostBytes(size);
            }

            private void free() {
                dmem.free();
                hmem.free();
            }
        }

        String name;
        private HostWorker hostWorker;
        private Profiler profiler;
        private CudaContext context;
        private ThreadPoolExecutor h2dExecutor;
        private ThreadPoolExecutor d2hExecutor;
        private ThreadPoolExecutor gpuExecutor;

        private ApplicationContext applicationContext;

        private CudaStream h2dStream;
        private CudaStream d2hStream;

        private Semaphore semaphore;
        private FutureQueue<ScratchMem> inputMems;
        private FutureQueue<ScratchMem> outputMems;

        private DeviceCache devCache;

        static ThreadPoolExecutor newFixedThreadPool(String nameFormat, int size) {
            return newFixedThreadPool(nameFormat, size, new LinkedBlockingQueue<>());
        }

        static ThreadPoolExecutor newFixedThreadPool(String nameFormat, int size, BlockingQueue<Runnable> queue) {
            ThreadFactory factory = new ThreadFactoryBuilder().setNameFormat(nameFormat).build();
            return new ThreadPoolExecutor(size, size, 0, TimeUnit.SECONDS, queue, factory);
        }

        public DeviceWorker(
                HostWorker hworker,
                Profiler profiler,
                CudaContext ctx,
                ApplicationContext applicationContext,
                int numInputs,
                int numOutputs,
                int numTasks,
                DeviceCache devCache
        ) {
            if (devCache.getNumEntries() < 2) {
                throw new IllegalArgumentException("cache requires at least two entries");
            }

            this.hostWorker = hworker;
            this.context = ctx;
            this.applicationContext = applicationContext;
            this.profiler = profiler;

            name = ctx.getDevice().getName() + "-" + ctx.getDevice().getDeviceNum();
            PriorityBlockingQueue<Runnable> gpuQueue = new PriorityBlockingQueue<>();
            this.gpuExecutor = newFixedThreadPool(name + "-execute", 1, gpuQueue);
            this.h2dExecutor = newFixedThreadPool(name + "-h2d", 1);
            this.d2hExecutor = newFixedThreadPool(name + "-d2h", 1);

            this.semaphore = new Semaphore(numTasks);

            this.inputMems = new FutureQueue<>();
            for (int i = 0; i < numInputs; i++) {
                long bufferSize = applicationContext.getMaxFileSize();
                inputMems.push(new ScratchMem(ctx, bufferSize));
            }

            this.outputMems = new FutureQueue<>();
            for (int i = 0; i < numOutputs; i++) {
                long bufferSize = applicationContext.getMaxOutputSize();
                outputMems.push(new ScratchMem(ctx, bufferSize));
            }

            this.devCache = devCache;
            this.h2dStream = ctx.createStream();
            this.d2hStream = ctx.createStream();
        }

        private Future<Long> loadInput(InputTask task, CudaMem devMem) {
            logger.trace("load input {}", task.getKey());

            Future<ScratchMem> scratch = new Future<>();
            Path[] paths = task.getInputs();
            Future<ByteBuffer[]> buffers;

            return (buffers = hostWorker.loadFilesAsync(paths))

                    // acquire scratch memory
                    .then(__ -> inputMems.popAsync().thenComplete(scratch))

                    // Load item
                    .then(__ -> hostWorker.submitHostTask(
                            "load_input",
                            () -> task.preprocess(buffers.get(), scratch.get().hmem.asByteBuffer())
                    ))

                    // Copy input to device
                    .then(size -> submitH2D(
                            scratch.get().hmem,
                            scratch.get().dmem,
                            size
                    ).andReturn(size))

                    // Run preprocessing
                    .then(size -> submitDeviceTask(
                            Long.MAX_VALUE,
                            "preprocess_input",
                            () -> task.execute(
                                    applicationContext,
                                    scratch.get().dmem.slice(0, size),
                                    devMem.asBytes())
                    ))

                    // Always release scratch memory.
                    .onComplete((r, e) -> {
                        scratch.thenRun(s -> inputMems.push(s));
                    });
        }

        private Future<Long> loadInputFromHostCacheAsync(InputTask input, CudaMem dmem) {
            HashableKey f = input.getKey();
            logger.trace("load input from host cache {}", f);

            if (hostWorker.hasHostCache()) {
                Future<Long> deviceDone = new Future<>();
                Future<Void> hostDone = hostWorker.loadFromHostCacheAsync(
                        f,

                        // Called on cache hit. We must copy from host to device memory.
                        hostSrc -> {
                            logger.trace("hostCache onHit {}", f);
                            long size = hostSrc.sizeInBytes();

                            return submitH2D(hostSrc, dmem, size)
                                    .thenRun(__ -> deviceDone.complete(size));
                        },

                        // Called on cache miss. We must copy from device to host memory (i.e., fill the host cache slot).
                        hostDst -> {
                            logger.trace("hostCache onMiss {}", f);
                            return loadInput(input, dmem)
                                    .thenRun(size -> deviceDone.complete(size))
                                    .then(size -> submitD2H(dmem, hostDst, size).andReturn(size));
                        });

                hostDone.onComplete((result, error) -> {
                    if (error != null) {
                        deviceDone.completeExceptionally(error);
                    } else {
                        deviceDone.completeExceptionally(new IllegalStateException("Host cache completed without called" +
                                "either the onHit or onMiss callback. This is an internal error and should not happen."));
                    }
                });


                return deviceDone;
            } else {

                // Load Input
                return loadInput(input, dmem);
            }
        }

        private Future<DeviceCache.Entry[]> loadInputFromDeviceCacheAsync(InputTask[] inputs) {
            HashableKey[] keys = new HashableKey[inputs.length];
            for (int i = 0; i < keys.length; i++) {
                keys[i] = inputs[i].getKey();
            }

            logger.trace("load input from device cache {}", (Object) keys);

            return devCache
                    .acquireAllAsync(keys)
                    .then(entries -> {
                        Future<DeviceCache.Entry[]> done = Future.ready(entries);

                        // Iterate over all entries to load them into memory if required.
                        for (DeviceCache.Entry entry: entries) {

                            // Cache miss, attempt to load data from host cache.
                            if (entry.isWriter()) {
                                HashableKey key = entry.getKey();
                                InputTask input = null;
                                for (int i = 0; i < keys.length; i++) {
                                    if (key.equals(keys[i])) {
                                        input = inputs[i];
                                        break;
                                    }
                                }

                                if (input == null) {
                                    throw new IllegalStateException();
                                }


                                profiler.report(name, "device cache misses", 1);
                                done = loadInputFromHostCacheAsync(input, entry.write())

                                        // Entry is now ready. Downgrade to reader.
                                        .thenRun(size -> entry.downgradeToReader(size))

                                        // Join with done
                                        .thenJoin(done, (__, v) -> v);
                            }

                            // Cache hit, we are done.
                            else {
                                profiler.report(name, "device cache hits", 1);
                            }
                        }

                        return done;
                    });
        }

        public Future<R> submit(LeafTask<R> task) {
            var scope = new Object() {
                ScratchMem scratch;
            };

            // Priority is set negative of the current time, this means that older jobs always run before newer jobs
            // (since the priority of newer jobs is "more negative" and thus lower.
            long priority = -System.nanoTime();

            // Acquire semaphore to limit number of concurrent jobs
            try {
                semaphore.acquire();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }

            logger.trace("start compute correlation {}", task);
            profiler.report("correlations executed", 1);
            Profiler.Record record = profiler.traceCorrelation(name, task);

            // Asynchronously load data from cache
            List<InputTask> inputs = task.getInputs();
            Future<DeviceCache.Entry[]> fut = loadInputFromDeviceCacheAsync(inputs.toArray(new InputTask[2]));

            // Wait until both cache entries have been loaded
            return fut

                    // Obtain output scratch memory to write results.
                    .then(__ -> outputMems.popAsync().thenRun(v -> scope.scratch = v))

                    // Submit correlation to GPU
                    .then(__ -> {
                        DeviceCache.Entry[] entries = fut.get();
                        CudaMem[] mems = new CudaMem[entries.length];

                        for (int i = 0; i < mems.length; i++) {
                            boolean found = false;
                            HashableKey key = inputs.get(i).getKey();

                            for (DeviceCache.Entry entry: entries) {
                                if (entry.getKey().equals(key)) {
                                    mems[i] = entry.read().slice(0, entry.size());
                                    found = true;
                                }
                            }

                            if (!found) {
                                throw new IllegalStateException("cache did not return requested entry: " + key);
                            }
                        }

                        return submitDeviceTask(
                            priority,
                            "correlate",
                            () -> task.execute(
                                    applicationContext,
                                    mems,
                                    scope.scratch.dmem
                            )
                        );
                    })

                    // Release cache entries, they are no longer needed.
                    .onComplete((r, e) -> {
                        // Release all entries
                        fut.thenRun(entries -> {
                            for (DeviceCache.Entry entry: entries) {
                                entry.release();
                            }
                        });
                    })

                    // Copy results to host.
                    .then(size ->
                            submitD2H(
                                scope.scratch.dmem,
                                scope.scratch.hmem,
                                size
                            ).andReturn(size)
                    )

                    // Post-process results on host.
                    .then(size ->
                            hostWorker.submitHostTask(
                                    "store_output",
                                    () -> task.postprocess(
                                            applicationContext,
                                            scope.scratch.hmem.sliceBytes(0, size).asByteBuffer()
                                    )
                            )
                    )

                    // Release scratch memory and release semaphore.
                    .onComplete((r, e) -> {
                        logger.trace("finish correlation {}", task);
                        if (scope.scratch != null) outputMems.push(scope.scratch);
                        semaphore.release();
                        record.close();
                    });
        }

        private Future<Void> submitH2D(CudaPinned srcHost, CudaMem dstDev, long size) {
            return Future.runAsync(
                    h2dExecutor,
                    () -> {
                        logger.trace("copying {} bytes from host to device", size);
                        profiler.report("bytes transferred host to device", size);

                        profiler.run("h2d", () -> {
                            dstDev.copyFromHostAsync(
                                    srcHost.asPointer(),
                                    size,
                                    h2dStream);

                            h2dStream.synchronize();
                        });

                        return null;
                    }
            );
        }

        private Future<Void> submitD2H(CudaMem srcDev, CudaPinned dstHost, long size) {
            return Future.runAsync(
                    d2hExecutor,
                    () -> {
                        logger.trace("copying {} bytes from device to host", size);
                        profiler.report("bytes transferred device to host", size);

                        profiler.run("d2h", () -> {
                            srcDev.copyToHostAsync(
                                    dstHost.asPointer(),
                                    size,
                                    d2hStream);

                            d2hStream.synchronize();
                        });

                        return null;
                    }
            );
        }

        private <T> Future<T> submitDeviceTask(long priority, String name, Supplier<T> task) {
            Future<T> fut = new Future<>();
            Runnable wrapper = () -> {
                context.with(() -> {
                    try {
                        T output = profiler.run(name, task);
                        fut.complete(output);
                    } catch (Exception e) {
                        fut.completeExceptionally(e);
                    }
                });
            };

            gpuExecutor.execute(new PriorityRunnable(priority, wrapper));
            return fut;
        }

        public void cleanup() {
            this.h2dStream.destroy();
            this.d2hStream.destroy();
            this.h2dExecutor.shutdown();
            this.d2hExecutor.shutdown();
            this.gpuExecutor.shutdown();
            this.devCache.cleanup();

            ScratchMem m;
            while ((m = outputMems.popBlocking()) != null) {
                m.free();
            }

            while ((m = inputMems.popBlocking()) != null) {
                m.free();
            }
        }
    }

    public static class HostWorker {
        protected static final Logger logger = LogManager.getLogger();

        private FileSystem fs;
        private Profiler profiler;
        private Optional<FileCache> fileCache;
        private Optional<DistributedCache> distributedCache;
        private Optional<HostCache> hostCache;
        private ThreadPoolExecutor localIoExecutor;
        private ThreadPoolExecutor remoteIoExecutor;
        private ThreadPoolExecutor generalExecutor;
        private Set<HashableKey> keysSeen;

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

        public interface CacheHitCallback {
            public Future<Void> call(CudaPinned src);
        }

        public interface CacheMissCallback {
            public Future<Long> call(CudaPinned dst);
        }

        private Future<Void> loadInput(
                HashableKey key,
                HostCache.Entry hostMem,
                CacheMissCallback miss
        ) {
            // Call miss
            return miss.call(hostMem.write())

                    // Complete host cache entry
                    .thenRun(hostMem::downgradeToReader)

                    .andVoid();
        }

        private Future<Void> loadInputFromDistributedCacheAsync(
                HashableKey key,
                HostCache.Entry hostMem,
                CacheHitCallback hit,
                CacheMissCallback miss
        ) {
            if (!distributedCache.isPresent()) {
                return loadInput(key, hostMem, miss);
            }

            logger.trace("loading input from distributed cache {}", key);
            ByteBuffer buffer = hostMem.write().asByteBuffer();
            Future<Optional<Long>> fut = distributedCache.get().request(key, buffer);

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
                HashableKey f,
                HostCache.Entry hostMem,
                CacheHitCallback hit,
                CacheMissCallback miss
        ) {
            if (fileCache.isEmpty()) {
                return loadInputFromDistributedCacheAsync(f, hostMem, hit, miss);
            }

            // Acquire slot in file cache
            Future<FileCache.Entry> fut = fileCache.get().acquireAsync(f);

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
                HashableKey key,
                CacheHitCallback hit,
                CacheMissCallback miss
        ) {
            if (keysSeen.add(key)) {
                profiler.report("first-time cache requests");
            }

            Future<HostCache.Entry> entryFut = hostCache.get().acquireAsync(key);

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
}
