package nl.esciencecenter.rocket.scheduler;

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import nl.esciencecenter.rocket.cache.DeviceCache;
import nl.esciencecenter.rocket.profiling.Profiler;
import nl.esciencecenter.rocket.util.Future;
import nl.esciencecenter.rocket.util.FutureQueue;
import nl.esciencecenter.rocket.util.PriorityTask;
import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaPinned;
import nl.esciencecenter.rocket.cubaapi.CudaMem;
import nl.esciencecenter.rocket.cubaapi.CudaMemByte;
import nl.esciencecenter.rocket.cubaapi.CudaStream;
import nl.esciencecenter.xenon.filesystems.Path;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.ByteBuffer;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

public class DeviceWorker<K, R> {
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
    private HostWorker<K>  hostWorker;
    private Profiler profiler;
    private CudaContext context;
    private ThreadPoolExecutor h2dExecutor;
    private ThreadPoolExecutor d2hExecutor;
    private ThreadPoolExecutor gpuExecutor;

    private ApplicationContext<K, R> applicationContext;

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
            HostWorker<K>  hworker,
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

    private Future<Long> loadInput(K f, CudaMem devMem) {
        logger.trace("load input {}", f);

        Future<ScratchMem> scratch = new Future<>();
        Path[] paths = applicationContext.getInputFiles(f);
        Future<ByteBuffer[]> buffers;

        return (buffers = hostWorker.loadFilesAsync(paths))

                // acquire scratch memory
                .then(__ -> inputMems.popAsync().thenComplete(scratch))

                // Load item
                .then(__ -> hostWorker.submitHostTask(
                        "load_input",
                        () -> applicationContext.parseFiles(f, buffers.get(), scratch.get().hmem.asByteBuffer())
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
                        () -> applicationContext.preprocessInputGPU(
                                f,
                                scratch.get().dmem.slice(0, size),
                                devMem.asBytes())
                ))

                // Always release scratch memory.
                .onComplete((r, e) -> {
                    scratch.thenRun(s -> inputMems.push(s));
                });
    }

    private Future<Long> loadInputFromHostCacheAsync(K f, CudaMem dmem) {
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
                        return loadInput(f, dmem)
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
            return loadInput(f, dmem);
        }
    }

    private Future<DeviceCache.Entry[]> loadInputFromDeviceCacheAsync(K a, K b) {
        logger.trace("load input from device cache ({},{})", a, b);
        return devCache
                .acquireAllAsync(a.toString(), b.toString())
                .then(entries -> {
                    Future<DeviceCache.Entry[]> done = Future.ready(entries);

                    // Iterate over all entries to load them into memory if required.
                    for (DeviceCache.Entry entry: entries) {

                        // Cache miss, attempt to load data from host cache.
                        if (entry.isWriter()) {
                            String key = entry.getKey();
                            K f = key.equals(a.toString()) ? a : b;

                            profiler.report(name, "device cache misses", 1);
                            done = loadInputFromHostCacheAsync(f, entry.write())

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

    public Future<R> submitCorrelation(K ki, K kj) {
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

        logger.trace("start compute correlation {}x{}", ki, kj);
        profiler.report("correlations executed", 1);
        Profiler.Record record = profiler.traceCorrelation(name, ki.toString(), kj.toString());

        // Asynchronously load data from cache
        Future<DeviceCache.Entry[]> fut = loadInputFromDeviceCacheAsync(ki, kj);

        // Wait until both cache entries have been loaded
        return fut

                // Obtain output scratch memory to write results.
                .then(__ -> outputMems.popAsync().thenRun(v -> scope.scratch = v))

                // Submit correlation to GPU
                .then(__ -> {
                    DeviceCache.Entry[] entries = fut.get();
                    String firstKey = entries[0].getKey();
                    DeviceCache.Entry a = firstKey.equals(ki.toString()) ? entries[0] : entries[1];
                    DeviceCache.Entry b = firstKey.equals(kj.toString()) ? entries[0] : entries[1];

                    return submitDeviceTask(
                        priority,
                        "correlate",
                        () -> applicationContext.correlateGPU(
                                ki,
                                a.read().slice(0, a.size()),
                                kj,
                                b.read().slice(0, b.size()),
                                scope.scratch.dmem)
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
                                () -> applicationContext.postprocessOutput(
                                        ki,
                                        kj,
                                        scope.scratch.hmem.sliceBytes(0, size).asByteBuffer()
                                )
                        )
                )

                // Release scratch memory and release semaphore.
                .onComplete((r, e) -> {
                    logger.trace("finish correlation {} x {}", ki, kj);
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

        gpuExecutor.execute(new PriorityTask(priority, wrapper));
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
