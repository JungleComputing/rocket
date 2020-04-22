/*
* Copyright 2015 Netherlands eScience Center, VU University Amsterdam, and Netherlands Forensic Institute
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance withSupplier the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
package nl.esciencecenter.rocket;

import ibis.constellation.Activity;
import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Constellation;
import ibis.constellation.ConstellationConfiguration;
import ibis.constellation.ConstellationCreationException;
import ibis.constellation.ConstellationFactory;
import ibis.constellation.ConstellationProperties;
import ibis.constellation.Context;
import ibis.constellation.NoSuitableExecutorException;
import ibis.constellation.OrContext;
import ibis.constellation.StealPool;
import ibis.constellation.StealStrategy;
import nl.esciencecenter.rocket.activities.Communicator;
import nl.esciencecenter.rocket.activities.CorrelationsCollectActivity;
import nl.esciencecenter.rocket.activities.CorrelationsLeafActivity;
import nl.esciencecenter.rocket.activities.CorrelationsMatrixActivity;
import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaDevice;
import nl.esciencecenter.rocket.cubaapi.CudaMemByte;
import nl.esciencecenter.rocket.cubaapi.CudaPinned;
import nl.esciencecenter.rocket.indexspace.HilbertCurveIndexSpace;
import nl.esciencecenter.rocket.indexspace.IndexSpace;
import nl.esciencecenter.rocket.indexspace.IndexSpaceDivision;
import nl.esciencecenter.rocket.indexspace.TilingIndexSpace;
import nl.esciencecenter.rocket.profiling.AggregateProfiler;
import nl.esciencecenter.rocket.profiling.DummyProfiler;
import nl.esciencecenter.rocket.profiling.FilterProfiler;
import nl.esciencecenter.rocket.profiling.Profiler;
import nl.esciencecenter.rocket.profiling.MasterProfiler;
import nl.esciencecenter.rocket.cache.DeviceCache;
import nl.esciencecenter.rocket.scheduler.DeviceWorker;
import nl.esciencecenter.rocket.cache.DistributedCache;
import nl.esciencecenter.rocket.cache.FileCache;
import nl.esciencecenter.rocket.cache.HostCache;
import nl.esciencecenter.rocket.scheduler.HostWorker;
import nl.esciencecenter.rocket.scheduler.ApplicationContext;
import nl.esciencecenter.rocket.util.CorrelationList;
import nl.esciencecenter.rocket.util.NodeInfo;
import nl.esciencecenter.rocket.util.Tuple;
import nl.esciencecenter.rocket.util.Util;
import nl.esciencecenter.xenon.filesystems.FileSystem;
import nl.esciencecenter.xenon.filesystems.Path;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.io.InputStream;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.nio.ByteBuffer;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.function.Function;

//stuff for output

public class RocketLauncher<K, R> {
    protected static final Logger logger = LogManager.getLogger();

    static public interface ApplicationFactory<K, R> {
        public ApplicationContext<K, R> create(CudaContext ctx) throws Throwable;
    }

    private String hostName;
    private ApplicationFactory<K, R> factory;
    private FileSystem fs;
    RocketLauncherArgs args;

    public RocketLauncher(RocketLauncherArgs args, FileSystem fs, ApplicationFactory<K, R> factory) {
        this.factory = factory;
        this.fs = fs;
        this.args = args;

        if (args.distributedCache && args.hostCacheSize == 0) {
            throw new IllegalArgumentException("distributed cache can not be used if the host cache is disabled");
        }

        try {
            this.hostName = InetAddress.getLocalHost().getHostName();
        } catch (IOException e) {
            logger.warn("failed to determine host name:", e);
            this.hostName = "<unknown>";
        }
    }

    private double benchmark(Runnable f) {
        long before = System.nanoTime();
        f.run();
        long after = System.nanoTime();

        return (after - before) * 1e-9;
    }

    private void runBenchmark(NodeInfo.DeviceInfo info, CudaContext context, ApplicationContext fun, List<Tuple<K, K>> corrs) {
        long bufferSize = fun.getMaxFileSize();
        CudaPinned scratchHost = context.allocHostBytes(bufferSize);
        CudaMemByte scratchDev = context.allocBytes(bufferSize);

        bufferSize = fun.getMaxInputSize();
        CudaMemByte patternLeft = context.allocBytes(bufferSize);
        CudaMemByte patternRight = context.allocBytes(bufferSize);

        bufferSize = fun.getMaxOutputSize();
        CudaMemByte result = context.allocBytes(fun.getMaxOutputSize());

        Random random = new Random();

        info.parsingTime = 0;
        info.loadingTime = 0;
        info.correlationTime = 0;
        info.preprocessingTime = 0;

        try {
            logger.info("performing calibration benchmark ({})", context.getDevice().getName());

            Function<K, ByteBuffer[]> loadInput = key -> {
                try {
                    List<ByteBuffer> inputs = new ArrayList<>();

                    for (Path path: fun.getInputFiles(key)) {
                        try (InputStream s = Util.readFile(fs, path)) {
                            inputs.add(ByteBuffer.wrap(s.readAllBytes()));
                        }
                    }

                    return inputs.toArray(new ByteBuffer[0]);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            };

            int runs = 0;
            long start = System.nanoTime();
            long end = start;

            // run for minimum of 5 seconds and minimum of 5 runs to get good averages.
            while (end - start < 5_000_000_000L || runs < 5) {
                runs++;
                end = System.nanoTime();


                K leftKey = corrs.get(runs % corrs.size()).getFirst();
                K rightKey = corrs.get(runs % corrs.size()).getSecond();
                ByteBuffer[][] inputs = new ByteBuffer[2][];
                long[] sizes = new long[2];

                context.with(() -> {
                    info.loadingTime += benchmark(() -> inputs[0] = loadInput.apply(leftKey));
                    info.parsingTime += benchmark(() -> sizes[0] = fun.parseFiles(leftKey, inputs[0], scratchHost.asByteBuffer()));
                    scratchHost.copyToDevice(scratchDev);
                    info.preprocessingTime += benchmark(() -> sizes[0] = fun.preprocessInputGPU(
                            leftKey,
                            scratchDev.slice(0, sizes[0]),
                            patternLeft));

                    info.loadingTime += benchmark(() -> inputs[1] = loadInput.apply(rightKey));
                    info.parsingTime += benchmark(() -> sizes[1] = fun.parseFiles(rightKey, inputs[1], scratchHost.asByteBuffer()));
                    scratchHost.copyToDevice(scratchDev);
                    info.preprocessingTime += benchmark(() -> sizes[1] = fun.preprocessInputGPU(
                            rightKey,
                            scratchDev.slice(0, sizes[1]),
                            patternRight));

                    info.correlationTime += benchmark(() -> fun.correlateGPU(
                            leftKey, patternLeft.slice(0, sizes[0]),
                            rightKey, patternRight.slice(0, sizes[1]),
                            result));
                });
            }

            info.loadingTime /= 2 * runs;
            info.preprocessingTime /= 2 * runs;
            info.correlationTime /= runs;

        } finally {
            result.free();
            patternLeft.free();
            patternRight.free();
            scratchHost.free();
            scratchDev.free();
        }
    }

    private List<Tuple<K,K>> extractCorrelations(IndexSpace<K> indexSpace) {
        IndexSpaceDivision<K> ctx = new IndexSpaceDivision<>();
        indexSpace.divide(ctx);

        List<Tuple<K, K>> corr = ctx.getEntries();
        List<IndexSpace<K>> children = ctx.getChildren();

        for (IndexSpace<K> t: children) {
            corr.addAll(extractCorrelations(t));
        }

        return corr;
    }

    private IndexSpace<K> createIndexSpace(K[] keys, boolean includeDiagonal) {
        int tileSize = args.minimumTileSize;

        if (args.tileScheduling) {
            return new TilingIndexSpace<>(tileSize, keys, includeDiagonal);
        } else {
            return new HilbertCurveIndexSpace<>(keys, tileSize * tileSize, includeDiagonal);
        }
    }

    private Profiler createProfiler(RocketLauncherArgs args, String hostName, Communicator communicator) throws IOException, NoSuitableExecutorException {
        boolean enable = (args.profileEventsAll
                || args.profileEventsCache
                || args.profileEventsAggregate
                || args.profileTraceAggregate
                || args.profileCorrelations
                || args.profileTrace);
        String file = args.profileFile;

        if (!enable || file.isBlank()) {
            if (enable) {
                logger.warn("profiling was requested but no target file was given so profiling is disabled");
            }

            if (!file.isBlank()) {
                logger.warn("profile file was given but no targets where given so profiling is disabled");
            }

            return new DummyProfiler();
        }

        Profiler target = new MasterProfiler(hostName, communicator, file);
        Profiler p = target;

        p = new FilterProfiler(
                p,
                args.profileTrace,
                args.profileEventsAll,
                args.profileEventsCache,
                args.profileCorrelations);

        if (args.profileEventsAggregate || args.profileTraceAggregate) {
            p = new AggregateProfiler(p, target, args.profileEventsAggregate, args.profileTraceAggregate);
        }

        return p;
    }

    private Communicator prepareConstellation(int numDevices, ConstellationProperties props)
            throws ConstellationCreationException, NoSuitableExecutorException, InterruptedException {

        ArrayList<ConstellationConfiguration> configs = new ArrayList<>();
        configs.addAll(CorrelationsCollectActivity.getConfigurations());
        configs.addAll(Communicator.getConfigurations());
        configs.addAll(MasterProfiler.getConfigurations());

        for (int i = 0; i < numDevices; i++) {
            configs.add(
                    new ConstellationConfiguration(
                            new OrContext(
                                new Context(CorrelationsMatrixActivity.LABEL),
                                new Context(CorrelationsLeafActivity.LABEL)
                            ),
                            StealPool.WORLD,
                            StealPool.WORLD,
                            StealStrategy.BIGGEST, // Always take the highest level (i.e., depth in the tree) locally
                            StealStrategy.SMALLEST, // Always steal lowest level remotely
                            StealStrategy.SMALLEST
                    )
            );
        }

        logger.info("launching constellation");

        Constellation c = ConstellationFactory.createConstellation(
                props,
                configs.toArray(new ConstellationConfiguration[0]));
        c.activate();

        return new Communicator(c, props);
    }

    private CorrelationList<K, R> computeCorrelations(Communicator c, IndexSpace<K> root) throws NoSuitableExecutorException {
        int total = root.size();

        // Launch collector on master
        CorrelationsCollectActivity collector = new CorrelationsCollectActivity(total);
        ActivityIdentifier collectorId = c.submit(collector);

        // Launch initial matrix activity
        Activity activity = new CorrelationsMatrixActivity<K, R>(collectorId, root, 0);
        c.submit(activity);

        return collector.waitUntilDone();
    }

    public CorrelationList<K, R> run(K[] keys, boolean includeDiagonal) throws Exception {
        printArguments();

        // Prepare CUDA
        CudaDevice[] devices = parseDeviceList();
        int n = devices.length;
        CudaContext[] contexts = new CudaContext[n];
        ApplicationContext<K, R>[] funs = new ApplicationContext[n];
        DeviceWorker<K, R>[] workers = new DeviceWorker[n];
        HostWorker<K>  hworker = null;
        HostCache hcache = null;

        Optional<FileCache> fcache = Optional.empty();
        if (!args.fileCacheDirectory.isEmpty()) {
            fcache = Optional.of(new FileCache(
                    Paths.get(args.fileCacheDirectory),
                    args.fileCacheSize));
        }
        // Create distributed cache
        logger.info("preparing distributed cache");
        Optional<DistributedCache> ccache = args.distributedCache ?
                Optional.of(new DistributedCache(hostName, 3)) :
                Optional.empty();

        // Launch constellation
        ConstellationProperties props = new ConstellationProperties();
        logger.info("preparing constellation");
        Communicator comm = prepareConstellation(n, props);

        // Activate profiler
        logger.info("preparing profiler");
        Profiler profiler = createProfiler(args, hostName, comm);

        if (n == 0) {
            throw new IllegalStateException("no CUDA capable devices detected");
        }

        for (int i = 0; i < n; i++) {
            logger.info("found device {}", devices[i].getName());

            logger.info("creating CUDA context");
            CudaContext ctx = devices[i].createContext();
            contexts[i] = ctx;

            logger.info("creating application context");
            funs[i] = ctx.withSupplier(() -> {
                try {
                    return factory.create(ctx);
                } catch (RuntimeException e) {
                    throw e;
                } catch (Throwable e) {
                    throw new RuntimeException(e);
                }
            });

            long entrySize = funs[i].getMaxInputSize();
            long maxCacheSize = 5000 * entrySize;

            if (i == 0) {
                if (args.hostCacheSize > 0) {
                    logger.info("creating host cache: {} MB", args.hostCacheSize / 1024.0 / 1024.0);
                    hcache = new HostCache(
                            contexts[i],
                            Math.min(args.hostCacheSize, maxCacheSize),
                            entrySize);
                }

                hworker = new HostWorker<K>(
                        fs,
                        profiler,
                        Optional.ofNullable(hcache),
                        fcache,
                        ccache,
                        args.numHostThreads);
            }

            long devCacheSize = args.devCacheSize > 0 ?
                                args.devCacheSize :
                                contexts[i].getFreeMemory() + args.devCacheSize; // Leave deviceCacheSize wiggle room

            logger.info("creating device cache: {} MB", devCacheSize / 1024.0 / 1024.0);
            DeviceCache dcache = new DeviceCache(
                    contexts[i],
                    Math.min(devCacheSize, maxCacheSize),
                    entrySize);

            logger.info("creating worker");
            workers[i] = new DeviceWorker<K, R>(
                    hworker,
                    profiler,
                    contexts[i],
                    funs[i],
                    args.concurrentInputs,
                    args.concurrentJobs,
                    args.concurrentJobs,
                    dcache);
        }

        try {
            if (ccache.isPresent()) {
                ccache.get().activate(hcache, profiler);
            }

            logger.trace("registering workers");
            for (int i = 0; i < n; i++) {
                CorrelationsLeafActivity.registerWorker(comm, workers[i]);
            }

            IndexSpace<K> root = null;
            CorrelationList<K, R> result = null;

            if (comm.isMaster()) {
                root = createIndexSpace(keys, includeDiagonal);
            }

            // Print benchmark results
            printInfos(root, contexts, funs, comm);

            // Launch application!
            if (comm.isMaster()) {
                long start = System.nanoTime();
                result = computeCorrelations(comm, root);
                long end = System.nanoTime();

                logger.info("computing similarity scores took {} seconds.", (end - start) / 1e9);
            }

            // Wait until all other nodes have reached this point.
            comm.barrier();

            return result;
        } finally {
            // Write profilers result
            profiler.shutdown();

            // Close communicator
            comm.shutdown();

            // Destroy all workers
            for (DeviceWorker w: workers) {
                w.cleanup();
            }
            hworker.cleanup();

            // Destroy all contexts
            for (CudaContext ctx: contexts) {
                ctx.destroy();
            }
        }
    }

    /**
     * Print the arguments from LauncherArgs. This is useful for debugging to determine if the parameters were passed
     * correctly.
     */
    private void printArguments() {
        logger.info("Parameters");

        logger.info(" - general");
        logger.info(" -- host threads: {}", args.numHostThreads);
        logger.info(" -- concurrent jobs: {}", args.concurrentJobs);
        logger.info(" -- scheduling: {}", args.tileScheduling ? "tiles" : "divide-and-conquer work-stealing");
        logger.info(" -- minimum tile-size: {} x {}", args.minimumTileSize, args.minimumTileSize);

        logger.info(" - cache");
        logger.info(" -- device cache size: {}", args.devCacheSize);
        logger.info(" -- host cache size: {}", args.hostCacheSize);
        logger.info(" -- distributed cache: {}", args.distributedCache ? "enabled" : "disabled");
        logger.info(" -- file cache: {}", !args.fileCacheDirectory.isBlank());

        boolean enabled = args.profileCorrelations ||
                args.profileTraceAggregate ||
                args.profileTrace ||
                args.profileEventsCache ||
                args.profileEventsAll ||
                args.profileEventsAggregate ||
                !args.profileFile.isBlank();
        logger.info(" - profiling: {}", enabled);

        if (enabled) {
            logger.info(" -- profile file: {}", args.profileFile);
            logger.info(" -- profile correlations: {}", args.profileCorrelations);
            logger.info(" -- profile events: {}", args.profileEventsAll ? "true" : args.profileEventsCache ? "only cache" : "false");
            logger.info(" -- profile tasks: {}", args.profileTrace);
            logger.info(" -- aggregate tasks: {}", args.profileTraceAggregate);
            logger.info(" -- aggregate events: {}", args.profileEventsAggregate);
        }

        ConstellationProperties props = new ConstellationProperties();
        logger.info(" - constellation");
        logger.info(" -- ibis server: {}:{}",
                props.getProperty("ibis.server.address", "?"),
                props.getProperty("ibis.server.port", "?"));
        logger.info(" -- ibis implementation: {}", props.getProperty("ibis.implementation", ""));
        logger.info(" -- pool size: {}", props.getProperty("ibis.pool.size", ""));
        logger.info(" -- pool name: {}", props.getProperty("ibis.pool.name", ""));
    }

    private CudaDevice[] parseDeviceList() {
        String[] list = args.deviceList.split(",");
        List<Integer> ordinals = new ArrayList<>();
        CudaDevice[] allDevices = CudaDevice.getDevices();

        for (String part: list) {
            if (!part.isBlank()) {
                int ordinal;

                try {
                    ordinal = Integer.parseInt(part.strip());
                } catch (NumberFormatException e) {
                    throw new RuntimeException("failed to parse " + part + " as CUDA device ordinal.");
                }

                if (ordinal < 0 || ordinal >= allDevices.length) {
                    throw new RuntimeException("ordinal " + ordinal + " is invalid for node having " + allDevices.length
                            + " CUDA devices");
                }

                if (ordinals.contains(ordinal)) {
                    throw new RuntimeException("ordinal " + ordinal + " is given multiple times");
                }

                ordinals.add(ordinal);
            }
        }

        if (ordinals.isEmpty()) {
            return allDevices;
        }

        CudaDevice[] devices = new CudaDevice[ordinals.size()];

        for (int i = 0; i < ordinals.size(); i++) {
            devices[i] = allDevices[ordinals.get(i)];
        }

        return devices;
    }

    private void printInfos(
            IndexSpace<K> root,
            CudaContext[] contexts,
            ApplicationContext<K, R>[] funs,
            Communicator comm
    ) throws UnknownHostException, NoSuitableExecutorException {
        int n = contexts.length;
        boolean performBenchmark = args.benchmark;
        NodeInfo.DeviceInfo[] devInfos = new NodeInfo.DeviceInfo[n];

        for (int i = 0; i < n; i++) {
            devInfos[i] = new NodeInfo.DeviceInfo(contexts[i]);
        }

        if (performBenchmark) {
            final List<Tuple<K, K>> corrs;

            if (comm.isMaster()) {
                List<Tuple<K, K>> c = extractCorrelations(root);
                Collections.shuffle(c);
                corrs = new ArrayList<>(c.subList(0, Math.min(1000, c.size())));

                comm.broadcast(corrs);
            } else {
                corrs = comm.broadcast(null);
            }

            for (int i = 0; i < n; i++) {
                runBenchmark(devInfos[i], contexts[i], funs[i], corrs);
            }
        }

        // Exchange info
        NodeInfo info = new NodeInfo();
        info.name = hostName;
        info.constellationIdentifier = comm.identifier();
        info.devices = devInfos;

        NodeInfo[] nodeInfos = comm.gather(info);

        // Master prints results
        if (comm.isMaster()) {
            logger.info("launching applications, {} node{} active",
                    nodeInfos.length,
                    nodeInfos.length == 1 ? "" : "s");

            for (int i = 0; i < nodeInfos.length; i++) {
                NodeInfo nodeInfo = nodeInfos[i];

                for (int j = 0; j < nodeInfo.devices.length; j++) {
                    NodeInfo.DeviceInfo devInfo = nodeInfo.devices[j];

                    logger.info(" - node: {} ({}), device: {} ({} MB)",
                            nodeInfo.name, nodeInfo.constellationIdentifier,
                            devInfo.name, devInfo.memorySize / 1000000.0);

                    if (performBenchmark) {
                        logger.info("   - loading: {} sec (throughput: {})",
                                devInfo.loadingTime, 1.0 / devInfo.loadingTime);
                        logger.info("   - parsing: {} sec (throughput: {})",
                                devInfo.parsingTime, 1.0 / devInfo.parsingTime);
                        logger.info("   - preprocessing: {} sec (throughput: {})",
                                devInfo.preprocessingTime, 1.0 / devInfo.preprocessingTime);
                        logger.info("   - correlation: {} sec (throughput: {})",
                                devInfo.correlationTime, 1.0 / devInfo.correlationTime);

                    }
                }
            }
        }
    }
}
