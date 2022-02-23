package nl.esciencecenter.rocket;

import com.beust.jcommander.Parameter;

import java.util.Optional;

public class RocketLauncherArgs {
    @Parameter(names="--devices", description="Ordinals of CUDA devices to use (for example \"0,2,3\"). If empty, all available devices are used.", arity=1)
    public String deviceList = "";

    @Parameter(names="--host-cache-size", description="Size of host cache in bytes.", arity=1)
    public long hostCacheSize = 40L * 1024 * 1024 * 1024; // 40GB by default

    @Parameter(names="--device-cache-size", description="Size of device cache in bytes. If negative, allocates total memory minus given size instead.", arity=1)
    public long devCacheSize = -500L * 1024 * 1024;

    @Parameter(names="--host-threads", description="Number of processing threads to use for CPU activities (load inputs/outputs, copying buffers, etc.).", arity=1)
    public int numHostThreads = 4;

    @Parameter(names="--concurrent-jobs", description="Number of concurrent jobs.", arity = 1)
    public int concurrentJobs = 500;

    @Parameter(names="--concurrent-inputs", description="Number of concurrent input jobs.", arity = 1)
    public int concurrentInputs = 8;

    @Parameter(names="--distributed-cache", description="Enable/disable the distributed cache.")
    public boolean distributedCache = false;

    @Parameter(names="--distributed-cache-hops", description="Maximum number of hops to use for distributed cache", arity = 1)
    public int distributedCacheHops = 1;

    @Parameter(names="--file-cache", description="Directory used by file cache for temporary storage. File cache is disabled if empty.", arity = 1)
    public String fileCacheDirectory = "";

    @Parameter(names="--file-cache-size", description="Maximum file cache size in number of active file.", arity = 1)
    public int fileCacheSize = 5000;

    @Parameter(names="--benchmark", description="Perform micro-benchmark to establish performance.")
    public boolean benchmark = false;

    @Parameter(names="--profile-file", description="File where to write trace information.", arity=1)
    public String profileFile = "";

    @Parameter(names="--profile-events-aggregate", description="Trace all events, but only report aggregated results.")
    public boolean profileEventsAggregate = false;

    @Parameter(names="--profile-events", description="Trace all events.")
    public boolean profileEventsAll = false;

    @Parameter(names="--profile-events-cache", description="Trace only cache-related events.")
    public boolean profileEventsCache = false;

    @Parameter(names="--profile-correlations", description="Trace correlations.")
    public boolean profileTasks = false;

    @Parameter(names="--profile-tasks", description="Trace all tasks. This is expensive and will affect performance.")
    public boolean profileTrace = false;

    @Parameter(names="--profile-tasks-aggregate", description="Trace all tasks, but only report aggregate results.")
    public boolean profileTraceAggregate = false;

    @Parameter(names="--tile-size", description="Tile size for minimum unit of work. Tiles larger than the given size cannot be stolen.", arity = 1)
    public int minimumTileSize = 1;

    @Parameter(names="--tile-scheduling", description="Switch to master-worker tile scheduling instead of divide-and-conquer work-stealing.", arity = 0)
    public boolean tileScheduling;
}
