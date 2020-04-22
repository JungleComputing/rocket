package nl.esciencecenter.rocket.profiling;


import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import static nl.esciencecenter.rocket.profiling.DummyProfiler.DUMMY_RECORD;

/**
 * Filters certain items to be written to the profile file.
 */
public class FilterProfiler implements Profiler {
    protected static final Logger logger = LogManager.getLogger();

    private final Profiler parent;
    private final boolean enableTrace;
    private final boolean enableEvents;
    private final boolean enableCacheEvents;
    private final boolean enableCorrelations;

    public FilterProfiler(Profiler parent, boolean enableTrace, boolean enableEvents, boolean enableCacheEvents, boolean enableCorrelations) {
        this.parent = parent;
        this.enableTrace = enableTrace;
        this.enableEvents = enableEvents;
        this.enableCorrelations = enableCorrelations;
        this.enableCacheEvents = enableCacheEvents;
    }

    @Override
    public Record trace(String actor, String operation) {
        return enableTrace ? parent.trace(actor, operation) : DUMMY_RECORD;
    }

    @Override
    public void report(String actor, String operation, long amount) {
        if (enableEvents || (enableCacheEvents && operation.contains("cache"))) {
            parent.report(actor, operation, amount);
        }
    }

    @Override
    public Record traceCorrelation(String actor, String left, String right) {
        return enableCorrelations ? parent.traceCorrelation(actor, left, right) : DUMMY_RECORD;
    }

    @Override
    public void shutdown() {
        parent.shutdown();
    }
}
