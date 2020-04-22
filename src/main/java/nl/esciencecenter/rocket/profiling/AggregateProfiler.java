package nl.esciencecenter.rocket.profiling;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Aggregates profiling information and writes these aggregates to the profile file at shutdown.
 */
public class AggregateProfiler implements Profiler {
    protected static final Logger logger = LogManager.getLogger();

    private Profiler parent;
    private Profiler sink;
    private AtomicReference<HashMap<String, AtomicLong>> metrics;
    private boolean aggregateEvents;
    private boolean aggregateTasks;

    public AggregateProfiler(Profiler parent, Profiler sink, boolean aggregateEvents, boolean aggregateTasks) {
        this.parent = parent;
        this.sink = sink;
        this.metrics = new AtomicReference<>(new HashMap<>());
        this.aggregateTasks = aggregateTasks;
        this.aggregateEvents = aggregateEvents;
    }

    @Override
    public Record trace(String actor, String operation) {
        Record record = parent.trace(actor, operation);
        if (!aggregateTasks) {
            return record;
        }

        long before = System.nanoTime();
        return () -> {
            long after = System.nanoTime();
            record.close();

            increment(actor, "task:" + operation + ":count", 1);
            increment(actor, "task:" + operation + ":total", after - before);
        };
    }

    @Override
    public Record traceCorrelation(String actor, String left, String right) {
        return parent.traceCorrelation(actor, left, right);
    }

    @Override
    public void report(String actor, String operation, long amount) {
        parent.report(actor, operation, amount);
        if (!aggregateEvents) {
            return;
        }

        increment(actor, operation + ":total", amount);
        increment(actor, operation + ":count", 1);
    }

    private void increment(String actor, String operation, long amount) {
        String key = actor + "\t" + operation;

        while (true) {
            HashMap<String, AtomicLong> map = metrics.get();
            AtomicLong counter = map.get(key);

            if (counter == null) {
                HashMap<String, AtomicLong> copy = new HashMap<>(map);
                copy.put(key, new AtomicLong(0));
                metrics.compareAndSet(map, copy);
            } else {
                counter.addAndGet(amount);
                break;
            }
        }
    }

    @Override
    public void shutdown() {
        HashMap<String, AtomicLong> counters = metrics.get();

        for (Map.Entry<String, AtomicLong> entry: counters.entrySet()) {
            String parts[] = entry.getKey().split("\t");
            String actor = parts[0];
            String operation = parts[1];
            long amount = entry.getValue().getAndSet(0);

            sink.report(actor, operation, amount);
        }


        parent.shutdown();
    }
}
