package nl.esciencecenter.rocket.profiling;

import org.apache.logging.log4j.LogManager;
import org.json.JSONObject;

import java.util.function.Supplier;

import static java.lang.Thread.currentThread;

public interface Profiler {
    public interface Record extends AutoCloseable {
        public void close();
    }

    public Record trace(String actor, String operation);
    public void report(String actor, String operation, long amount);
    public Profiler.Record traceCorrelation(String actor, String left, String right);

    default public void report(String action) {
        report(action, 1);
    }

    default public void report(String actor, String operation) {
        report(actor, operation, 1);
    }

    default public void report(String action, long data) {
        report(currentThread().getName(), action, data);
    }

    default public <T> T run(String actor, String operation, Supplier<T> fn) {
        try (Record r = trace(actor, operation)) {
            return fn.get();
        }
    }

    default public void run(String actor, String operation, Runnable fn) {
        try (Record r = trace(actor, operation)) {
            fn.run();
        }
    }

    default public Record trace(String operation) {
        return trace(currentThread().getName(), operation);
    }

    default public <T> T run(String operation, Supplier<T> fn) {
        return run(currentThread().getName(), operation, fn);
    }

    default public void run(String operation, Runnable fn) {
        run(currentThread().getName(), operation, fn);
    }

    void shutdown();
}


