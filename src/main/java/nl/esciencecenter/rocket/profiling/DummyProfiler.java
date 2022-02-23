package nl.esciencecenter.rocket.profiling;

import nl.esciencecenter.rocket.types.LeafTask;

import java.util.function.Supplier;

public class DummyProfiler implements Profiler {
    public static final Profiler.Record DUMMY_RECORD = new Profiler.Record() {
        @Override
        public void close() {

        }
    };

    @Override
    public Profiler.Record traceCorrelation(String actor, LeafTask task) {
        return DUMMY_RECORD;
    }

    @Override
    public Record trace(String actor, String operation) {
        return DUMMY_RECORD;
    }

    @Override
    public Record trace(String operation) {
        return DUMMY_RECORD;
    }

    @Override
    public <T> T run(String operation, Supplier<T> fn) {
        return fn.get();
    }

    @Override
    public <T> T run(String actor, String operation, Supplier<T> fn) {
        return fn.get();
    }

    @Override
    public void run(String operation, Runnable fn) {
        fn.run();
    }

    @Override
    public void run(String actor, String operation, Runnable fn) {
        fn.run();
    }

    @Override
    public void report(String actor, String operation, long amount) {
        //
    }

    @Override
    public void report(String action, long amount) {
        //
    }

    @Override
    public void report(String action) {
        //
    }

    @Override
    public void shutdown() {
        //
    }
}