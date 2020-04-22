package nl.esciencecenter.rocket.profiling;

import ibis.constellation.Activity;
import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Constellation;
import ibis.constellation.ConstellationConfiguration;
import ibis.constellation.Context;
import ibis.constellation.Event;
import ibis.constellation.NoSuitableExecutorException;
import ibis.constellation.StealPool;
import ibis.constellation.StealStrategy;
import ibis.constellation.util.MemorySizes;
import nl.esciencecenter.rocket.activities.Communicator;
import nl.esciencecenter.rocket.util.Tuple;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorOutputStream;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.BufferedOutputStream;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.io.Writer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.zip.Deflater;
import java.util.zip.GZIPOutputStream;

import static nl.esciencecenter.rocket.profiling.DummyProfiler.DUMMY_RECORD;

public class MasterProfiler implements Profiler {
    protected static final Logger logger = LogManager.getLogger();

    public static List<ConstellationConfiguration> getConfigurations() {
        return Collections.singletonList(new ConstellationConfiguration(
                new Context(ProfilerActivity.LABEL),
                StealPool.NONE,
                StealPool.NONE,
                StealStrategy.SMALLEST,
                StealStrategy.SMALLEST,
                StealStrategy.SMALLEST
        ));
    }

    static private class ShutdownEvent implements Serializable {
        private static final long serialVersionUID = 3854521298930028204L;
    }

    static private class RecordsEvent implements Serializable {
        private static final long serialVersionUID = -44450542720559224L;
        private byte[] recordsAsString;

        public RecordsEvent(List<String> records) {
            StringBuilder result = new StringBuilder();

            for (String record: records) {
                result.append(record);
                result.append('\n');
            }

            this.recordsAsString = result.toString().getBytes(StandardCharsets.US_ASCII);
        }
    }

    static private abstract class ProfilerActivity extends Activity {
        final static public String LABEL = ProfilerActivity.class.getCanonicalName();
        private boolean done;

        public ProfilerActivity() {
            super(new Context(LABEL), false, true);
            this.done = false;
        }

        @Override
        public int initialize(Constellation constellation) {
            return SUSPEND;
        }

        @Override
        public void cleanup(Constellation constellation) {
            synchronized (this) {
                done = true;
                notifyAll();
            }
        }

        public void waitUntilDone() {
            synchronized (this) {
                while (!done) {
                    try {
                        wait();
                    } catch (InterruptedException e) {
                        //
                    }
                }
            }
        }
    }

    static private class MasterProfilerActivity extends ProfilerActivity {
        private int numWorkers;
        private OutputStream stream;

        public MasterProfilerActivity(OutputStream stream, int numWorkers) {
            super();
            this.stream = new BufferedOutputStream(stream, 1024 * 1024);
            this.numWorkers = numWorkers;
        }


        private void writeRecords(RecordsEvent data) {
            if (stream != null) {
                try {
                    stream.write(data.recordsAsString);
                } catch (IOException e) {
                    logger.warn("An I/O exception occurred while writing profiling data. Further events " +
                            "will be discarded: {}", e.getMessage());
                    stream = null;
                }
            }
        }

        @Override
        public int process(Constellation constellation, Event event) {
            Object data = event.getData();

            if (data instanceof RecordsEvent) {
                writeRecords(((RecordsEvent) data));
            } else if (data instanceof ShutdownEvent) {
                numWorkers--;
            } else {
                logger.warn("could not handle event of type {}", data.getClass().getCanonicalName());
            }

            return numWorkers > 0 ? SUSPEND : FINISH;
        }

        @Override
        public void cleanup(Constellation constellation) {
            super.cleanup(constellation);

            if (stream != null) {
                try {
                    stream.flush();
                    stream.close();
                } catch (IOException e) {
                    logger.warn("An I/O exception occurred while writing profiling data. Some events may " +
                            "have been discarded: {}", e.getMessage());
                }
            }
        }
    }

    static private class WorkerProfilerActivity extends ProfilerActivity {
        ActivityIdentifier parent;

        public WorkerProfilerActivity(ActivityIdentifier parent) {
            super();
            this.parent = parent;
        }

        @Override
        public int process(Constellation constellation, Event event) {
            Object data = event.getData();
            constellation.send(new Event(identifier(), parent, data));

            return (data instanceof ShutdownEvent) ? FINISH : SUSPEND;
        }
    }



    private boolean shutdown;
    private long globalStart;
    private String hostName;
    private Communicator communicator;
    private ArrayBlockingQueue<String> pendingRecords;
    private ProfilerActivity activity;

    public MasterProfiler(String hostName, Communicator c, String profilingFile) throws IOException, NoSuitableExecutorException {
        this.hostName = hostName;
        this.communicator = c;
        this.pendingRecords = new ArrayBlockingQueue<>(1024);

        long millis = communicator.broadcast(System.currentTimeMillis());
        globalStart = System.nanoTime() - (System.currentTimeMillis() - millis) * 1_000_000;

        if (communicator.isMaster()) {
            OutputStream stream = new FileOutputStream(profilingFile);
            String name = profilingFile.toLowerCase();

            if (name.endsWith(".gz")) {
                stream = new GZIPOutputStream(stream, 1024 * 1024);
            }

            if (name.endsWith(".bz2")) {
                stream = new BZip2CompressorOutputStream(stream, 1024 * 1024);
            }

            logger.info("writing trace data to {}", profilingFile);
            activity = new MasterProfilerActivity(
                    stream,
                    communicator.getPoolSize()
            );

            ActivityIdentifier aid = communicator.submit(activity);
            communicator.broadcast(aid);
        } else {
            ActivityIdentifier aid = communicator.broadcast(null);
            activity = new WorkerProfilerActivity(aid);
            communicator.submit(activity);
        }
    }

    @Override
    public Profiler.Record traceCorrelation(String actor, String left, String right) {
        long before = System.nanoTime() - globalStart;

        return () -> {
            long after = System.nanoTime() - globalStart;
            String record = String.format("%s\t%s\t%s\t%d\t%d\t%s\t%s",
                    hostName,
                    actor,
                    "correlation",
                    before, after,
                    left, right);

            appendRecord(record);
        };
    }

    @Override
    public Profiler.Record trace(String actor, String operation) {
        long before = System.nanoTime() - globalStart;

        return () -> {
            long after = System.nanoTime() - globalStart;
            String record = String.format("%s\t%s\t%s\t%d\t%d",
                    hostName, actor, operation,
                    before, after);

            appendRecord(record);
        };
    }

    @Override
    public void report(String actor, String action, long amount) {
        long offset = globalStart;
        long time = System.nanoTime() - offset;

        String record = String.format("%s\t%s\t%s\t%d\t%d\t%d",
                hostName, actor, action,
                time, time,
                amount);

        appendRecord(record);
    }

    private void appendRecord(String record) {
        while (!pendingRecords.offer(record)) {
            flushRecords();
        }
    }

    private void flushRecords() {
        ArrayList<String> records = new ArrayList<>();
        pendingRecords.drainTo(records);

        communicator.send(
                activity.identifier(),
                activity.identifier(),
                new RecordsEvent(records));
    }

    @Override
    public void shutdown() {
        if (!shutdown) {
            communicator.barrier();

            flushRecords();
            communicator.send(
                    activity.identifier(),
                    activity.identifier(),
                    new ShutdownEvent());

            activity.waitUntilDone();
            shutdown = true;
        }
    }
}
