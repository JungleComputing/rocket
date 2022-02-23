package nl.esciencecenter.rocket.activities;

import ibis.constellation.Activity;
import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Constellation;
import ibis.constellation.ConstellationConfiguration;
import ibis.constellation.Context;
import ibis.constellation.Event;
import ibis.constellation.NoSuitableExecutorException;
import ibis.constellation.StealPool;
import ibis.constellation.StealStrategy;
import ibis.constellation.util.SimpleActivity;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ResultsCollectActivity<R extends Comparable<? super R>> extends Activity {
    protected static final Logger logger = LogManager.getLogger();

    final private static String LABEL = "ResultsCollectActivity";
    final private int PERIOD = 5000; // 5 second

    static public List<ConstellationConfiguration> getConfigurations() {
        return Collections.singletonList(
                new ConstellationConfiguration(
                    new Context(ResultsCollectActivity.LABEL),
                    StealPool.NONE,
                    StealPool.NONE,
                    StealStrategy.SMALLEST,
                    StealStrategy.SMALLEST,
                    StealStrategy.SMALLEST)
        );
    }

    final static private List<?> PENDING_SUBMISSION = new ArrayList<>();

    static public void sendToMaster(Communicator comm, ActivityIdentifier id, Object corr) {
        boolean isFirst;

        // Add entry to PENDING_SUBMISSION
        synchronized (PENDING_SUBMISSION) {
            isFirst = PENDING_SUBMISSION.isEmpty();
            ((List<Object>) PENDING_SUBMISSION).add(corr);
        }

        // If list was empty, launch an activity to submit the correlation at some moment in the future.
        if (isFirst) {
            try {
                comm.submit(new SimpleActivity(id, new Context(LABEL), false) {
                    @Override
                    public void simpleActivity(Constellation c) {
                        try {
                            Thread.sleep(1000 + (long)(Math.random() * 1000));
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }

                        List<?> correlations;
                        synchronized (PENDING_SUBMISSION) {
                            correlations = new ArrayList<>(PENDING_SUBMISSION);
                            PENDING_SUBMISSION.clear();
                        }

                        c.send(new Event(identifier(), getParent(), correlations));
                    }
                });
            } catch (NoSuitableExecutorException e) {
                throw new RuntimeException(e);
            }
        }
    }

    final private List<R> results;
    private int prevProgress;
    private long startTime;
    private long prevTime;

    public ResultsCollectActivity() {
        super(new Context(LABEL), false, true);
        this.results = new ArrayList<>();
    }

    public List<R> waitUntilDone(int totalResults) {
        long sleep = 100;

        while (true) {
            // Break if done
            synchronized (this) {
                if (results.size() == totalResults) {
                    Collections.sort(results);
                    return new ArrayList<>(results);
                }
            }

            // Print progress if more than PERIOD seconds have past.
            printProgress(totalResults);

            // Sleep for a bit.
            try {
                Thread.sleep(sleep);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
    }

    static private String formatMillis(long millis) {
        long hours = millis / 3600000;
        long minutes = (millis % 3600000) / 60000;
        double milliseconds = (millis % 60000) / 1000.0;

        return String.format("%02d:%02d:%05.2f", hours, minutes, milliseconds);
    }

    private void printProgress(int totalResults) {
        double ratio, throughput;
        long elapsed;

        synchronized (this) {
            int progress = this.results.size();
            int prevProgress = this.prevProgress;
            long nowTime = System.currentTimeMillis();
            long startTime = this.startTime;
            long prevTime = this.prevTime;

            if (nowTime - prevTime < PERIOD) {
                return;
            }

            ratio = Math.min(1, progress / (double) totalResults);
            elapsed = nowTime - startTime;
            throughput = (progress - prevProgress) / ((double)(nowTime - prevTime) / 1000.0);

            this.prevProgress = progress;
            this.prevTime = nowTime;
        }


        String progressString = String.format("%.2f%%", ratio * 100.0);
        String elapsedString = formatMillis(elapsed);
        String remainingString = ratio > 0 ? formatMillis((long)(elapsed  * (1 - ratio) / ratio)) : "inf";
        String throughputString = String.format("%.2f", throughput);

        logger.info("progress: {}, elapsed: {}, remaining: {}, throughput: {} items/sec",
                progressString, elapsedString, remainingString, throughputString);
    }

    @Override
    public int initialize(Constellation constellation) {
        synchronized (this) {
            startTime = prevTime = System.currentTimeMillis();
        }

        logger.info("initializing timer");
        return SUSPEND;
    }

    @Override
    public int process(Constellation constellation, Event event) {
        synchronized (this) {
            @SuppressWarnings("unchecked")
            Iterable<R> data = (Iterable<R>) event.getData();

            for (R row: data) {
                results.add(row);
            }

            printProgress(1);
            return SUSPEND;
        }
    }

    @Override
    public void cleanup(Constellation constellation) {
        //
    }
}
