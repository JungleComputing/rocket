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
import nl.esciencecenter.rocket.util.Correlation;
import nl.esciencecenter.rocket.util.CorrelationList;
import nl.esciencecenter.rocket.util.InternPool;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CorrelationsCollectActivity<K, R> extends Activity {
    protected static final Logger logger = LogManager.getLogger();

    final private static String LABEL = "CorrelationsCollectActivity";
    final private int PERIOD = 5000; // 5 second

    static public List<ConstellationConfiguration> getConfigurations() {
        return Collections.singletonList(
                new ConstellationConfiguration(
                    new Context(CorrelationsCollectActivity.LABEL),
                    StealPool.NONE,
                    StealPool.NONE,
                    StealStrategy.SMALLEST,
                    StealStrategy.SMALLEST,
                    StealStrategy.SMALLEST)
        );
    }

    final static private List<Correlation<?, ?>> PENDING_SUBMISSION = new ArrayList<>();

    static public void sendToMaster(Communicator comm, ActivityIdentifier id, Correlation<?, ?> corr) {
        boolean isFirst;

        // Add entry to PENDING_SUBMISSION
        synchronized (PENDING_SUBMISSION) {
            isFirst = PENDING_SUBMISSION.isEmpty();
            PENDING_SUBMISSION.add(corr);
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

                        List<Correlation<?, ?>> correlations;
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

    final private int total;
    final private InternPool<K> cachedKeys;
    final private List<Correlation<K, R>> results;
    private int prevProgress;
    private long startTime;
    private long prevTime;

    public CorrelationsCollectActivity(int total) {
        super(new Context(LABEL), false, true);
        this.total = total;
        this.cachedKeys = new InternPool<>();
        this.results = new ArrayList<>();
    }

    public CorrelationList<K, R> waitUntilDone() {
        long sleep = 100;
        long lastPrinted = 0;

        while (true) {
            // Break if done
            synchronized (this) {
                if (results.size() == total) {
                    return new CorrelationList<>(results);
                }
            }

            // Print progress if more than PERIOD seconds have past.
            if (lastPrinted > PERIOD) {
                lastPrinted -= PERIOD;
                printProgress();
            }

            // Sleep for a bit.
            try {
                Thread.sleep(sleep);
                lastPrinted += sleep;
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

    private void printProgress() {
        double ratio, throughput;
        long elapsed;

        synchronized (this) {
            int progress = this.results.size();
            int prevProgress = this.prevProgress;
            long nowTime = System.currentTimeMillis();
            long startTime = this.startTime;
            long prevTime = this.prevTime;

            ratio = Math.min(1, progress / (double) total);
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
        return total > 0 ? SUSPEND : FINISH;
    }

    @Override
    public int process(Constellation constellation, Event event) {
        synchronized (this) {
            @SuppressWarnings("unchecked")
            Iterable<Correlation<K, R>> correlations = (Iterable<Correlation<K, R>>) event.getData();

            for (Correlation<K, R> corr: correlations) {
                K left = cachedKeys.intern(corr.getI());
                K right = cachedKeys.intern(corr.getJ());
                R coeff = corr.getCoefficient();

                results.add(new Correlation<>(left, right, coeff));
            }

            return results.size() < total ? SUSPEND : FINISH;
        }
    }

    @Override
    public void cleanup(Constellation constellation) {
        //
    }
}
