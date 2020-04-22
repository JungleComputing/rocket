package nl.esciencecenter.rocket.activities;

import ibis.constellation.Activity;
import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Constellation;
import ibis.constellation.ConstellationConfiguration;
import ibis.constellation.ConstellationIdentifier;
import ibis.constellation.ConstellationProperties;
import ibis.constellation.Context;
import ibis.constellation.Event;
import ibis.constellation.NoSuitableExecutorException;
import ibis.constellation.StealPool;
import ibis.constellation.StealStrategy;
import ibis.constellation.util.FlexibleEventCollector;
import ibis.constellation.util.SimpleActivity;
import nl.esciencecenter.rocket.indexspace.IndexSpace;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.lang.reflect.Array;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.LinkedBlockingQueue;

public class Communicator {
    protected static final Logger logger = LogManager.getLogger();
    final public static String INIT_LABEL = "Communicator_BroadcastMasterActivity";
    final public static String COLLECT_LABEL = "Communicator_CollectActivity";

    final static private LinkedBlockingQueue<BroadcastMasterActivity> initActivities = new LinkedBlockingQueue<>();

    public static List<ConstellationConfiguration> getConfigurations() {
        return Arrays.asList(
            new ConstellationConfiguration(
                    new Context(Communicator.INIT_LABEL),
                    StealPool.WORLD,
                    StealPool.WORLD,
                    StealStrategy.SMALLEST,
                    StealStrategy.SMALLEST,
                    StealStrategy.SMALLEST),
            new ConstellationConfiguration(
                    new Context(Communicator.COLLECT_LABEL),
                    StealPool.NONE,
                    StealPool.NONE,
                    StealStrategy.SMALLEST,
                    StealStrategy.SMALLEST,
                    StealStrategy.SMALLEST)
        );
    }

    private static class BroadcastMasterActivity extends SimpleActivity {
        private boolean done;

        BroadcastMasterActivity(ActivityIdentifier parent) {
            super(parent, new Context(INIT_LABEL));
        }

        @Override
        synchronized public void simpleActivity(Constellation c) {
            logger.trace("running BroadcastMasterActivity on {}", c.identifier());
            boolean result = initActivities.offer(this);

            // Block this thread forever
            while (!done) {
                try {
                    wait();
                } catch (InterruptedException e) {
                    //
                }
            }
        }

        synchronized public void shutdown() {
            done = true;
            notifyAll();
        }
    }

    private static class CollectActivity extends Activity {
        private static LinkedBlockingQueue<Event> events = new LinkedBlockingQueue<>();

        CollectActivity() {
            super(new Context(COLLECT_LABEL), false, true);
        }

        @Override
        public int initialize(Constellation constellation) {
            return SUSPEND;
        }

        @Override
        public int process(Constellation constellation, Event event) {
            events.offer(event);
            return SUSPEND;
        }

        @Override
        public void cleanup(Constellation constellation) {
            //
        }

        public Event waitForEvent() {
            try {
                return events.take();
            } catch (InterruptedException e) {
                return waitForEvent();
            }
        }
    }

    private static int parsePoolSize(ConstellationProperties props) {
        int poolSize = -1;
        String val = props.getProperty("ibis.pool.size",
                props.getProperty("ibis.constellation.pool.size", ""));

        try {
            poolSize = Integer.parseInt(val);
        } catch (NumberFormatException e) {
            //
        }

        if (poolSize <= 0) {
            throw new IllegalArgumentException("could not determine pool size, found invalid value: " + val);
        }

        return poolSize;
    }

    private int poolSize;
    private Constellation constellation;
    private ActivityIdentifier masterActivityId;
    private CollectActivity collectActivity;
    private BroadcastMasterActivity initializeActivity;

    public Communicator(Constellation c) throws NoSuitableExecutorException, InterruptedException {
        this(c, new ConstellationProperties());
    }

    public Communicator(Constellation c, ConstellationProperties props) throws NoSuitableExecutorException, InterruptedException {
        this(c, parsePoolSize(props));
    }

    public Communicator(Constellation c, int poolSize) throws NoSuitableExecutorException, InterruptedException {
        logger.trace("launching Communicator on {}", c.identifier());
        this.poolSize = poolSize;
        this.constellation = c;
        this.collectActivity = new CollectActivity();
        c.submit(collectActivity);

        if (isMaster()) {
            for (int i = 0; i < poolSize; i++) {
                ActivityIdentifier aid = c.submit(new BroadcastMasterActivity(collectActivity.identifier()));
            }
        }

        this.initializeActivity = initActivities.take();
        this.masterActivityId = initializeActivity.getParent();
    }

    public Constellation getConstellation() {
        return constellation;
    }

    public <T> T[] gather(T value) {
        logger.trace("gather with class {}", value.getClass());

        if (isMaster()) {
            int numPeers = poolSize - 1;
            ActivityIdentifier[] peers = new ActivityIdentifier[numPeers];
            T[] data = (T[]) Array.newInstance(value.getClass(), poolSize);
            int index = 0;

            while (index < numPeers) {
                Event e = collectActivity.waitForEvent();
                peers[index] = e.getSource();
                data[index] = (T) e.getData();
                index++;
            }

            // Last entry is our own value
            data[index] = value;

            logger.trace("sending results to peers: {}", (Object) peers);

            for (int i = 0; i< numPeers; i++) {
                send(masterActivityId, peers[i], data);
            }

            return data;
        } else {
            send(collectActivity.identifier(), masterActivityId, value);
            return (T[]) collectActivity.waitForEvent().getData();
        }
    }

    public <K> K broadcast(K data) {
        if (isMaster()) {
            ActivityIdentifier[] peers = new ActivityIdentifier[poolSize - 1];
            int index = 0;

            while (index < peers.length) {
                Event e = collectActivity.waitForEvent();
                peers[index++] = e.getSource();
            }

            for (int i = 0; i< peers.length; i++) {
                send(masterActivityId, peers[i], data);
            }

            return data;
        } else {
            send(collectActivity.identifier(), masterActivityId, "");
            return (K) collectActivity.waitForEvent().getData();
        }
    }

    public void barrier() {
        gather("");
    }

    public <T> void send(ActivityIdentifier src, ActivityIdentifier dst, T data) {
        constellation.send(new Event(src, dst, data));
    }

    public ActivityIdentifier submit(Activity a) throws NoSuitableExecutorException {
        return constellation.submit(a);
    }

    public ConstellationIdentifier identifier() {
        return constellation.identifier();
    }

    public int getPoolSize() {
        return poolSize;
    }

    public void shutdown() {
        barrier();
        initializeActivity.shutdown();
        constellation.done();
    }

    public boolean isMaster() {
        return constellation.isMaster();
    }
}
