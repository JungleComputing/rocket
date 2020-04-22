package nl.esciencecenter.rocket.activities;

import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Constellation;
import ibis.constellation.Context;
import ibis.constellation.util.SimpleActivity;
import nl.esciencecenter.rocket.scheduler.DeviceWorker;
import nl.esciencecenter.rocket.util.Correlation;
import nl.esciencecenter.rocket.util.Tuple;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingDeque;

public class CorrelationsLeafActivity<K, R> extends SimpleActivity {
    protected static final Logger logger = LogManager.getLogger();
    final public static String LABEL = CorrelationsLeafActivity.class.getCanonicalName();

    private ActivityIdentifier collector;
    private K fi;
    private K fj;

    public CorrelationsLeafActivity(ActivityIdentifier collector, ActivityIdentifier parent, K fi, K fj) {
        super(parent, new Context(LABEL, Integer.MAX_VALUE), false);
        this.collector = collector;
        this.fi = fi;
        this.fj = fj;
    }

    static private BlockingQueue<Tuple<Communicator, DeviceWorker<?, ?>>> unassignedWorkers = new LinkedBlockingDeque<>();
    static private ThreadLocal<Tuple<Communicator, DeviceWorker<?, ?>>> localWorker = new ThreadLocal<>();

    static public void registerWorker(Communicator c, DeviceWorker<?, ?> s) {
        unassignedWorkers.offer(new Tuple<Communicator, DeviceWorker<?, ?>>(c, s));
    }

    @Override
    public void simpleActivity(Constellation c) {
        logger.trace("start {}x{} on {}", fi, fj, c.identifier());

        try {
            Tuple<Communicator, DeviceWorker<?, ?>> pair = localWorker.get();
            if (pair == null) {
                pair = unassignedWorkers.take();
                localWorker.set(pair);
            }

            Communicator comm = pair.getFirst();
            DeviceWorker<K, R> worker = (DeviceWorker<K, R>) pair.getSecond();

            worker
                    .submitCorrelation(fi, fj)
                    .handleException(error -> {
                        logger.warn("correlation {} x {} failed", fi, fj);
                        logger.warn("exception trace", error);
                        return null;
                    })
                    .thenRun(result -> {
                        logger.trace("finish {} x {} on {}", fi, fj, comm.identifier());
                        Correlation<K, R> corr = new Correlation<>(fi, fj, result);
                        CorrelationsCollectActivity.sendToMaster(comm, collector, corr);
                    });
        } catch (Throwable e) {
            logger.warn("Correlation submission failed", e);
        }
    }
}
