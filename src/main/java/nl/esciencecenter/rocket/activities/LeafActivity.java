package nl.esciencecenter.rocket.activities;

import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Constellation;
import ibis.constellation.Context;
import ibis.constellation.util.SimpleActivity;
import nl.esciencecenter.rocket.types.LeafTask;
import nl.esciencecenter.rocket.util.Tuple;
import nl.esciencecenter.rocket.worker;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingDeque;

public class LeafActivity<R> extends SimpleActivity {
    protected static final Logger logger = LogManager.getLogger();
    final public static String LABEL = LeafActivity.class.getCanonicalName();

    private ActivityIdentifier collector;
    private LeafTask<R> task;

    public LeafActivity(ActivityIdentifier collector, ActivityIdentifier parent, LeafTask<R> task) {
        super(parent, new Context(LABEL, Integer.MAX_VALUE), false);
        this.collector = collector;
        this.task = task;
    }

    static private BlockingQueue<Tuple<Communicator, worker.DeviceWorker<?>>> unassignedWorkers = new LinkedBlockingDeque<>();
    static private ThreadLocal<Tuple<Communicator, worker.DeviceWorker<?>>> localWorker = new ThreadLocal<>();

    static public void registerWorker(Communicator c, worker.DeviceWorker<?> s) {
        unassignedWorkers.offer(new Tuple<Communicator, worker.DeviceWorker<?>>(c, s));
    }

    @Override
    public void simpleActivity(Constellation c) {
        logger.trace("start {} on {}", task, c.identifier());

        try {
            Tuple<Communicator, worker.DeviceWorker<?>> pair = localWorker.get();
            if (pair == null) {
                pair = unassignedWorkers.take();
                localWorker.set(pair);
            }

            Communicator comm = pair.getFirst();
            worker.DeviceWorker<R> worker = (nl.esciencecenter.rocket.worker.DeviceWorker<R>) pair.getSecond();

            worker
                    .submit(task)
                    .handleException(error -> {
                        logger.warn("correlation {} failed", task);
                        logger.warn("exception trace", error);
                        return null;
                    })
                    .thenRun(result -> {
                        logger.trace("finish {} on {}", task, comm.identifier());
                        ResultsCollectActivity.sendToMaster(comm, collector, result);
                    });
        } catch (Throwable e) {
            logger.warn("Correlation submission failed", e);
        }
    }
}
