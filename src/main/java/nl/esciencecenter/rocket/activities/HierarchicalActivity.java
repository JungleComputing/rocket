package nl.esciencecenter.rocket.activities;

import ibis.constellation.*;
import nl.esciencecenter.rocket.types.HierarchicalTask;
import nl.esciencecenter.rocket.types.LeafTask;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class HierarchicalActivity<R> extends Activity {
    protected static final Logger logger = LogManager.getLogger();
    final public static String LABEL = HierarchicalActivity.class.getCanonicalName();

    private ActivityIdentifier parent;
    private ActivityIdentifier collector;
    private HierarchicalTask<R> task;
    private int level;
    private int totalLeafs = 0;
    private int eventsPending = 0;

    public HierarchicalActivity(
            ActivityIdentifier parent,
            ActivityIdentifier collector,
            HierarchicalTask<R> task,
            int level
    ) {
        super(new Context(LABEL, level), true);
        this.parent = parent;
        this.collector = collector;
        this.task = task;
        this.level = level;
    }


    @Override
    public int initialize(Constellation constellation) {
        try {
            // Launch matrix activity for each child.
            for (HierarchicalTask<R> t: task.split()) {
                constellation.submit(new HierarchicalActivity<R>(
                        identifier(),
                        collector,
                        t,
                        level + 1
                ));

                eventsPending++;
            }

            // Launch leaf activity for each leaf.
            for (LeafTask<R> p: task.getLeafs()) {
                constellation.submit(new LeafActivity<>(
                        collector,
                        identifier(),
                        p
                ));

                totalLeafs++;
            }

        } catch (Exception e) {
            logger.error("error occurred while submitting tasks", e);
            throw new RuntimeException(e);
        }

        return eventsPending > 0 ? SUSPEND : FINISH;
    }


    @Override
    public int process(Constellation constellation, Event event) {
        totalLeafs += (Integer) event.getData();

        eventsPending--;
        return eventsPending > 0 ? SUSPEND : FINISH;
    }

    @Override
    public void cleanup(Constellation constellation) {
        constellation.send(new Event(
                identifier(),
                parent,
                (Integer) totalLeafs
        ));
    }
}
