package nl.esciencecenter.rocket.activities;

import ibis.constellation.*;
import nl.esciencecenter.rocket.indexspace.IndexSpace;
import nl.esciencecenter.rocket.indexspace.IndexSpaceDivision;
import nl.esciencecenter.rocket.util.Correlation;
import nl.esciencecenter.rocket.util.CorrelationList;
import nl.esciencecenter.rocket.util.Tuple;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.List;

public class CorrelationsMatrixActivity<K, R> extends Activity {
    protected static final Logger logger = LogManager.getLogger();
    final public static String LABEL = CorrelationsMatrixActivity.class.getCanonicalName();

    private ActivityIdentifier collector;
    private IndexSpace<K> indexSpace;
    private int level;

    public CorrelationsMatrixActivity(
            ActivityIdentifier collector,
            IndexSpace<K> indexSpace,
            int level
    ) {
        super(new Context(LABEL, level), true);
        this.collector = collector;
        this.indexSpace = indexSpace;
        this.level = level;
    }


    @Override
    public int initialize(Constellation constellation) {
        try {
            IndexSpaceDivision<K> ctx = new IndexSpaceDivision<>();
            indexSpace.divide(ctx);

            // Launch matrix activity for each child.
            for (IndexSpace<K> t: ctx.getChildren()) {
                constellation.submit(new CorrelationsMatrixActivity<K, R>(
                        collector,
                        t,
                        level + 1
                ));
            }

            // Launch leaf activity for each leaf.
            for (Tuple<K, K> p: ctx.getEntries()) {
                constellation.submit(new CorrelationsLeafActivity<>(
                        collector,
                        identifier(),
                        p.getFirst(),
                        p.getSecond()
                ));
            }

        } catch (Exception e) {
            logger.error("error occurred while submitting tasks", e);
            throw new RuntimeException(e);
        }

        return FINISH;
    }


    @Override
    synchronized public int process(Constellation constellation, Event event) {
        return FINISH;
    }

    @Override
    public void cleanup(Constellation constellation) {
        //
    }
}
