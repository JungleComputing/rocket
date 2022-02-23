package nl.esciencecenter.rocket.types;

import java.util.List;

public interface HierarchicalTask<R> {

    /**
     *
     * @return
     */
    public List<HierarchicalTask<R>> split();

    /**
     *
     * @return
     */
    public List<LeafTask<R>> getLeafs();
}
