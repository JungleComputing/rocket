package nl.esciencecenter.rocket.indexspace;

import nl.esciencecenter.rocket.util.Tuple;

import java.util.ArrayList;
import java.util.List;

public class IndexSpaceDivision<K> {
    protected ArrayList<IndexSpace<K>> subIndexSpaces;
    protected ArrayList<Tuple<K, K>> leafTasks;

    public IndexSpaceDivision() {
        subIndexSpaces = new ArrayList<>();
        leafTasks = new ArrayList<>();
    }

    public void addSubspace(IndexSpace<K> indexSpace) {
        subIndexSpaces.add(indexSpace);
    }

    public void addEntry(K i, K j) {
        leafTasks.add(new Tuple(i, j));
    }

    public List<IndexSpace<K>> getChildren() {
        return subIndexSpaces;
    }

    public List<Tuple<K, K>> getEntries() {
        return leafTasks;
    }
}
