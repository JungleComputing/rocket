package nl.esciencecenter.rocket.util;


import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

public class CorrelationList<K, R> implements Iterable<Correlation<K, R>>, Serializable {
    private static final long serialVersionUID = 8985935534379029190L;

    private List<Correlation<K, R>> list;

    public CorrelationList() {
        list = Collections.EMPTY_LIST;
    }

    public CorrelationList(K left, K right, R result) {
        list = Collections.singletonList(new Correlation<>(left, right, result));
    }

    public CorrelationList(List<Correlation<K, R>> other) {
        list = Collections.unmodifiableList(other);
    }

    public List<Correlation<K, R>> toList() {
        return new ArrayList<>(list);
    }

    public List<Correlation<K, R>> toList(Comparator<K> cmp) {
        List<Correlation<K, R>> list = toList();

        list.sort((x, y) -> {
            int c = 0;
            if (c == 0) c = cmp.compare(x.getI(), y.getI());
            if (c == 0) c = cmp.compare(x.getJ(), y.getJ());
            return c;
        });

        return list;
    }

    @Override
    public Iterator<Correlation<K, R>> iterator() {
        return list.iterator();
    }
}
