package nl.esciencecenter.dummy;

import nl.esciencecenter.rocket.types.HashableKey;

public class DummyIdentifier implements HashableKey {
    private static final long serialVersionUID = -2985516246452858565L;

    private int index;
    public DummyIdentifier(int index) {
        this.index = index;
    }

    public int getIndex() {
        return index;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        DummyIdentifier that = (DummyIdentifier) o;
        return index == that.index;
    }

    @Override
    public int hashCode() {
        return index;
    }

    @Override
    public String toString() {
        return String.valueOf(index);
    }

    @Override
    public int compareTo(HashableKey that) {
        if (that == null || getClass() != that.getClass()) return 0;
        return Integer.compare(this.index, ((DummyIdentifier) that).index);
    }
}
