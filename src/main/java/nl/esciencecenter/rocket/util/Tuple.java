package nl.esciencecenter.rocket.util;

import java.io.Serializable;
import java.util.Objects;

public class Tuple<A, B> implements Serializable {


    private A first;
    private B second;

    static public <A, B> Tuple<A, B> of(A a, B b) {
        return new Tuple<A, B>(a, b);
    }

    public Tuple(A first, B second) {
        this.first = first;
        this.second = second;
    }

    public A getFirst() {
        return first;
    }

    public B getSecond() {
        return second;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Tuple<?, ?> tuple = (Tuple<?, ?>) o;
        return Objects.equals(first, tuple.first) &&
                Objects.equals(second, tuple.second);
    }

    @Override
    public int hashCode() {
        return Objects.hash(first, second);
    }

    @Override
    public String toString() {
        return "Tuple{" +
                "first=" + first +
                ", second=" + second +
                '}';
    }
}
