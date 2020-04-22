package nl.esciencecenter.phylogenetics_analysis;


import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.Objects;

public class SequenceIdentifier implements Serializable, Comparable<SequenceIdentifier> {
    private static final long serialVersionUID = -4841143185809316055L;

    private String path;

    public SequenceIdentifier(String path) {
        this.path = path;
    }

    public String getPath() {
        return path;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        SequenceIdentifier that = (SequenceIdentifier) o;
        return Objects.equals(path, that.path);
    }

    @Override
    public int hashCode() {
        return Objects.hash(path);
    }

    @Override
    public String toString() {
        return path;
    }

    @Override
    public int compareTo(SequenceIdentifier that) {
        return this.path.compareTo(that.path);
    }
}
