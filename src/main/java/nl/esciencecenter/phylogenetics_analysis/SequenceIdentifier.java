package nl.esciencecenter.phylogenetics_analysis;


import nl.esciencecenter.rocket.types.HashableKey;

import java.util.Objects;

public class SequenceIdentifier implements HashableKey {
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
    public int compareTo(HashableKey that) {
        if (that == null || getClass() != that.getClass()) return 0;
        return this.path.compareTo(((SequenceIdentifier) that).path);
    }
}
