package nl.esciencecenter.radio_correlator;

import nl.esciencecenter.rocket.types.HashableKey;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Objects;

public class StationIdentifier implements HashableKey {
    private static final long serialVersionUID = 8341483315405170456L;

    private String path;

    public StationIdentifier(String path) {
        this.path = path;
    }

    public String getPath() {
        return path;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        StationIdentifier that = (StationIdentifier) o;
        return Objects.equals(path, that.path);
    }

    @Override
    public int hashCode() {
        return Objects.hash(path);
    }

    @Override
    public String toString() {
        return "StationIdentifier{" +
                "path='" + path + '\'' +
                '}';
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        path = path.intern();
    }

    @Override
    public int compareTo(HashableKey that) {
        if (that == null || !(that instanceof StationIdentifier)) return 0;
        return this.path.compareTo(((StationIdentifier) that).path);
    }
}
