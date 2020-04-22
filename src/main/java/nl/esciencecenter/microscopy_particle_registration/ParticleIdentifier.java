package nl.esciencecenter.microscopy_particle_registration;

import nl.esciencecenter.common_source_identification.ImageIdentifier;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.Objects;

public class ParticleIdentifier implements Serializable, Comparable<ParticleIdentifier> {
    private static final long serialVersionUID = 1041546049524125024L;

    private int index;
    private String path;
    private int size;

    public ParticleIdentifier(int index, String path, int size) {
        this.index = index;
        this.path = path;
        this.size = size;
    }

    public int getIndex() {
        return index;
    }

    public String getPath() {
        return path;
    }

    public int getNumberOfPoints() {
        return size;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ParticleIdentifier that = (ParticleIdentifier) o;
        return Objects.equals(path, that.path) &&
                size == that.size &&
                index == that.index;
    }

    @Override
    public int hashCode() {
        return index;
    }

    @Override
    public String toString() {
        return path;
    }

    @Override
    public int compareTo(ParticleIdentifier that) {
        return this.path.compareTo(that.path);
    }
}
