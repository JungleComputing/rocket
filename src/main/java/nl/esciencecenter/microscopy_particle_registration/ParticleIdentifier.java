package nl.esciencecenter.microscopy_particle_registration;

import nl.esciencecenter.rocket.types.HashableKey;

import java.util.Objects;

public class ParticleIdentifier implements HashableKey {
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
    public int compareTo(HashableKey that) {
        if (that == null || getClass() != that.getClass()) return 0;
        return this.path.compareTo(((ParticleIdentifier) that).path);
    }
}
