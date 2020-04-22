package nl.esciencecenter.common_source_identification.util;

import java.io.Serializable;
import java.util.Objects;

public class Dimension implements Serializable  {
    private static final long serialVersionUID = 2814654887691834545L;

    private int width;
    private int height;

    public Dimension(int height, int width) {
        this.width = width;
        this.height = height;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Dimension dimension = (Dimension) o;
        return width == dimension.width &&
                height == dimension.height;
    }

    @Override
    public int hashCode() {
        return Objects.hash(width, height);
    }

    @Override
    public String toString() {
        return "Dimension{" +
                "width=" + width +
                ", height=" + height +
                '}';
    }


}
