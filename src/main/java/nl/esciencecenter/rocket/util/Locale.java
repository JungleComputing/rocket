package nl.esciencecenter.rocket.util;

import ch.qos.logback.core.spi.LogbackLock;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

public class Locale {
    enum Type {
        Cluster,
        Node,
        Processor
    }

    private String name;
    private Type type;
    private Optional<Locale> parent;
    private Locale[] children;

    public Locale(String name, Type type) {
        this.name = name;
        this.type = type;
        this.parent = Optional.empty();
        this.children = new Locale[0];
    }

    public Locale(String name, Type type, Locale[] children) {
        this.name = name;
        this.type = type;
        this.parent = Optional.empty();
        this.children = Arrays.copyOf(children, children.length);

        for (Locale c: children) {
            if (c.parent.isPresent()) {
                throw new IllegalArgumentException("child Locale already has parent assigned to it");
            }

            c.parent = Optional.of(this);
        }
    }

    public Locale[] findByType(Type type) {
        List<Locale> results = new ArrayList<>();
        Deque<Locale> queue = new ArrayDeque<>();
        queue.push(this);

        while (!queue.isEmpty()) {
            Locale l = queue.pop();
            if (l.type == type) {
                results.add(l);
            }

            for (Locale c: l.children) {
                queue.add(c);
            }
        }

        return results.toArray(new Locale[0]);
    }

    public Locale getRoot() {
        if (parent.isPresent()) {
            return parent.get().getRoot();
        } else {
            return this;
        }
    }

    public String getFullName() {
        if (parent.isPresent()) {
            return parent.get().getFullName() + ":" + name;
        } else {
            return name;
        }
    }

    public String getName() {
        return name;
    }

    public Type getType() {
        return type;
    }

    public Optional<Locale> getParent() {
        return parent;
    }

    public Locale[] getChildren() {
        return children;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Locale locale = (Locale) o;
        return Objects.equals(name, locale.name) &&
                type == locale.type &&
                Objects.equals(parent, locale.parent);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, type, parent);
    }

    @Override
    public String toString() {
        return "Locale{" +
                "name='" + getFullName() + '\'' +
                ", type=" + type +
                '}';
    }
}
