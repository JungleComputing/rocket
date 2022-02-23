package nl.esciencecenter.rocket.types;

import java.io.Serializable;

public interface HashableKey extends Serializable, Comparable<HashableKey> {
    // must override:
    // - toString
    // - hash
}
