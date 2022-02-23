package nl.esciencecenter.rocket.indexspace;

import nl.esciencecenter.rocket.types.LeafTask;

import java.io.Serializable;

public interface CorrelationSpawner<K, R> extends Serializable {
    LeafTask<R> spawn(K left, K right);
}
