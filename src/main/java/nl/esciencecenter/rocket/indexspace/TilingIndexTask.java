package nl.esciencecenter.rocket.indexspace;

import nl.esciencecenter.rocket.types.HierarchicalTask;
import nl.esciencecenter.rocket.types.LeafTask;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import static nl.esciencecenter.rocket.util.Util.calculateTriangleSize;

public class TilingIndexTask<K, R> implements HierarchicalTask<R>, Serializable {
    private static final long serialVersionUID = -7952208898196702639L;
    protected static final Logger logger = LogManager.getLogger();

    private final CorrelationSpawner<K, R> spawner;
    private final int tileSize;
    private final int rowBegin;
    private final int rowEnd;
    private final int colBegin;
    private final int colEnd;
    private final boolean includeDiagonal;
    private final K[] keys;

    public TilingIndexTask(
            CorrelationSpawner spawner,
            int tileSize,
            K[] keys,
            boolean includeDiagonal
    ) {
        this(spawner,
                tileSize,
                0, keys.length,
                0, keys.length,
                keys,
                includeDiagonal);
    }

    public TilingIndexTask(
            CorrelationSpawner<K, R> spawner,
            int tileSize,
            int rowBegin, int rowEnd,
            int colBegin, int colEnd,
            K[] keys,
            boolean includeDiagonal
    ) {
        this.spawner = spawner;
        this.rowBegin = rowBegin;
        this.rowEnd = rowEnd;
        this.colBegin = colBegin;
        this.colEnd = colEnd;
        this.tileSize = tileSize;
        this.includeDiagonal = includeDiagonal;
        this.keys = keys;
    }

    public List<HierarchicalTask<R>> split() {
        List<HierarchicalTask<R>> result = new ArrayList<>();
        int numRows = rowEnd - rowBegin;
        int numCols = colEnd - colBegin;

        if (numRows > tileSize || numCols > tileSize) {
            // Divide matrix into tiles
            for (int i = rowBegin; i < rowEnd; i += tileSize) {
                for (int j = colBegin; j < colEnd; j+= tileSize) {
                    TilingIndexTask<K, R> subspace = new TilingIndexTask<>(
                            spawner,
                            tileSize,
                            i, Math.min(i + tileSize, rowEnd),
                            j, Math.min(j + tileSize, colEnd),
                            keys,
                            includeDiagonal);

                    if (subspace.size() > 0) {
                        result.add(subspace);
                    }
                }
            }
        }

        return result;
    }

    public List<LeafTask<R>> getLeafs() {
        List<LeafTask<R>> result = new ArrayList<>();
        int numRows = rowEnd - rowBegin;
        int numCols = colEnd - colBegin;

        if (numRows > tileSize || numCols > tileSize) {
            return List.of();
        }

        // Divide tile into individual entries.
        for (int i = rowBegin; i < rowEnd; i++) {
            for (int j = colBegin; j < colEnd; j++) {
                if (i < j || (i == j && includeDiagonal)) {
                    result.add(spawner.spawn(keys[i], keys[j]));
                }
            }
        }

        return result;
    }


    public int size() {
        return calculateTriangleSize(rowBegin, rowEnd, colBegin, colEnd, includeDiagonal);
    }
}
