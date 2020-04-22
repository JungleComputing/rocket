package nl.esciencecenter.rocket.indexspace;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Collections;
import java.util.Random;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import static nl.esciencecenter.rocket.util.Util.calculateTriangleSize;

public class TilingIndexSpace<K> implements IndexSpace<K>, Serializable {
    private static final long serialVersionUID = -7952208898196702639L;
    protected static final Logger logger = LogManager.getLogger();

    private final int tileSize;
    private final int rowBegin;
    private final int rowEnd;
    private final int colBegin;
    private final int colEnd;
    private final boolean includeDiagonal;
    private final K[] keys;

    public TilingIndexSpace(
            int tileSize,
            K[] keys,
            boolean includeDiagonal
    ) {
        this(tileSize,
                0, keys.length,
                0, keys.length,
                keys,
                includeDiagonal);
    }
    public TilingIndexSpace(
            int tileSize,
            int rowBegin, int rowEnd,
            int colBegin, int colEnd,
            K[] keys,
            boolean includeDiagonal
    ) {
        this.rowBegin = rowBegin;
        this.rowEnd = rowEnd;
        this.colBegin = colBegin;
        this.colEnd = colEnd;
        this.tileSize = tileSize;
        this.includeDiagonal = includeDiagonal;
        this.keys = keys;
    }

    @Override
    public void divide(IndexSpaceDivision<K> result) {
        int numRows = rowEnd - rowBegin;
        int numCols = colEnd - colBegin;

        if (numRows > tileSize || numCols > tileSize) {
            // Divide matrix into tiles
            for (int i = rowBegin; i < rowEnd; i += tileSize) {
                for (int j = colBegin; j < colEnd; j+= tileSize) {
                    TilingIndexSpace<K> subspace = new TilingIndexSpace<>(
                            tileSize,
                            i, Math.min(i + tileSize, rowEnd),
                            j, Math.min(j + tileSize, colEnd),
                            keys,
                            includeDiagonal);

                    if (subspace.size() > 0) {
                        result.addSubspace(subspace);
                    }
                }
            }
        } else {
            // Divide tile into individual entries.
            for (int i = rowBegin; i < rowEnd; i++) {
                for (int j = colBegin; j < colEnd; j++) {
                    if (i < j || (i == j && includeDiagonal)) {
                        result.addEntry(keys[i], keys[j]);
                    }
                }
            }
        }
    }

    @Override
    public int size() {
        return calculateTriangleSize(rowBegin, rowEnd, colBegin, colEnd, includeDiagonal);
    }
}
