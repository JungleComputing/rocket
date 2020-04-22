package nl.esciencecenter.rocket.indexspace;

import java.io.Serializable;

import static nl.esciencecenter.rocket.util.Util.calculateTriangleSize;

public class HilbertCurveIndexSpace<K> implements IndexSpace<K>, Serializable {
    private static final long serialVersionUID = 2805522066207471234L;

    private static int[][][] spaceFillingRules;
    static {
        int a = 0, b = 1, c = 2, d = 3;

        spaceFillingRules = new int[][][] {
                { {0, 0, d}, {0, 1, a}, {1, 1, a}, {1, 0, b}}, // Hilbert rule A
                { {1, 1, c}, {0, 1, b}, {0, 0, b}, {1, 0, a}}, // Hilbert rule B
                { {1, 1, b}, {1, 0, c}, {0, 0, c}, {0, 1, d}}, // Hilbert rule C
                { {0, 0, a}, {1, 0, d}, {1, 1, d}, {0, 1, c}}, // Hilbert rule D
        };
    }

    private int leftOffset;
    private int leftLength;
    private int rightOffset;
    private int rightLength;
    private K[] keys;
    private byte rule;
    private int minSplitSize;
    private boolean includeDiagonal;

    public HilbertCurveIndexSpace(K[] keys, int minSplitSize, boolean includeDiagonal) {
        this(0, keys.length, 0, keys.length,
                keys,
                minSplitSize, (byte) 0, includeDiagonal);
    }

    public HilbertCurveIndexSpace(int leftOffset, int leftLength,
                                  int rightOffset, int rightLength,
                                  K[] keys,
                                  int minSplitSize, byte rule, boolean includeDiagonal) {
        this.leftOffset = leftOffset;
        this.leftLength = leftLength;
        this.rightOffset = rightOffset;
        this.rightLength = rightLength;
        this.keys = keys;
        this.minSplitSize = minSplitSize;
        this.rule = rule;
        this.includeDiagonal = includeDiagonal;
    }

    private void launchSplits(IndexSpaceDivision<K> result) {
        int leftHalf = leftLength / 2;
        int rightHalf = rightLength / 2;

        for (int i = 0; i < 4; i++) {
            int p = spaceFillingRules[rule][i][0];
            int q = spaceFillingRules[rule][i][1];
            byte nextRule = (byte) spaceFillingRules[rule][i][2];

            // Split the left interval [leftOffset...leftOffset + leftLength]
            // into subinterval [l0...l1] with length ln
            int l0 = leftOffset + Math.min(leftLength, p == 0 ? 0 : leftHalf);
            int l1 = leftOffset + Math.min(leftLength, p == 0 ? leftHalf: leftLength);
            int ln = l1 - l0;

            // Split the right interval [rightOffset...rightOffset + rightLength]
            // into subinterval [r0...r1] with length rn
            int r0 = rightOffset + Math.min(rightLength, q == 0 ? 0 : rightHalf);
            int r1 = rightOffset + Math.min(rightLength, q == 0 ? rightHalf : rightLength);
            int rn = r1 - r0;

            // One of the two subintervals is empty, skip this submatrix.
            if (ln == 0 || rn == 0) {
                continue;
            }

            HilbertCurveIndexSpace<K> task = new HilbertCurveIndexSpace<K>(
                    l0,
                    ln,
                    r0,
                    rn,
                    keys,
                    minSplitSize,
                    nextRule,
                    includeDiagonal);

            // Skip emtpy matrices
            if (task.size() == 0) {
                continue;
            }

            if (ln * rn < minSplitSize) {
                task.divide(result);
            } else {
                result.addSubspace(task);
            }
        }
    }

    private void launchEntry(IndexSpaceDivision<K> spawner) {
        if (leftOffset < rightOffset || (leftOffset == rightOffset && includeDiagonal)) {
            spawner.addEntry(keys[leftOffset], keys[rightOffset]);
        }
    }

    @Override
    public void divide(IndexSpaceDivision<K> result) {
        if (leftLength > 1 || rightLength > 1) {
            launchSplits(result);
        } else {
            launchEntry(result);
        }
    }

    @Override
    public int size() {
        return calculateTriangleSize(
                leftOffset, leftLength + leftOffset,
                rightOffset, rightLength + rightOffset,
                includeDiagonal);
    }

}
