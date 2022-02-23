package nl.esciencecenter.rocket.indexspace;

import nl.esciencecenter.rocket.types.HierarchicalTask;
import nl.esciencecenter.rocket.types.LeafTask;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

import static nl.esciencecenter.rocket.util.Util.calculateTriangleSize;

public class HilbertIndexTask<K, R> implements HierarchicalTask<R>, Serializable {
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

    private final CorrelationSpawner<K, R> spawner;
    private int leftOffset;
    private int leftLength;
    private int rightOffset;
    private int rightLength;
    private K[] keys;
    private byte rule;
    private int minSplitSize;
    private boolean includeDiagonal;

    public HilbertIndexTask(CorrelationSpawner<K, R> spawner, K[] keys, int minSplitSize, boolean includeDiagonal) {
        this(spawner,
                0, keys.length, 0, keys.length,
                keys,
                minSplitSize, (byte) 0, includeDiagonal);
    }

    public HilbertIndexTask(CorrelationSpawner<K, R> spawner,
                            int leftOffset, int leftLength,
                            int rightOffset, int rightLength,
                            K[] keys,
                            int minSplitSize, byte rule, boolean includeDiagonal) {
        this.spawner = spawner;
        this.leftOffset = leftOffset;
        this.leftLength = leftLength;
        this.rightOffset = rightOffset;
        this.rightLength = rightLength;
        this.keys = keys;
        this.minSplitSize = minSplitSize;
        this.rule = rule;
        this.includeDiagonal = includeDiagonal;
    }

    public void calculateSplit(List<HilbertIndexTask<K, R>> result) {
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

            HilbertIndexTask<K, R> task = new HilbertIndexTask<>(
                    spawner,
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

            result.add(task);
        }
    }

    public boolean shouldSplit() {
        return leftLength * rightLength >= minSplitSize;
    }

    @Override
    public List<HierarchicalTask<R>> split() {
        if (!shouldSplit()) {
            return List.of();
        }

        List<HilbertIndexTask<K, R>> result = new ArrayList<>();
        calculateSplit(result);

        // Dirty!!!
        return (List<HierarchicalTask<R>>) (List<?>) result;
    }

    @Override
    public List<LeafTask<R>> getLeafs() {
        if (shouldSplit()) {
            return List.of();
        }

        List<LeafTask<R>> result = new ArrayList<>();
        Stack<HilbertIndexTask<K, R>> stack = new Stack<>();
        stack.add(this);

        while (!stack.isEmpty()) {
            HilbertIndexTask<K, R> task = stack.pop();

            if (task.leftLength == 0 || task.rightLength == 0) {
                continue;
            }

            if (task.leftLength > 1 || task.rightLength > 1) {
                task.calculateSplit(stack);
            } else {
                K left = keys[task.leftOffset];
                K right = keys[task.rightOffset];
                result.add(spawner.spawn(left, right));
            }
        }

        return result;
    }

    public int size() {
        return calculateTriangleSize(
                leftOffset, leftLength + leftOffset,
                rightOffset, rightLength + rightOffset,
                includeDiagonal);
    }

}
