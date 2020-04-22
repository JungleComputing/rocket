package nl.esciencecenter.phylogenetics_analysis;

import nl.esciencecenter.phylogenetics_analysis.kernels.CompositionVectorKernels;
import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaDevice;
import nl.esciencecenter.rocket.cubaapi.CudaMemByte;
import nl.esciencecenter.rocket.cubaapi.CudaMemDouble;
import nl.esciencecenter.rocket.cubaapi.CudaMemFloat;
import nl.esciencecenter.rocket.cubaapi.CudaMemInt;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.function.Supplier;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class BindingsTest {
    CudaContext context;

    @Before
    public void setUp() throws Exception {
        context = CudaDevice.getBestDevice().createContext();
    }

    public String generateInput(int seed, int n, String alphabet) {
        String letters = null;
        if (alphabet.equals("DNA")) letters = "ACGT";
        if (alphabet.equals("protein")) letters = "ACDEFGHIKLMNPQRSTVWYX";

        StringBuilder input = new StringBuilder();
        Random random = new Random(seed);

        while (input.length() < n) {
            for (int i = random.nextInt(500); i > 0; i--) {
                input.append(letters.charAt(random.nextInt(letters.length())));
            }

            input.append((byte) '\n');
        }

        return input.toString();
    }

    public int buildVectorCPU(String string, String alphabet, int k, int[] keys, float[] values) throws Throwable {
        int maxVectorSize = Math.min(keys.length, values.length);
        CompositionVectorKernels kernels = new CompositionVectorKernels(context, alphabet, k, maxVectorSize, string.length());

        int outSize = kernels.buildCPU(string, keys, values);

        kernels.cleanup();
        return outSize;
    }

    public int buildVectorGPU(String string, String alphabet, int k, int[] keys, float[] values) throws Throwable {
        int maxVectorSize = Math.min(keys.length, values.length);
        CompositionVectorKernels kernels = new CompositionVectorKernels(context, alphabet, k, maxVectorSize, string.length());

        CudaMemByte d_string = context.allocBytes(string.getBytes("ASCII"));
        CudaMemInt d_keys = context.allocInts(maxVectorSize);
        CudaMemFloat d_values = context.allocFloats(maxVectorSize);

        int outSize = kernels.buildVectorGPU(d_string, d_keys, d_values);

        d_keys.copyToHost(keys, maxVectorSize);
        d_values.copyToHost(values, maxVectorSize);

        kernels.cleanup();
        d_string.free();
        d_keys.free();
        d_values.free();

        return outSize;
    }

    public void testBuildVector(int seed, int n, int k, String alphabet) throws Throwable {
        String input = generateInput(seed, n, alphabet);

        int maxVectorSize = 1000 * n;
        int[] cpuKeys = new int[maxVectorSize];
        int[] gpuKeys = new int[maxVectorSize];
        float[] cpuValues = new float[maxVectorSize];
        float[] gpuValues = new float[maxVectorSize];

        int size = buildVectorCPU(input.toString(), alphabet, k, cpuKeys, cpuValues);
        int sizeGPU = buildVectorGPU(input.toString(), alphabet, k, gpuKeys, gpuValues);

        Assert.assertEquals("composition vector sizes do not match from CPU and GPU",
                size, sizeGPU);
        Assert.assertArrayEquals("composition vector keys do not match from CPU and GPU",
                Arrays.copyOf(cpuKeys, size), Arrays.copyOf(gpuKeys, size));
        Assert.assertArrayEquals("composition vector values do not match from CPU and GPU",
                Arrays.copyOf(cpuValues, size), Arrays.copyOf(gpuValues, size), 1e-6f);
    }

    public void testBuildVectors(String alphabet, int k) throws Throwable {
        for (int n: new int[]{10, 100, 1000, 10000}) {
            for (int seed: new int[]{0, 1, 2, 3, 4}) {
                //testBuildVector(seed, n, k, alphabet);
            }
        }
    }

    @Test
    public void testBuildVectorsProtein() throws Throwable {
        for (int k: new int[]{3, 4, 5, 6}) {
            testBuildVectors("protein", k);
        }
    }

    @Test
    public void testBuildVectorsDNA() throws Throwable {
        for (int k: new int[]{3, 5, 7, 9}) {
            testBuildVectors("DNA", k);
        }
    }


    public double compareVectorCPU(int[] leftKeys, float[] leftValues, int[] rightKeys, float[] rightValues) throws Throwable {
        int maxVectorSize = Math.max(leftKeys.length, rightKeys.length);
        CompositionVectorKernels kernels = new CompositionVectorKernels(context, "DNA", 3, maxVectorSize, 1);

        double output = kernels.compareCPU(leftKeys, leftValues, rightKeys, rightValues);

        kernels.cleanup();
        return output;
    }

    public double compareVectorGPU(int[] leftKeys, float[] leftValues, int[] rightKeys, float[] rightValues) throws Throwable {
        int maxVectorSize = Math.max(leftKeys.length, rightKeys.length);
        CompositionVectorKernels kernels = new CompositionVectorKernels(context, "DNA", 3, maxVectorSize, 1);

        int leftSize = leftKeys.length;
        int rightSize = rightKeys.length;

        CudaMemInt d_leftKeys = context.allocInts(leftKeys);
        CudaMemFloat d_leftValues = context.allocFloats(leftValues);
        CudaMemInt d_rightKeys = context.allocInts(rightKeys);
        CudaMemFloat d_rightValues = context.allocFloats(rightValues);
        CudaMemDouble d_output = context.allocDoubles(1);
        double[] output = new double[1];

        kernels.compareVectorsGPU(d_leftKeys, d_leftValues, leftSize, d_rightKeys, d_rightValues, rightSize, d_output);
        d_output.copyToHost(output);

        d_leftKeys.free();
        d_leftValues.free();
        d_rightKeys.free();
        d_rightValues.free();
        d_output.free();
        kernels.cleanup();

        return output[0];
    }

    public void testCompareVector(int seed, int n, double overlapRatio) throws Throwable {
        final Random random = new Random(seed);
        final HashSet<Integer> keys = new HashSet<>();
        Supplier<Integer> generateKey = () -> {
            while (true) {
                int key = Math.abs(random.nextInt());
                if (keys.add(key)) return key;
            }
        };

        int leftSize = random.nextInt(n) + n;
        int rightSize = random.nextInt(n) + n;

        // Generate left vector
        int[] leftKeys = new int[leftSize];
        float[] leftValues = new float[leftSize];

        for (int i = 0; i < leftSize; i++) {
            leftKeys[i] = generateKey.get();
            leftValues[i] = random.nextFloat();
        }

        // Generate right vector
        int[] rightKeys = new int[rightSize];
        float[] rightValues = new float[rightSize];

        for (int i = 0; i < rightSize; i++) {
            rightKeys[i] = generateKey.get();
            rightValues[i] = random.nextFloat();
        }

        // Make sure "overlapRatio*n" entries are the same.
        for (int i = 0; i < overlapRatio * n; i++) {
            leftKeys[i] = rightKeys[i] = generateKey.get();
        }

        Arrays.sort(leftKeys);
        Arrays.sort(rightKeys);

        double cpu = compareVectorCPU(leftKeys, leftValues, rightKeys, rightValues);
        double gpu = compareVectorGPU(leftKeys, leftValues, rightKeys, rightValues);

        Assert.assertEquals(cpu, gpu, 1e-5);
    }

    @Test
    public void testCompareVectors() throws Throwable {
        for (int n: new int[]{10, 100, 1000, 10000, 100000}) {
            for (double ratio: new double[]{0.0, 0.5, 1.0}) {
                for (int seed: new int[]{0, 1, 2, 3}) {
                    testCompareVector(seed, n, ratio);
                }
            }
        }
    }

    @After
    public void tearDown() throws Exception {
        context.destroy();
    }
}
