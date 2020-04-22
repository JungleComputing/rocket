package nl.esciencecenter.phylogenetics_analysis.kernels;

import jcuda.CudaException;
import jcuda.Sizeof;
import jcuda.runtime.cudaError;
import nl.esciencecenter.phylogenetics_analysis.Bindings;
import nl.esciencecenter.phylogenetics_analysis.PointerHack;
import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaMem;
import nl.esciencecenter.rocket.cubaapi.CudaMemByte;
import nl.esciencecenter.rocket.cubaapi.CudaMemDouble;
import nl.esciencecenter.rocket.cubaapi.CudaStream;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class CompositionVectorKernels {
    public static final int KEY_SIZE = Sizeof.INT;
    public static final int VALUE_SIZE = Sizeof.FLOAT;

    private CudaContext context;
    private CudaStream stream;
    private CudaMemByte scratch;

    private int k;
    private String alphabet;
    private long maxVectorSize;

    public CompositionVectorKernels(CudaContext context, String alphabet, int k, int maxVectorSize, int maxInputSize) {
        if (k < 3 || k > 25) throw new IllegalArgumentException("k should be in the range 3-25, not " + k);

        this.k = k;
        this.alphabet = alphabet;
        this.context = context;
        this.stream = context.createStream();
        this.maxVectorSize = maxVectorSize;

        long scratchSize = Bindings.INSTANCE.estimateScratchMemory(alphabet, k, maxVectorSize);
        this.scratch = context.allocBytes(scratchSize);
    }

    public int buildVectorGPU(CudaMemByte input, CudaMem outputKeys, CudaMem outputValues) {
        int[] size = new int[]{0};
        int err = Bindings.INSTANCE.buildCompositionVector(
                PointerHack.ptrOf(stream),
                PointerHack.ptrOf(scratch),
                scratch.elementCount(),
                alphabet,
                k,
                PointerHack.ptrOf(input),
                input.elementCount(),
                PointerHack.ptrOf(outputKeys),
                PointerHack.ptrOf(outputValues),
                size,
                (int) maxVectorSize);

        if (err != cudaError.cudaSuccess) {
            throw new CudaException(cudaError.stringFor(err));
        }

        stream.synchronize();
        return size[0];
    }

    public void compareVectorsGPU(
            CudaMem leftKeys, CudaMem leftValues, int leftSize,
            CudaMem rightKeys, CudaMem rightValues, int rightSize,
            CudaMemDouble result) {
        int err = Bindings.INSTANCE.compareCompositionVectors(
                PointerHack.ptrOf(stream),
                PointerHack.ptrOf(scratch),
                scratch.elementCount(),
                PointerHack.ptrOf(leftKeys),
                PointerHack.ptrOf(leftValues),
                leftSize,
                PointerHack.ptrOf(rightKeys),
                PointerHack.ptrOf(rightValues),
                rightSize,
                PointerHack.ptrOf(result));

        if (err != cudaError.cudaSuccess) {
            throw new CudaException(cudaError.stringFor(err));
        }

        stream.synchronize();
    }

    public int buildCPU(String input, int[] outputKeys, float[] outputValues) {
        String letters;

        if (alphabet.toUpperCase().equals("DNA")) {
            letters = "ACGT";
        } else if (alphabet.toLowerCase().equals("protein")) {
            letters = "ACDEFGHIKLMNPQRSTVWYX";
        } else {
            throw new IllegalArgumentException("invalid alphabet " + alphabet);
        }

        HashMap<String, Float> k0mers = countKmers(input, k, letters);
        HashMap<String, Float> k1mers = countKmers(input, k - 1, letters);
        HashMap<String, Float> k2mers = countKmers(input, k - 2, letters);
        HashMap<String, Float> results = new HashMap<>();

        for (String middle: k2mers.keySet()) {
            for (char before : letters.toCharArray()) {
                for (char after : letters.toCharArray()) {
                    String key = before + middle + after;
                    String prefix = before + middle;
                    String suffix = middle + after;

                    float p0 = k1mers.getOrDefault(suffix, 0.0f)
                            * k1mers.getOrDefault(prefix, 0.0f)
                            / k2mers.get(middle);
                    float p = k0mers.getOrDefault(key, 0.0f);

                    if (p0 != 0) {
                        float alpha = (p - p0) / p0;
                        results.put(key, alpha);
                    }
                }
            }
        }

        double norm = results.values().stream().mapToDouble(x -> x * x).sum();
        TreeMap<Integer, Float> output = new TreeMap<>();

        for (Map.Entry<String, Float> entry: results.entrySet()) {
            int key = kmerToDigits(entry.getKey(), letters);
            float value = entry.getValue() / ((float) Math.sqrt(norm));
            output.put(key, value);
        }

        int index = 0;

        for (Map.Entry<Integer, Float> entry: output.entrySet()) {
            outputKeys[index] = entry.getKey();
            outputValues[index] = entry.getValue();
            index++;
        }

        return index;
    }

    public double compareCPU(int[] leftKeys, float[] leftValues, int[] rightKeys, float[] rightValues) {

        List<Float> ab = new ArrayList<>();

        int i = 0, j = 0;
        while (i < leftKeys.length && j < rightKeys.length) {
            if (leftKeys[i] == rightKeys[j]) {
                //System.out.printf("CPU found %d %d (%d == %d): %f * %f == %f\n",
                //        i, j, leftKeys[i], rightKeys[j], leftValues[i], rightValues[j], leftValues[i] * rightValues[j]);

                ab.add(leftValues[i] * rightValues[j]);
                i++;

            } else if (leftKeys[i] < rightKeys[j]) {
                i++;
            } else {
                j++;
            }
        }

        return sumFloats(ab);
    }

    private HashMap<String, Float> countKmers(String input, int k, String letters) {
        HashMap<String, Integer> counts = new HashMap<>();
        int n = 0;

        for (int i = 0; i < input.length() - k + 1; i++) {
            String key = input.substring(i, i + k);
            boolean valid = true;

            for (char c: key.toCharArray()) {
                if (letters.indexOf(c) == -1) {
                    valid = false;
                }
            }

            if (valid) {
                counts.merge(key, 1, (a, b) -> a + b);
                n++;
            }
        }

        HashMap<String, Float> probability = new HashMap<>();

        for (Map.Entry<String, Integer> entry: counts.entrySet()) {
            probability.put(
                    entry.getKey(),
                    entry.getValue() / ((float) n));
        }

        return probability;
    }

    private int kmerToDigits(String key, String letters) {
        int w = 0;

        for (char c: key.toCharArray()) {
            w = w * letters.length() + letters.indexOf(c);
        }

        return w;
    }

    private double sumFloats(List<Float> l) {
        int n = l.size();

        if (n == 0) {
            return 0.0;
        } else if (n == 1) {
            return l.get(0);
        } else {
            int mid = n / 2;
            return sumFloats(l.subList(0, mid)) + sumFloats(l.subList(mid, n));
        }
    }

    /**
     * Cleans up GPU memory
     */
    public void cleanup() {
        stream.destroy();
        scratch.free();
    }

}
