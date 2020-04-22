package nl.esciencecenter.radio_correlator.kernels;

import jcuda.CudaException;
import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaFunction;
import nl.esciencecenter.rocket.cubaapi.CudaMemFloat;
import nl.esciencecenter.rocket.cubaapi.CudaModule;
import nl.esciencecenter.rocket.cubaapi.CudaStream;

import java.io.IOException;

public class Correlator {
    static final public int NUM_POLARIZATIONS = 2;
    static final public int BLOCK_SIZE = 256;

    private CudaStream stream;
    private CudaContext context;
    private CudaModule module;
    private CudaFunction correlate;
    private int numChannels;
    private int numTimes;

    public Correlator(CudaContext context, int numChannels, int numTimes) throws IOException, CudaException {
        this.context = context;
        this.module = context.compileModule(
                getClass().getResource("gpu_correlator_1x1.cu"),
                "-DBLOCK_SIZE=" + BLOCK_SIZE);
        this.correlate = module.getFunction("correlate");
        this.stream = context.createStream();

        this.numChannels = numChannels;
        this.numTimes = numTimes;
    }

    public void compareGPU(CudaMemFloat left, CudaMemFloat right, CudaMemFloat result) {
        if (left.elementCount() != numChannels * numTimes * NUM_POLARIZATIONS * 2 ||
                right.elementCount() != numChannels * numTimes * NUM_POLARIZATIONS * 2 ||
                result.elementCount() != NUM_POLARIZATIONS * NUM_POLARIZATIONS * numChannels * 2) {
            throw new IllegalArgumentException("invalid buffer size");
        }

        correlate.setDim(numChannels, BLOCK_SIZE);
        correlate.launch(
                stream,
                numTimes,
                left,
                right,
                result);

        stream.synchronize();
    }

    public void compareCPU(float[] left, float[] right, float[] result) {
        if (left.length != numChannels * numTimes * NUM_POLARIZATIONS * 2 ||
                right.length != numChannels * numTimes * NUM_POLARIZATIONS * 2 ||
                result.length != NUM_POLARIZATIONS * NUM_POLARIZATIONS * numChannels * 2) {
            throw new IllegalArgumentException("invalid buffer size");
        }

        for (int channel = 0; channel < numChannels; channel++) {
            float xxr = 0, xxi = 0;
            float xyr = 0, xyi = 0;
            float yxr = 0, yxi = 0;
            float yyr = 0, yyi = 0;

            for (int t = 0; t < numTimes; t++) {
                // array[channel][t][polarization][component]
                int offset = (numTimes * channel + t) * NUM_POLARIZATIONS * 2;

                float lxr = left[offset + 0];
                float lxi = left[offset + 1];
                float lyr = left[offset + 2];
                float lyi = left[offset + 3];

                float rxr = right[offset + 0];
                float rxi = right[offset + 1];
                float ryr = right[offset + 2];
                float ryi = right[offset + 3];

                // lx * conj(rx)
                xxr += lxr * rxr + lxi * rxi;
                xxi += lxi * rxr - lxr * rxi;

                // lx * conj(ry)
                xyr += lxr * ryr + lxi * ryi;
                xyi += lxi * ryr - lxr * ryi;

                // ly * conj(rx)
                yxr += lyr * rxr + lyi * rxi;
                yxi += lyi * rxr - lyr * rxi;

                // ly * conj(ry)
                yyr += lyr * ryr + lyi * ryi;
                yyi += lyi * ryr - lyr * ryi;
            }

            int offset = channel * NUM_POLARIZATIONS * NUM_POLARIZATIONS * 2;
            result[offset + 0] = xxr;
            result[offset + 1] = xxi;
            result[offset + 2] = xyr;
            result[offset + 3] = xyi;
            result[offset + 4] = yxr;
            result[offset + 5] = yxi;
            result[offset + 6] = yyr;
            result[offset + 7] = yyi;
        }
    }

    public void cleanup() {
        stream.destroy();
        module.cleanup();
    }
}
