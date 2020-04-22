package nl.esciencecenter.radio_correlator.kernels;

import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaDevice;
import nl.esciencecenter.rocket.cubaapi.CudaMemFloat;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.*;

public class CorrelatorTest {

    final static private int NUM_CHANNELS = 1200;
    final static private int NUM_TIMES = 768;
    static final private int NUM_POLARIZATIONS = 2;
    protected CudaContext context;
    private Correlator correlator;

    /**
     * @throws java.lang.Exception
     */
    @Before
    public void setUp() throws Exception {
        context = CudaDevice.getBestDevice().createContext();
        correlator = new Correlator(context, NUM_CHANNELS, NUM_TIMES);
    }

    @Test
    public void compareGPUTest() {
        float[] resultCPU = new float[NUM_CHANNELS * NUM_POLARIZATIONS * NUM_POLARIZATIONS * 2];
        float[] resultGPU = new float[resultCPU.length];

        float[] left = new float[NUM_CHANNELS * NUM_TIMES * NUM_POLARIZATIONS * 2];
        float[] right = new float[NUM_CHANNELS * NUM_TIMES * NUM_POLARIZATIONS * 2];
        Random random = new Random(0);

        for (int i = 0; i < NUM_CHANNELS; i++) {
            for (int t = 0; t < NUM_TIMES; t++) {
                for (int p = 0; p < NUM_POLARIZATIONS; p++) {
                    for (int c = 0; c < 2; c++) {
                        int offset = ((i * NUM_TIMES + t) * NUM_POLARIZATIONS + p) * 2 + c;

                        left[offset] = random.nextFloat();
                        right[offset] = random.nextFloat();
                    }
                }
            }
        }

        correlator.compareCPU(left, right, resultCPU);


        // Alloc GPU memory
        CudaMemFloat dleft = context.allocFloats(left);
        CudaMemFloat dright = context.allocFloats(right);
        CudaMemFloat dresult = context.allocFloats(resultGPU.length);

        // Run
        correlator.compareGPU(dleft, dright, dresult);

        // Copy output
        dresult.copyToHost(resultGPU);

        // Free GPU memory
        dleft.free();
        dright.free();
        dresult.free();

        assertArrayEquals(resultCPU, resultGPU, 0.001f);
    }

    /**
     * @throws java.lang.Exception
     */
    @After
    public void tearDown() throws Exception {
        correlator.cleanup();
        context.destroy();
    }
}