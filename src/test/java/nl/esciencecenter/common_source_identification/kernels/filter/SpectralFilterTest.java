package nl.esciencecenter.common_source_identification.kernels.filter;

import nl.esciencecenter.rocket.util.Util;
import nl.esciencecenter.rocket.cubaapi.CudaMemFloat;
import org.junit.Test;

public class SpectralFilterTest extends AbstractFilterTest {
    @Test
    public void applyGPUTest() {
        float[][] outputCPU = new float[HEIGHT][2 * (WIDTH / 2 + 1)];
        float[] outputGPU = new float[HEIGHT * 2 * (WIDTH / 2 + 1)];

        float[][] pixelsCPU = Util.copy(pixels);
        filter.getSpectralFilter().applyCPU(pixelsCPU, outputCPU);

        float[] pixelsGPU = Util.from2DTo1D(HEIGHT, WIDTH, pixels);
        CudaMemFloat dinput = context.allocFloats(pixelsGPU.length);
        CudaMemFloat doutput = context.allocFloats(outputGPU.length);

        dinput.copyFromHost(pixelsGPU);
        filter.getSpectralFilter().applyGPU(dinput, doutput);
        doutput.copyToHost(outputGPU);

        dinput.free();
        doutput.free();

        //applyGPU CPU and GPU result
        assertArrayEquals(
                Util.from2DTo1D(outputCPU.length, outputCPU[0].length, outputCPU),
                outputGPU);
    }
}
