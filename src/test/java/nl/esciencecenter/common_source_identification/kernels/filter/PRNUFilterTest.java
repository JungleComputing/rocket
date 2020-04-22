package nl.esciencecenter.common_source_identification.kernels.filter;

import nl.esciencecenter.rocket.util.Util;
import nl.esciencecenter.rocket.cubaapi.CudaMemByte;
import nl.esciencecenter.rocket.cubaapi.CudaMemFloat;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class PRNUFilterTest extends AbstractFilterTest {
    @Test
    public void applyGPUTest() {
        byte[] rgbImageGPU = new byte[HEIGHT * WIDTH * 3];
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                for (int i = 0; i < 3; i++) {
                    rgbImageGPU[(y * WIDTH + x) * 3 + i] = rgbImage[y][x][i];
                }
            }
        }

        float[][] outputCPU = new float[HEIGHT][2 * (WIDTH / 2 + 1)];
        filter.applyCPU(rgbImage, outputCPU);

        float[] outputGPU = new float[HEIGHT * 2 * (WIDTH / 2 + 1)];
        CudaMemByte dinput = context.allocBytes(rgbImageGPU.length);
        CudaMemFloat doutput = context.allocFloats(outputGPU.length);

        dinput.copyFromHost(rgbImageGPU);
        filter.applyGPU(dinput, doutput);
        doutput.copyToHost(outputGPU);

        dinput.free();
        doutput.free();


        //applyGPU CPU and GPU result
        assertArrayEquals(
                Util.from2DTo1D(outputCPU.length, outputCPU[0].length, outputCPU),
                outputGPU);
    }
}
