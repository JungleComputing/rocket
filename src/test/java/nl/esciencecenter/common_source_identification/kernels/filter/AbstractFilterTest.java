package nl.esciencecenter.common_source_identification.kernels.filter;

import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaDevice;
import org.junit.After;
import org.junit.Before;

import java.util.Random;

import static org.junit.Assert.assertEquals;

public abstract class AbstractFilterTest {
    protected int HEIGHT = 2304;
    protected int WIDTH = 3072;

    protected CudaContext context;
    protected PRNUFilter filter;
    protected float[][] pixels;
    protected byte[][][] rgbImage;

    /**
     * @throws java.lang.Exception
     */
    @Before
    public void setUp() throws Exception {
        context = CudaDevice.getBestDevice().createContext();
        filter = new PRNUFilter(HEIGHT, WIDTH, context, true);
        Random random = new Random();
        pixels = new float[HEIGHT][WIDTH];

        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                pixels[y][x] = random.nextFloat();
            }
        }

        rgbImage = new byte[HEIGHT][WIDTH][3];

        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                random.nextBytes(rgbImage[y][x]);
            }
        }
    }

    /**
     * @throws java.lang.Exception
     */
    @After
    public void tearDown() throws Exception {
        filter.cleanup();
        context.destroy();
    }

    public void assertArrayEquals(float[] a, float[] b) {
        assertEquals(a.length, b.length);
        float eps = (float) 0.05;

        for (int i = 0; i < a.length; i++) {
            float x = a[i];
            float y = b[i];

            float abs = Math.abs(x - y);
            float rel = abs / Math.max(Math.abs(x), Math.abs(y));

            if (Float.isNaN(x) && Float.isNaN(y)) {
                // valid
            } else if (Float.isInfinite(x) && Float.isInfinite(y)) {
                // valid
            } else if (abs >= eps && rel >= eps) {
                assertEquals("at index " + i, x, y, eps);
            }
        }

    }
}
