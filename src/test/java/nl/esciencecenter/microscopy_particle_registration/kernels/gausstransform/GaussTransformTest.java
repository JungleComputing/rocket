package nl.esciencecenter.microscopy_particle_registration.kernels.gausstransform;

import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaDevice;
import nl.esciencecenter.rocket.cubaapi.CudaMemDouble;
import nl.esciencecenter.microscopy_particle_registration.kernels.transformrigid2d.TransformRigid2D;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.*;

public class GaussTransformTest {
    protected CudaContext context;
    protected GaussTransform gaussTransform;
    protected int maxSize = 4096;

    /**
     * @throws java.lang.Exception
     */
    @Before
    public void setUp() throws Exception {
        context = CudaDevice.getBestDevice().createContext();
        TransformRigid2D rigidTransform = new TransformRigid2D(context);
        gaussTransform = new GaussTransform(context, rigidTransform, maxSize);
    }


    @Test
    public void applyGPUTest() {
        double scale = 1.3;
        int m = 1712;
        int n = 1513;
        double[] a = new double[2 * m];
        double[] b = new double[2 * n];
        double theta = 0;
        double tx = 0;
        double ty = 0;

        Random random = new Random(0);

        for (int i = 0; i < a.length / 2; i++) {
            double x = random.nextDouble();
            double y = random.nextDouble();

            a[2 * i + 0] = x;
            a[2 * i + 1] = y;
        }

        for (int i = 0; i < b.length; i++) {
            b[i] = random.nextDouble();
        }

        double[] gradientCPU = new double[3];
        double[] gradientGPU = new double[3];
        double[] crossTermGPU = new double[1];

        // Run on CPU
        double crossTermCPU = gaussTransform.applyCPU(a, b, tx, ty, theta, scale, gradientCPU);

        // Alloc GPU memory
        CudaMemDouble da = context.allocDoubles(a.length);
        CudaMemDouble db = context.allocDoubles(b.length);
        CudaMemDouble dgradient = context.allocDoubles(gradientCPU.length);
        CudaMemDouble dcrossTerm = context.allocDoubles(1);

        // Run on GPU
        da.copyFromHost(a);
        db.copyFromHost(b);
        gaussTransform.applyGPU(da, db, tx, ty, theta, scale, dgradient, dcrossTerm);
        dgradient.copyToHost(gradientGPU);
        dcrossTerm.copyToHost(crossTermGPU);

        // Free GPU memory
        da.free();
        db.free();
        dgradient.free();
        dcrossTerm.free();

        assertEquals(crossTermCPU, crossTermGPU[0], 1e-9);
        assertArrayEquals(gradientCPU, gradientGPU, 1e-9);
    }

    /**
     * @throws java.lang.Exception
     */
    @After
    public void tearDown() throws Exception {
        gaussTransform.cleanup();
        context.destroy();
    }
}