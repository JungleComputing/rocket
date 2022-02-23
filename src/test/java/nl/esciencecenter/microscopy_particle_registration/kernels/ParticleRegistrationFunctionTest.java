package nl.esciencecenter.microscopy_particle_registration.kernels;

import nl.esciencecenter.microscopy_particle_registration.ParticleRegistrationContext;
import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaDevice;
import nl.esciencecenter.rocket.cubaapi.CudaMemDouble;
import nl.esciencecenter.microscopy_particle_registration.ParticleIdentifier;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

public class ParticleRegistrationFunctionTest {
    protected CudaContext context;
    protected ParticleRegistrationContext registrationFunction;
    protected int maxSize = 4096;

    /**
     * @throws java.lang.Exception
     */
    @Before
    public void setUp() throws Exception {
        context = CudaDevice.getBestDevice().createContext();
        //registrationFunction = new ParticleRegistrationContext(context, 1.0, maxSize);
    }


    @Test
    public void correlateGPUTest() {
        double scale = 1.3;
        int m = 1712;
        int n = 1513;
        double[] a = new double[2 * m];
        double[] b = new double[2 * n];
        double theta = 0.5;
        double tx = 1;
        double ty = 2;

        Random random = new Random(0);

        for (int i = 0; i < m; i++) {
            double x = ((double)i) / m * 10.0 - 5.0;
            double y = 0;

            a[2 * i + 0] = x;
            a[2 * i + 1] = y;
        }

        for (int i = 0; i < n; i++) {
            int index = random.nextInt(m);
            double x = a[2 * index + 0];
            double y = a[2 * index + 1];

            b[2 * i + 0] = Math.cos(theta) * x - Math.sin(theta) * y + tx;
            b[2 * i + 1] = Math.sin(theta) * x + Math.cos(theta) * y + ty;
        }

        double[] output = new double[3];

        // Alloc GPU memory
        CudaMemDouble da = context.allocDoubles(a.length);
        CudaMemDouble db = context.allocDoubles(b.length);
        CudaMemDouble doutput = context.allocDoubles(3);

        // Run on GPU
        da.copyFromHost(a);
        db.copyFromHost(b);

        //registrationFunction.execute(
        //        new ParticleIdentifier("", m),
        //        da,
        //        new ParticleIdentifier("", n),
        //        db,
        //        doutput);

        doutput.copyToHost(output);

        // Free GPU memory
        doutput.free();
        da.free();
        db.free();

        //assertEquals(1, 2, 1e-9);
    }

    /**
     * @throws java.lang.Exception
     */
    @After
    public void tearDown() throws Exception {
        registrationFunction.destroy();
        context.destroy();
    }
}