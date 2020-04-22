package nl.esciencecenter.microscopy_particle_registration.kernels.transformrigid2d;

import jcuda.CudaException;
import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaFunction;
import nl.esciencecenter.rocket.cubaapi.CudaMemDouble;
import nl.esciencecenter.rocket.cubaapi.CudaModule;
import nl.esciencecenter.rocket.cubaapi.CudaStream;

import java.io.IOException;

public class TransformRigid2D {

    //cuda handles
    protected CudaModule _module;
    protected CudaFunction _transformRigid;

    public TransformRigid2D(CudaContext context) throws CudaException, IOException {
        _module = context.compileModule(getClass().getResource("kernels.cu"));
        _transformRigid = _module.getFunction("TransformRigid2D");
    }

    public void applyGPU(CudaStream stream, CudaMemDouble input, CudaMemDouble output,
                         int n, double theta, double translate_x, double translate_y) {
        if (input.elementCount() < 2 * n || output.elementCount() < 2 * n) {
            throw new IllegalArgumentException("invalid element count " );
        }

        _transformRigid.setDim(n / 1024 + 1, 1024);
        _transformRigid.launch(
                stream,
                input,
                output,
                n,
                Math.cos(theta),
                Math.sin(theta),
                translate_x,
                translate_y);
    }

    public void applyCPU(double[] input, double[] output, int n, double theta, double translate_x, double translate_y) {
        for (int i = 0; i < n; i++) {
            double x = input[2 * i + 0];
            double y = input[2 * i + 1];

            output[2 * i + 0] = x * Math.cos(theta) - y * Math.sin(theta) + translate_x;
            output[2 * i + 1] = x * Math.sin(theta) + y * Math.cos(theta) + translate_y;
        }
    }

    public void cleanup() {
        _module.cleanup();
    }
}
