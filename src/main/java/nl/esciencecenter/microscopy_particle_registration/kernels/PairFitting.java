package nl.esciencecenter.microscopy_particle_registration.kernels;

import jcuda.CudaException;
import jcuda.Sizeof;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.FunctionBase;
import jsat.math.FunctionVec;
import jsat.math.optimization.BFGS;
import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaMemDouble;
import nl.esciencecenter.rocket.cubaapi.CudaPinned;
import nl.esciencecenter.microscopy_particle_registration.util.FunctionVecBase;
import nl.esciencecenter.rocket.util.Tuple;
import nl.esciencecenter.microscopy_particle_registration.kernels.gausstransform.GaussTransform;
import nl.esciencecenter.microscopy_particle_registration.kernels.transformrigid2d.TransformRigid2D;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.HashMap;
import java.util.function.Function;

public class PairFitting {
    protected static final Logger logger = LogManager.getLogger();
    private double scale;
    private double tolerance;
    private int maxIterations;
    private int maxSize;

    private TransformRigid2D rigidTransform;
    private GaussTransform gaussTransform;
    private CudaPinned gradient;
    private CudaPinned crossTerm;

    public PairFitting(CudaContext context, double scale, double tolerance, int maxIterations, int maxSize) throws IOException, CudaException {
        this.rigidTransform = new TransformRigid2D(context);
        this.gaussTransform = new GaussTransform(context, rigidTransform, maxSize);
        this.gradient = context.allocHostBytes(Sizeof.DOUBLE * 3);
        this.crossTerm = context.allocHostBytes(Sizeof.DOUBLE);
        this.scale = scale;
        this.maxSize = maxSize;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
    }

    private Tuple<Double, Vec> run(
            HashMap<Vec, Tuple<Double, Vec>> history,
            Vec param,
            CudaMemDouble model,
            CudaMemDouble scene
    ) {
        if (!Double.isFinite(param.sum())) {
            return new Tuple(Double.NaN, DenseVector.toDenseVec(Double.NaN, Double.NaN, Double.NaN));
        }

        Tuple<Double, Vec> result = history.get(param);
        if (result != null) {
            return result;
        }

        gaussTransform.applyGPU(
                model,
                scene,
                param.get(0),
                param.get(1),
                param.get(2),
                scale,
                gradient.asCudaMem().asDoubles(),
                crossTerm.asCudaMem().asDoubles());

        double[] g = new double[3];
        gradient.asByteBuffer().asDoubleBuffer().get(g);

        double[] c = new double[1];
        crossTerm.asByteBuffer().asDoubleBuffer().get(c);

        result = new Tuple(c[0], new DenseVector(g));
        history.put(param, result);

        return result;
    }

    public double applyGPU(
            CudaMemDouble model,
            CudaMemDouble scene,
            Vec initialParam,
            Vec outputParam) {
        int m = (int) model.elementCount() / 2;
        int n = (int) scene.elementCount() / 2;

        if (m > maxSize || n > maxSize) {
            throw new IllegalArgumentException("input argument exceeds maximum size");
        }

        HashMap<Vec, Tuple<Double, Vec>> history = new HashMap<>();

        FunctionBase fun = new FunctionBase() {
            @Override
            public double f(Vec param) {
                return run(history, param, model, scene).getFirst() * -1.0;
            }
        };

        FunctionVec dfun = new FunctionVecBase() {
            @Override
            public Vec f(Vec param) {
                return run(history, param, model, scene).getSecond().multiply(-1.0);
            }
        };

        BFGS optimizer = new BFGS();
        optimizer.setMaximumIterations(maxIterations);
        optimizer.optimize(
                tolerance,
                outputParam,
                initialParam,
                fun,
                dfun,
                null);

        return history.get(outputParam).getFirst();
    }

    public void cleanup() {
        rigidTransform.cleanup();
        gaussTransform.cleanup();
        gradient.free();
        crossTerm.free();
    }
}
