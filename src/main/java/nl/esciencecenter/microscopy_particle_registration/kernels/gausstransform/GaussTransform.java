package nl.esciencecenter.microscopy_particle_registration.kernels.gausstransform;

import jcuda.CudaException;
import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaFunction;
import nl.esciencecenter.rocket.cubaapi.CudaMemDouble;
import nl.esciencecenter.rocket.cubaapi.CudaModule;
import nl.esciencecenter.rocket.cubaapi.CudaStream;
import nl.esciencecenter.microscopy_particle_registration.kernels.transformrigid2d.TransformRigid2D;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;

public class GaussTransform {
    protected static final Logger logger = LogManager.getLogger();

    //cuda handles
    protected CudaContext _context;
    protected CudaStream _stream;
    protected CudaModule _module;

    protected CudaFunction _gaussTransform;
    protected CudaFunction _reduceTerms;

    final protected int dim = 2;
    protected int maxSize;
    protected CudaMemDouble _d_crossTermPartials;
    protected CudaMemDouble _d_gradientPartials;
    protected CudaMemDouble _d_transformedModel;
    protected TransformRigid2D _transformKernel;

    public GaussTransform(CudaContext context, TransformRigid2D transformKernel, int maxSize) throws CudaException, IOException {
        _transformKernel = transformKernel;
        _context = context;
        _stream = context.createStream();
        _module = context.compileModule(getClass().getResource("kernels.cu"));

        _gaussTransform = _module.getFunction("GaussTransform");
        _reduceTerms = _module.getFunction("reduce_terms");

        _d_crossTermPartials = context.allocDoubles(maxSize);
        _d_gradientPartials = context.allocDoubles(maxSize * dim); // dim=2
        _d_transformedModel = context.allocDoubles(maxSize * dim); // dim=2

        this.maxSize = maxSize;
    }

    public void applyGPU(
            CudaMemDouble model,
            CudaMemDouble scene,
            double translate_x,
            double translate_y,
            double rotation,
            double scale,
            CudaMemDouble gradient,
            CudaMemDouble crossTerm
    ) {
        int m = (int)model.elementCount() / 2;
        int n = (int)scene.elementCount() / 2;

        if (n > maxSize || m > maxSize ||
                model.elementCount() < m * dim ||
                scene.elementCount() < n * dim ||
                crossTerm.elementCount() < 1 ||
                gradient.elementCount() < 3) {
            throw new IllegalArgumentException("invalid element count");
        }

        _transformKernel.applyGPU(
                _stream,
                model,
                _d_transformedModel,
                m,
                rotation,
                translate_x,
                translate_y);

        double scaleSq = scale * scale;

        _gaussTransform.setDim(m, 128);
        _gaussTransform.launch(
                _stream,
                _d_transformedModel,
                scene,
                m,
                n,
                scaleSq,
                _d_gradientPartials,
                _d_crossTermPartials);

        _reduceTerms.setDim(1, 128);
        _reduceTerms.launch(
                _stream,
                crossTerm,
                gradient,
                m,
                n,
                _d_crossTermPartials,
                _d_gradientPartials,
                model,
                rotation);

        _stream.synchronize();
    }

    public double applyCPU(
            double[] model,
            double[] scene,
            double translate_x,
            double translate_y,
            double rotation,
            double scale,
            double[] grad
    ) {
        int m = model.length / 2;
        int n = scene.length / 2;

        double crossTerm = 0.0;
        double scaleSq = scale * scale;
        double[] gradient = new double[m * dim];
        double[] transformedModel = new double[m * dim];

        _transformKernel.applyCPU(model, transformedModel, m, rotation, translate_x, translate_y);

        // Calculate component-wise gradients
        for (int i = 0; i < m; ++i) {
            double partialCostTerm = 0.0;

            for (int j = 0; j < n; ++j) {
                double dist_ij = 0;

                for (int d = 0; d < dim; ++d) {
                    double a = transformedModel[i * dim + d];
                    double b = scene[j * dim + d];

                    dist_ij +=  Math.pow(a - b, 2);
                }

                double cost_ij = Math.exp(-1.0 * dist_ij / scaleSq);

                for (int d = 0; d < dim; ++d){
                    double a = transformedModel[i * dim + d];
                    double b = scene[j * dim + d];

                    gradient[i * dim + d] -= cost_ij * 2.0 * (a - b);
                }

                partialCostTerm += cost_ij;
            }

            crossTerm += partialCostTerm;
        }

        // Normalize
        crossTerm /= (m * n);

        for (int i = 0; i < m * dim; ++i) {
            gradient[i] /= (scaleSq * m * n);
        }

        // Calculate gradient along axes
        for (int d = 0; d < dim; ++d) {
            grad[d] = 0;

            for (int i = 0; i < m; i++) {
                grad[d] += gradient[i * dim + d];
            }
        }

        // Calculate gradient for rotation
        double[][] mat = new double[2][2];
        for (int d = 0; d < dim; ++d) {
            for (int e = 0; e < dim; ++e) {
                for (int i = 0; i < m; i++) {
                    mat[d][e] += gradient[i * dim + d] * model[i * dim + e];
                }
            }
        }

        grad[dim] = mat[0][0] * -Math.sin(rotation) + mat[0][1] * -Math.cos(rotation)
                + mat[1][0] * Math.cos(rotation) + mat[1][1] * -Math.sin(rotation);

        return crossTerm;
    }


    /**
     * Cleans up GPU memory
     */
    public void cleanup() {
        _stream.destroy();
        _module.cleanup();
        _d_crossTermPartials.free();
        _d_gradientPartials.free();
        _d_transformedModel.free();
    }

}
