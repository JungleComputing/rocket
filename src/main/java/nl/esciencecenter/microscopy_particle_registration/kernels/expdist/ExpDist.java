package nl.esciencecenter.microscopy_particle_registration.kernels.expdist;

import jcuda.CudaException;
import jcuda.Sizeof;
import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaFunction;
import nl.esciencecenter.rocket.cubaapi.CudaMemDouble;
import nl.esciencecenter.rocket.cubaapi.CudaModule;
import nl.esciencecenter.rocket.cubaapi.CudaPinned;
import nl.esciencecenter.rocket.cubaapi.CudaStream;
import nl.esciencecenter.microscopy_particle_registration.kernels.transformrigid2d.TransformRigid2D;

import java.io.IOException;

public class ExpDist {
    static final private int blockSizeX = 32;
    static final private int blockSizeY = 4;
    static final private int tileSizeX = 4;
    static final private int tileSizeY = 2;
    static final private int dim = 2;

    //cuda handles
    protected CudaContext _context;
    protected CudaStream _stream;
    protected CudaModule _module;

    protected CudaFunction _expDist;
    protected CudaFunction _reduceCrossTerm;
    protected TransformRigid2D _transformKernel;
    protected CudaMemDouble _d_transformedModel;

    protected int maxSize;
    protected CudaMemDouble _d_crossTerm;
    protected CudaPinned _h_crossTerm;

    public ExpDist(CudaContext context, int maxSize) throws CudaException, IOException {
        _transformKernel = new TransformRigid2D(context);
        _context = context;
        _stream = context.createStream();
        _module = context.compileModule(getClass().getResource("kernels.cu"));

        _expDist = _module.getFunction("ExpDist");
        _reduceCrossTerm = _module.getFunction("reduce_cross_term");
        _d_transformedModel = context.allocDoubles(maxSize * dim); // dim=2

        int n = ceilDiv(maxSize, blockSizeX * tileSizeX) * ceilDiv(maxSize, blockSizeY * tileSizeY);
        _d_crossTerm = context.allocDoubles(n); // dim=2

        _h_crossTerm = context.allocHostBytes(Sizeof.DOUBLE);

        this.maxSize = maxSize;
    }

    static private int ceilDiv(int a, int b) {
        return (a / b) + (a % b == 0 ? 0 : 1);
    }

    public double applyGPU(
            CudaMemDouble a,
            CudaMemDouble b,
            double translate_x,
            double translate_y,
            double rotation,
            CudaMemDouble scaleA,
            CudaMemDouble scaleB
    ) {
        int m = (int)a.elementCount() / dim;
        int n = (int)b.elementCount() / dim;

        if (n > maxSize || m > maxSize ||
                a.elementCount() != m * dim ||
                b.elementCount() != n * dim ||
                scaleA.elementCount() != m ||
                scaleB.elementCount() != n) {
            throw new IllegalArgumentException("invalid element count");
        }

        _transformKernel.applyGPU(
                _stream,
                a,
                _d_transformedModel,
                m,
                rotation,
                translate_x,
                translate_y);

        int gridX = ceilDiv(m, blockSizeX * tileSizeX);
        int gridY = ceilDiv(n, blockSizeY * tileSizeY);

        _expDist.setDim(gridX, gridY, blockSizeX, blockSizeY);
        _expDist.launch(_stream,
                _d_transformedModel,
                b,
                m,
                n,
                scaleA,
                scaleB,
                _d_crossTerm);

        _reduceCrossTerm.setDim(1, blockSizeX);
        _reduceCrossTerm.launch(
                _stream,
                _h_crossTerm.asCudaMem(),
                _d_crossTerm,
                m,
                n,
                gridX * gridY);

        _stream.synchronize();

        return _h_crossTerm.asByteBuffer().asDoubleBuffer().get(0);
    }

    public double applyCPU(
            double[] a,
            double[] b,
            double translate_x,
            double translate_y,
            double rotation,
            double[] scaleA,
            double[] scaleB
    ) {
        int m = (int)a.length / dim;
        int n = (int)b.length / dim;

        double[] transformed = new double[dim * m];
        _transformKernel.applyCPU(a, transformed, m, rotation, translate_x, translate_y);

        double cross_term = 0;

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                double dist_ij = 0;

                for (int d = 0; d < dim; ++d) {
                    dist_ij += Math.pow(transformed[dim * i + d] - b[dim * j + d], 2);
                }

                cross_term += Math.exp(-dist_ij / (scaleA[i] + scaleB[j]));
            }
        }

        return cross_term;
    }


    /**
     * Cleans up GPU memory
     */
    public void cleanup() {
        _transformKernel.cleanup();
        _stream.destroy();
        _d_crossTerm.free();
        _d_transformedModel.free();
        _h_crossTerm.free();
    }

}
