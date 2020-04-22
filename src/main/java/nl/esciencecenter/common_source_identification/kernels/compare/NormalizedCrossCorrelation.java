/*
 * Copyright (c) 2012-2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.esciencecenter.common_source_identification.kernels.compare;

import nl.esciencecenter.rocket.cubaapi.*;
import jcuda.*;
import jcuda.runtime.JCuda;
import jcuda.driver.*;

import java.io.IOException;


/**
 * This class is performs a Normalized Cross Correlation on the GPU
 */
public class NormalizedCrossCorrelation implements PatternComparator {
    private static int threads = 256;
    private static int reducing_thread_blocks = 1024; //optimally this equals the number of SMs in the GPU

    //cuda handles
    protected CudaModule _module;
    protected CudaContext _context;
    protected CudaStream _stream;

    //handles to CUDA kernels
    protected CudaFunction _computeSums;
    protected CudaFunction _computeNCC;

    // CUDA memory
    protected CudaMemDouble _partialXX;
    protected CudaMemDouble _partialX;
    protected CudaMemDouble _partialYY;
    protected CudaMemDouble _partialY;
    protected CudaMemDouble _partialXY;

    //PRNU pattern dimensions
    int h;
    int w;

    /**
     * Constructor for the Normalized Cross Correlation GPU implementation
     *
     * @param h - the image height in pixels
     * @param w - the image width in pixels
     * @param context   - the CudaContext as created by the factory
     */
    public NormalizedCrossCorrelation(int h, int w, CudaContext context, String... compileArgs)
            throws CudaException, IOException {
        JCudaDriver.setExceptionsEnabled(true);
        _module = context.compileModule(getClass().getResource("normalizedcrosscorrelation.cu"), compileArgs);
        _context = context;
        _stream = new CudaStream();
        this.h = h;
        this.w = w;

        //setup CUDA functions
        JCudaDriver.setExceptionsEnabled(true);

        _partialX = context.allocDoubles(reducing_thread_blocks);
        _partialXX = context.allocDoubles(reducing_thread_blocks);
        _partialY = context.allocDoubles(reducing_thread_blocks);
        _partialYY = context.allocDoubles(reducing_thread_blocks);
        _partialXY = context.allocDoubles(reducing_thread_blocks);

        _computeSums = _module.getFunction("computeSums");
        _computeSums.setDim(reducing_thread_blocks, threads);

        _computeNCC = _module.getFunction("computeNCC");
        _computeNCC.setDim(1, threads);
    }

    public long getInputSize() {
        return Sizeof.FLOAT * h * w;
    }

    public long getOutputSize() {
        return  Sizeof.DOUBLE;
    }

    public void applyGPU(CudaMem left, CudaMem right, CudaMem result) {
        if (left.sizeInBytes() < getInputSize() ||
                right.sizeInBytes() < getInputSize() ||
                result.sizeInBytes() < getOutputSize()) {
            throw new IllegalArgumentException();
        }

        //call the kernel
        try {
            _computeSums.launch(
                    _stream,
                    w * h,
                    left,
                    right,
                    _partialXX,
                    _partialX,
                    _partialYY,
                    _partialY,
                    _partialXY);

            _computeNCC.launch(
                    _stream,
                    reducing_thread_blocks,
                    _partialXX,
                    _partialX,
                    _partialYY,
                    _partialY,
                    _partialXY,
                    result);
        } finally {
            _stream.synchronize();
        }
    }


    /**
     * This method performs an array of comparisons between patterns on the CPU
     * It computes the NCC scores between pattern x and y
     *
     * @param x         a PRNU patterns stored as a float array
     * @param sumsq_x   the sum squared of pattern x
     * @param y         a PRNU patterns stored as a float array
     * @param sumsq_y   the sum squared of pattern y
     * @returns         the NCC scores from comparing patterns x and y
     */
    public static double applyCPU(final float[] x, double sumsq_x, final float[] y, double sumsq_y) {
	    double sum_xy = 0.0;
        for (int i=0; i<x.length; i++) {
            sum_xy += x[i] * y[i];
        }
    	return (sum_xy / Math.sqrt(sumsq_x * sumsq_y));
    }


    /**
     * Cleans up GPU memory 
     */
    public void cleanup() {
        _module.cleanup();
        _stream.destroy();
    }


}
