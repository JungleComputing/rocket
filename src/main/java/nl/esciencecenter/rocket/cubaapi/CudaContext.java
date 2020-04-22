/*
 * Copyright (c) 2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.esciencecenter.rocket.cubaapi;

import static jcuda.driver.JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP;
import static jcuda.driver.JCudaDriver.CU_MEMHOSTALLOC_PORTABLE;
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuCtxPopCurrent;
import static jcuda.driver.JCudaDriver.cuCtxPushCurrent;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuGetErrorName;
import static jcuda.driver.JCudaDriver.cuGetErrorString;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemHostAlloc;
import static jcuda.runtime.cudaError.cudaSuccess;

import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.cudaError;
import org.apache.commons.io.IOUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.function.Supplier;

public final class CudaContext {
    protected static final Logger logger = LogManager.getLogger();
    private final CUcontext _context;
    CudaDevice device;

    CudaContext(final CUdevice device, CudaDevice cudaDev) {
        _context = new CUcontext();
        this.device = cudaDev;

        cuCtxCreate(_context, 0, device);
    }

    public void with(Runnable fn) {
        withSupplier(() -> {
            fn.run();
            return null;
        });
    }

    public <T> T withSupplier(Supplier<T> fn) {
        cuCtxPushCurrent(_context);

        try {
            return fn.get();
        } finally {
            CUcontext unused = new CUcontext();
            cuCtxPopCurrent(unused);
        }
    }

    public CudaStream createStream() {
        return withSupplier(() -> new CudaStream());
    }

    public CudaEvent createEvent() {
        return withSupplier(() -> new CudaEvent());
    }

    public CudaModule compileModule(URL url, String... options) throws IOException {
        String source = IOUtils.toString(url);
        return compileModule(source, options);
    }

    public CudaModule compileModule(String source, String... options) {
        int cc[] = getDevice().getMajorMinor();
        String architecture = "compute_" + cc[0] + "" + cc[1];
        String capability = "sm_" + cc[0] + "" + cc[1];

        int n = options.length;
        options = Arrays.copyOf(options, n + 2);
        options[n + 0] = "-gencode=arch=" + architecture + ",code=" + capability;
        options[n + 1] = "-O3";

        //compile the CUDA code to run on the GPU
        return loadModule(source, options);
    }

    public CudaModule loadModule(final String cuSource, final String... nvccOptions) {
        return new CudaModule(cuSource, nvccOptions);
    }

    public CudaDevice getDevice() {
        return this.device;
    }

    public long getFreeMemory() {
        long[] free = new long[1];
        with(() -> {
            JCudaDriver.cuMemGetInfo(free, new long[1]);
        });
        return free[0];
    }

    public long getTotalMemory() {
        long[] total = new long[1];

        with(() -> {
            JCudaDriver.cuMemGetInfo(new long[1], total);
        });

        return total[0];
    }

    public void synchronize() {
        cuCtxSynchronize();
    }

    public void destroy() {
        cuCtxDestroy(_context);
    }

    public CudaPinned allocHostBytes(final long elementCount) {
        return allocHostBytes(elementCount, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP);
    }

    public CudaPinned allocHostBytes(final long elementCount, int flags) {
        return withSupplier(() -> {
            Pointer ptr = new Pointer();
            int err = cuMemHostAlloc(ptr, elementCount, flags);

            return new CudaPinned(ptr, elementCount, flags);
        });
    }

    public CudaMemByte allocBytes(final long elementCount) {
        return withSupplier(() -> {
            CUdeviceptr ptr = new CUdeviceptr();
            int err = cuMemAlloc(ptr, elementCount);

            return new CudaMemByte(ptr, elementCount);
        });
    }

    public CudaMemByte allocBytes(final byte[] data) {
        final CudaMemByte mem = allocBytes(data.length);
        mem.copyFromHost(data);
        return mem;
    }

    public CudaMemInt allocInts(final long elementCount) {
        return allocBytes(elementCount * CudaMemInt.ELEMENT_SIZE).asInts();
    }

    public CudaMemInt allocInts(final int[] data) {
        final CudaMemInt mem = allocInts(data.length);
        mem.copyFromHost(data);
        return mem;
    }

    public CudaMemFloat allocFloats(final long elementCount) {
        return allocBytes(elementCount * CudaMemFloat.ELEMENT_SIZE).asFloats();
    }

    public CudaMemFloat allocFloats(final float[] data) {
        final CudaMemFloat mem = allocFloats(data.length);
        mem.copyFromHost(data);
        return mem;
    }

    public CudaMemDouble allocDoubles(final long elementCount) {
        return allocBytes(elementCount * CudaMemDouble.ELEMENT_SIZE).asDoubles();
    }

    public CudaMemLong allocLongs(final long elementCount) {
        return allocBytes(elementCount * CudaMemLong.ELEMENT_SIZE).asLongs();
    }

    @Override
    public String toString() {
        return _context.toString();
    }

}
