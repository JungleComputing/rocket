/*
 * Copyright (c) 2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.esciencecenter.rocket.cubaapi;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;

public final class CudaMemDouble extends CudaMem {
    protected static final long ELEMENT_SIZE = Sizeof.DOUBLE;

    CudaMemDouble(CUdeviceptr ptr, long elementCount) {
        super(ptr, ELEMENT_SIZE * elementCount);
    }

    public long elementCount() {
        return sizeInBytes() / ELEMENT_SIZE;
    }

    public void copyFromHost(final double[] src) {
        copyFromHost(src, src.length);
    }

    public void copyToHost(final double[] dst) {
        copyToHost(dst, dst.length);
    }

    public void copyFromHost(final double[] src, final int elementCount) {
        super.copyFromHost(Pointer.to(src), (ELEMENT_SIZE * elementCount));
    }

    public void copyFromHostAsync(final double[] src, final int elementCount) {
        super.copyFromHost(Pointer.to(src), (ELEMENT_SIZE * elementCount));
    }

    public void copyToHost(final double[] dst, final int elementCount) {
        super.copyToHost(Pointer.to(dst), (ELEMENT_SIZE * elementCount));
    }
    
    public void copyFromHostAsync(final double[] src, CudaStream stream) {
        super.copyFromHostAsync(Pointer.to(src), (ELEMENT_SIZE * src.length), stream);
    }

    public void copyToHostAsync(final double[] dst, CudaStream stream) {
        super.copyToHostAsync(Pointer.to(dst), (ELEMENT_SIZE * dst.length), stream);
    }

    public void copyToDevice(final CudaMemDouble mem) {
        super.copyToDevice(mem.asDevicePointer(), mem.sizeInBytes());
    }

    public void copyFromDevice(final CudaMemDouble mem) {
        super.copyFromDevice(mem.asDevicePointer(), mem.sizeInBytes());
    }

    public void copyToDeviceAsync(final CudaMemDouble mem, CudaStream stream) {
        super.copyToDeviceAsync(mem.asDevicePointer(), mem.sizeInBytes(), stream);
    }

    public void copyFromDeviceAsync(final CudaMemDouble mem, CudaStream stream) {
        super.copyFromDeviceAsync(mem.asDevicePointer(), mem.sizeInBytes(), stream);
    }

    public CudaMemDouble slice(long offset) {
        return slice(offset, elementCount() - offset);
    }

    public CudaMemDouble slice(long offset, long length) {
        return sliceBytes(offset * Sizeof.DOUBLE, length * Sizeof.DOUBLE).asDoubles();
    }
}