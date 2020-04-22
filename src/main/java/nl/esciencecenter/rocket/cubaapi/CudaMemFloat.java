/*
 * Copyright (c) 2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.esciencecenter.rocket.cubaapi;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;

public final class CudaMemFloat extends CudaMem {
    protected static final long ELEMENT_SIZE = Sizeof.FLOAT;

    CudaMemFloat(CUdeviceptr ptr, long elementCount) {
        super(ptr, ELEMENT_SIZE * elementCount);
    }

    public long elementCount() {
        return sizeInBytes() / ELEMENT_SIZE;
    }

    public void copyFromHost(final float[] src, final int elementCount) {
        super.copyFromHost(Pointer.to(src), (ELEMENT_SIZE * elementCount));
    }

    public void copyToHost(final float[] dst, final int elementCount) {
        super.copyToHost(Pointer.to(dst), (ELEMENT_SIZE * elementCount));
    }

    public void copyFromHost(final float[] src) {
        copyFromHost(src, src.length);
    }

    public void copyToHost(final float[] dst) {
        copyToHost(dst, dst.length);
    }
    
    public void copyFromHostAsync(final float[] src, CudaStream stream) {
        super.copyFromHostAsync(Pointer.to(src), (ELEMENT_SIZE * src.length), stream);
    }

    public void copyToHostAsync(final float[] dst, CudaStream stream) {
        super.copyToHostAsync(Pointer.to(dst), (ELEMENT_SIZE * dst.length), stream);
    }

    public void copyToDevice(final CudaMemFloat mem) {
        super.copyToDevice(mem.asDevicePointer(), mem.sizeInBytes());
    }

    public void copyFromDevice(final CudaMemFloat mem) {
        super.copyFromDevice(mem.asDevicePointer(), mem.sizeInBytes());
    }

    public void copyToDeviceAsync(final CudaMemFloat mem, CudaStream stream) {
        super.copyToDeviceAsync(mem.asDevicePointer(), mem.sizeInBytes(), stream);
    }

    public void copyFromDeviceAsync(final CudaMemFloat mem, CudaStream stream) {
        super.copyFromDeviceAsync(mem.asDevicePointer(), mem.sizeInBytes(), stream);
    }

    public void fill(float val) {
        memsetD32(Float.floatToIntBits(val));
    }

    public void fillAsync(float val, CudaStream stream) {
        memsetD32Async(Float.floatToIntBits(val), stream);
    }

    public CudaMemFloat slice(long offset) {
        return slice(offset, elementCount() - offset);
    }

    public CudaMemFloat slice(long offset, long length) {
        return sliceBytes(offset * Sizeof.FLOAT, length * Sizeof.FLOAT).asFloats();
    }
}
