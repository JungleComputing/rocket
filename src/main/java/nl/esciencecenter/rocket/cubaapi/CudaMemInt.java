/*
 * Copyright (c) 2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.esciencecenter.rocket.cubaapi;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;

public final class CudaMemInt extends CudaMem {
    protected static final long ELEMENT_SIZE = Sizeof.INT;

    CudaMemInt(CUdeviceptr ptr, long elementCount) {
        super(ptr, ELEMENT_SIZE * elementCount);
    }

    public long elementCount() {
        return sizeInBytes() / ELEMENT_SIZE;
    }

    public void copyToHost(final int[] dst, final int elementCount) {
        super.copyToHost(Pointer.to(dst), (ELEMENT_SIZE * elementCount));
    }

    public void copyFromHost(final int[] data, final int elementCount) {
        super.copyFromHost(Pointer.to(data), (ELEMENT_SIZE * elementCount));
    }
    public void copyToHost(final int[] dst) {
        copyToHost(dst, dst.length);
    }

    public void copyFromHost(final int[] data) {
        copyFromHost(data, data.length);
    }
    
    public void copyFromHostAsync(final int[] src, CudaStream stream) {
        super.copyFromHostAsync(Pointer.to(src), (ELEMENT_SIZE * src.length), stream);
    }

    public void copyToHostAsync(final int[] dst, CudaStream stream) {
        super.copyToHostAsync(Pointer.to(dst), (ELEMENT_SIZE * dst.length), stream);
    }

    public void copyToDevice(final CudaMemInt mem) {
        super.copyToDevice(mem.asDevicePointer(), mem.sizeInBytes());
    }

    public void copyFromDevice(final CudaMemInt mem) {
        super.copyFromDevice(mem.asDevicePointer(), mem.sizeInBytes());
    }

    public void copyToDeviceAsync(final CudaMemInt mem, CudaStream stream) {
        super.copyToDeviceAsync(mem.asDevicePointer(), mem.sizeInBytes(), stream);
    }

    public void copyFromDeviceAsync(final CudaMemInt mem, CudaStream stream) {
        super.copyFromDeviceAsync(mem.asDevicePointer(), mem.sizeInBytes(), stream);
    }

    public void fill(int val) {
        memsetD32(val);
    }

    public void fillAsync(int val, CudaStream stream) {
        memsetD32Async(val, stream);
    }

    public CudaMemInt slice(long offset) {
        return slice(offset, elementCount() - offset);
    }

    public CudaMemInt slice(long offset, long length) {
        return sliceBytes(offset * ELEMENT_SIZE, length * ELEMENT_SIZE).asInts();
    }
}