/*
 * Copyright (c) 2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.esciencecenter.rocket.cubaapi;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;

public final class CudaMemLong extends CudaMem {
    protected static final long ELEMENT_SIZE = Sizeof.LONG;

    CudaMemLong(CUdeviceptr ptr, long elementCount) {
        super(ptr, ELEMENT_SIZE * elementCount);
    }

    public long elementCount() {
        return sizeInBytes() / ELEMENT_SIZE;
    }

    public void copyToHost(final long[] dst, final long elementCount) {
        super.copyToHost(Pointer.to(dst), (ELEMENT_SIZE * elementCount));
    }

    public void copyFromHost(final long[] data, final long elementCount) {
        super.copyFromHost(Pointer.to(data), (ELEMENT_SIZE * elementCount));
    }
    
    public void copyFromHostAsync(final long[] src, CudaStream stream) {
        super.copyFromHostAsync(Pointer.to(src), (ELEMENT_SIZE * src.length), stream);
    }

    public void copyToHostAsync(final long[] dst, CudaStream stream) {
        super.copyToHostAsync(Pointer.to(dst), (ELEMENT_SIZE * dst.length), stream);
    }

    public void copyToDevice(final CudaMemLong mem) {
        super.copyToDevice(mem.asDevicePointer(), mem.sizeInBytes());
    }

    public void copyFromDevice(final CudaMemLong mem) {
        super.copyFromDevice(mem.asDevicePointer(), mem.sizeInBytes());
    }

    public void copyToDeviceAsync(final CudaMemLong mem, CudaStream stream) {
        super.copyToDeviceAsync(mem.asDevicePointer(), mem.sizeInBytes(), stream);
    }

    public void copyFromDeviceAsync(final CudaMemLong mem, CudaStream stream) {
        super.copyFromDeviceAsync(mem.asDevicePointer(), mem.sizeInBytes(), stream);
    }

    public CudaMemLong slice(long offset) {
        return slice(offset, elementCount() - offset);
    }

    public CudaMemLong slice(long offset, long length) {
        return sliceBytes(offset * ELEMENT_SIZE, length * ELEMENT_SIZE).asLongs();
    }
}