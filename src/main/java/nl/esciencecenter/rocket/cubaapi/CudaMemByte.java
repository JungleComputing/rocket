/*
 * Copyright (c) 2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.esciencecenter.rocket.cubaapi;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;

public final class CudaMemByte extends CudaMem {
    protected static final long ELEMENT_SIZE = Sizeof.BYTE;

    CudaMemByte(CUdeviceptr ptr, long elementCount) {
        super(ptr, ELEMENT_SIZE * elementCount);
    }

    public long elementCount() {
        return sizeInBytes() / ELEMENT_SIZE;
    }

    public void copyFromHost(final byte[] data) {
        super.copyFromHost(Pointer.to(data), (ELEMENT_SIZE * data.length));
    }

    public void copyToHost(final byte[] dst) {
        super.copyToHost(Pointer.to(dst), ELEMENT_SIZE * dst.length);
    }

    public void copyFromHostAsync(final byte[] src, CudaStream stream) {
        super.copyFromHostAsync(Pointer.to(src), (ELEMENT_SIZE * src.length), stream);
    }

    public void copyToHostAsync(final byte[] dst, CudaStream stream) {
        super.copyToHostAsync(Pointer.to(dst), (ELEMENT_SIZE * dst.length), stream);
    }

    public void copyToDevice(final CudaMemByte mem) {
        super.copyToDevice(mem.asDevicePointer(), mem.sizeInBytes());
    }

    public void copyFromDevice(final CudaMemByte mem) {
        super.copyFromDevice(mem.asDevicePointer(), mem.sizeInBytes());
    }

    public void copyToDeviceAsync(final CudaMemByte mem, CudaStream stream) {
        super.copyToDeviceAsync(mem.asDevicePointer(), mem.sizeInBytes(), stream);
    }

    public void copyFromDeviceAsync(final CudaMemByte mem, CudaStream stream) {
        super.copyFromDeviceAsync(mem.asDevicePointer(), mem.sizeInBytes(), stream);
    }

    public CudaMemByte slice(long offset) {
        return slice(offset, elementCount() - offset);
    }

    public CudaMemByte slice(long offset, long length) {
        return sliceBytes(offset * Sizeof.BYTE, length * Sizeof.BYTE).asBytes();
    }
}