/*
 * Copyright (c) 2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.esciencecenter.rocket.cubaapi;

import static jcuda.driver.JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP;
import static jcuda.driver.JCudaDriver.CU_MEMHOSTALLOC_PORTABLE;
import static jcuda.driver.JCudaDriver.cuMemFreeHost;
import static jcuda.driver.JCudaDriver.cuMemHostAlloc;
import static jcuda.driver.JCudaDriver.cuMemHostGetDevicePointer;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoHAsync;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoDAsync;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;

import java.nio.ByteBuffer;

public class CudaPinned {
    private final Pointer _hostptr;
    private final long _sizeInByte;
    private final int _flags;

    CudaPinned(Pointer hostptr, long bytesize, int flags) {
        _hostptr = hostptr;
        _sizeInByte = bytesize;
        _flags = flags;
    }

    public long sizeInBytes() {
        return _sizeInByte;
    }

    public Pointer asPointer() {
        return _hostptr;
    }

    public CudaMem asCudaMem() {
        return new CudaMem(asDevicePointer(), _sizeInByte);
    }

    public ByteBuffer asByteBuffer() {
        return _hostptr.getByteBuffer(0, _sizeInByte);
    }

    public CUdeviceptr asDevicePointer() {
        CUdeviceptr d_ptr = new CUdeviceptr();
        cuMemHostGetDevicePointer(d_ptr, _hostptr, 0);
        return d_ptr;
    }

    public void copyToDevice(final CudaMem dstDev) {
        copyToDevice(dstDev, dstDev.sizeInBytes());
    }

    public void copyToDevice(final CudaMem dstDev, final long byteCount) {
        copyToDevice(dstDev.asDevicePointer(), byteCount);
    }

    public void copyToDevice(final CUdeviceptr dstDev, final long byteCount) {
        cuCtxSynchronize();
        cuMemcpyHtoD(dstDev, _hostptr, byteCount);
        cuCtxSynchronize();
    }

    public void copyFromDevice(final CudaMem srcDev) {
        copyFromDevice(srcDev.asDevicePointer(), srcDev.sizeInBytes());
    }

    public void copyFromDevice(final CudaMem srcDev, final long byteCount) {
        copyFromDevice(srcDev.asDevicePointer(), byteCount);
    }

    public void copyFromDevice(final CUdeviceptr srcDev, final long byteCount) {
        cuCtxSynchronize();
        cuMemcpyDtoH(_hostptr, srcDev, byteCount);
        cuCtxSynchronize();
    }

    public void copyToDeviceAsync(final CudaMem srcDev, final long byteCount, CudaStream stream) {
        copyToDeviceAsync(srcDev.asDevicePointer(), byteCount, stream);
    }

    public void copyToDeviceAsync(final CUdeviceptr srcDev, final long byteCount, CudaStream stream) {
        cuMemcpyHtoDAsync(srcDev, _hostptr, byteCount, stream.cuStream());
    }

    public void copyFromDeviceAsync(final CudaMem dstDev, final long byteCount, CudaStream stream) {
        copyFromDeviceAsync(dstDev.asDevicePointer(), byteCount, stream);
    }

    public void copyFromDeviceAsync(final CUdeviceptr dstDev, final long byteCount, CudaStream stream) {
        cuMemcpyDtoHAsync(_hostptr, dstDev, byteCount, stream.cuStream());
    }

    public CudaPinned sliceBytes(long offset, long length) {
        checkBounds(offset, length);
        return new CudaPinned(_hostptr.withByteOffset(offset), length, _flags);
    }

    public void free() {
        cuMemFreeHost(_hostptr);
    }

    public boolean isPortable() {
        return (_flags & CU_MEMHOSTALLOC_PORTABLE) != 0;
    }

    public boolean isDeviceMapped() {
        return (_flags & CU_MEMHOSTALLOC_DEVICEMAP) != 0;
    }

    private void checkBounds(final long byteOffset, final long byteCount) {
        if (byteOffset + byteCount > _sizeInByte){
            throw new IllegalArgumentException("indices " + byteOffset + " to " + (byteOffset + byteCount) +
                    " is out of range for buffer of size " + _sizeInByte);
        }
    }

    @Override
    public String toString() {
        return _hostptr.toString();
    }
}
