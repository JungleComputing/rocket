/*
 * Copyright (c) 2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.esciencecenter.rocket.cubaapi;

import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoD;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoDAsync;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoHAsync;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoDAsync;
import static jcuda.driver.JCudaDriver.cuMemsetD32;
import static jcuda.driver.JCudaDriver.cuMemsetD32Async;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;

public class CudaMem {
    private final CUdeviceptr _deviceptr;
    private final long _sizeInBytes;

    CudaMem(CUdeviceptr ptr, final long size) {
        _deviceptr = ptr;
        _sizeInBytes = size;
    }

    public long sizeInBytes() {
        return _sizeInBytes;
    }

    public CUdeviceptr asDevicePointer() {
        return _deviceptr;
    }

    public CudaMemFloat asFloats() {
        if (sizeInBytes() %  CudaMemFloat.ELEMENT_SIZE != 0) {
            throw new IllegalArgumentException("invalid buffer size");
        }

        return new CudaMemFloat(_deviceptr, _sizeInBytes / CudaMemFloat.ELEMENT_SIZE);
    }

    public CudaMemDouble asDoubles() {
        if (sizeInBytes() %  CudaMemDouble.ELEMENT_SIZE != 0) {
            throw new IllegalArgumentException("invalid buffer size");
        }

        return new CudaMemDouble(_deviceptr, _sizeInBytes / CudaMemDouble.ELEMENT_SIZE);
    }

    public CudaMemInt asInts() {
        if (sizeInBytes() %  CudaMemInt.ELEMENT_SIZE != 0) {
            throw new IllegalArgumentException("invalid buffer size");
        }

        return new CudaMemInt(_deviceptr, _sizeInBytes / CudaMemInt.ELEMENT_SIZE);
    }

    public CudaMemByte asBytes() {
        if (sizeInBytes() %  CudaMemByte.ELEMENT_SIZE != 0) {
            throw new IllegalArgumentException("invalid buffer size");
        }

        return new CudaMemByte(_deviceptr, _sizeInBytes / CudaMemByte.ELEMENT_SIZE);
    }

    public CudaMemLong asLongs() {
        if (sizeInBytes() %  CudaMemLong.ELEMENT_SIZE != 0) {
            throw new IllegalArgumentException("invalid buffer size");
        }

        return new CudaMemLong(_deviceptr, _sizeInBytes / CudaMemLong.ELEMENT_SIZE);
    }

    public void copyFromHost(final Pointer srcHost, final long byteCount) {
        checkBounds(0, byteCount);
    	cuCtxSynchronize();
        cuMemcpyHtoD(_deviceptr, srcHost, byteCount);
    	cuCtxSynchronize();
    }

    public void copyToHost(final Pointer dstHost, final long byteCount) {
        checkBounds(0, byteCount);
        cuCtxSynchronize();
        cuMemcpyDtoH(dstHost, _deviceptr, byteCount);
        cuCtxSynchronize();
    }

    public void copyToDevice(final CUdeviceptr dstDev, final long byteCount) {
        checkBounds(0, byteCount);
        cuCtxSynchronize();
        cuMemcpyDtoD(dstDev, _deviceptr, byteCount);
        cuCtxSynchronize();
    }

    public void copyFromDevice(final CUdeviceptr srcDev, final long byteCount) {
        checkBounds(0, byteCount);
        cuCtxSynchronize();
        cuMemcpyDtoD(_deviceptr, srcDev, byteCount);
        cuCtxSynchronize();
    }

    public void copyFromHostAsync(final Pointer srcHost, final long byteCount, CudaStream stream) {
        checkBounds(0, byteCount);
    	cuMemcpyHtoDAsync(_deviceptr, srcHost, byteCount, stream.cuStream());
    }

    public void copyToHostAsync(final Pointer dstHost, final long byteCount, CudaStream stream) {
        checkBounds(0, byteCount);
    	cuMemcpyDtoHAsync(dstHost, _deviceptr, byteCount, stream.cuStream());
    }

    public void copyToDeviceAsync(final CUdeviceptr dstDev, long byteCount, CudaStream stream) {
        checkBounds(0, byteCount);
        cuMemcpyDtoDAsync(dstDev, _deviceptr, byteCount, stream.cuStream());
    }

    public void copyFromDeviceAsync(final CUdeviceptr srcDev, long byteCount, CudaStream stream) {
        checkBounds(0, byteCount);
        cuMemcpyDtoDAsync(_deviceptr, srcDev, byteCount, stream.cuStream());
    }

    private void checkBounds(final long byteOffset, final long byteCount) {
        if (byteOffset + byteCount > _sizeInBytes){
            throw new IllegalArgumentException("indices " + byteOffset + " to " + (byteOffset + byteCount) +
                    " is out of range for buffer of size " + _sizeInBytes);
        }
    }
    
    protected void memsetD32(final int ui) {
        cuMemsetD32(_deviceptr, ui, _sizeInBytes / Sizeof.INT);
    }

    protected void memsetD32Async(final int ui, CudaStream stream) {
        cuMemsetD32Async(_deviceptr, ui, _sizeInBytes / Sizeof.INT, stream.cuStream());
    }

    public CudaMem sliceBytes(long offset, long length) {
        checkBounds(0, length);
        return new CudaMem(_deviceptr.withByteOffset(offset), length);
    }

    public void free() {
        cuMemFree(_deviceptr);
    }

    @Override
    public String toString() {
        return _deviceptr.toString();
    }
}
