/*
 * Copyright (c) 2013-2014, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.esciencecenter.rocket.cubaapi;

import static jcuda.driver.CUresult.CUDA_SUCCESS;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuDeviceGetCount;
import static jcuda.driver.JCudaDriver.cuDeviceGetName;
import static jcuda.driver.JCudaDriver.cuInit;

import jcuda.driver.CUdevice;
import jcuda.driver.CUdevice_attribute;
import jcuda.driver.CUdevprop;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;

public final class CudaDevice {
    private final CUdevice _device;
    int device_id;

    CudaDevice(final int ordinal) {
        _device = new CUdevice();
        this.device_id = ordinal;

        cuDeviceGet(_device, ordinal);
    }

    public CudaContext createContext() {
        return new CudaContext(_device, this);
    }

    public void setSharedMemConfig(final int config) {
        JCuda.cudaDeviceSetSharedMemConfig(config);
    }

    public static CudaDevice[] getDevices() {
        initialize();

        final int[] count = new int[1];
        cuDeviceGetCount(count);

        final CudaDevice[] devices = new CudaDevice[count[0]];
        for (int i = 0; i < devices.length; i++) {
            devices[i] = new CudaDevice(i);
        }
        return devices;
    }

    public static CudaDevice getBestDevice() {
        initialize();

        final int[] count = new int[1];
        cuDeviceGetCount(count);

        //selecting a GPU based on the largest number of SMs in all GPUs
        //probably not perfect, but high-end cards tend to have more SMs
        int max_SM = 0;
        int max_ind = -1;
        for (int i = 0; i < count[0]; i++) {
            cudaDeviceProp prop = new cudaDeviceProp();
            JCuda.cudaGetDeviceProperties(prop, i);
            int SMs = prop.multiProcessorCount;
            if (max_SM < SMs) {
                max_SM = SMs;
                max_ind = i;
            }
        }

        return new CudaDevice(max_ind);
    }

    private static void initialize() {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
    }

    public int getDeviceNum() {
        return device_id;
    }

    public int getComputeModules() {
        final int[] pi = new int[1];
        JCudaDriver.cuDeviceGetAttribute(pi, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, _device);
        return pi[0];
    }

    public int getTotalCores() {
        final int[] results = getMajorMinor();
        return calculateTotalCores(results[0], results[1]) * getComputeModules();
    }

    private int calculateTotalCores(final int major, final int minor) {
        if (major == 1) {
            if (minor == 0 || minor == 1 || minor == 2 || minor == 3) {
                return 8;
            }
            return 0;
        }
        if (major == 2) {
            if (minor == 0) {
                return 32;
            }
            if (minor == 1) {
                return 48;
            }
            return 0;
        }
        if (major == 3) {
            if (minor == 0 || minor == 5) {
                return 192;
            }
            return 0;
        }

        return 0;
    }

    public String getComputeCapability() {
        final int[] result = getMajorMinor();
        return result[0] + "." + result[1];
    }

    public int[] getMajorMinor() {
        final int[] major = new int[1];
        final int[] minor = new int[1];
        JCudaDriver.cuDeviceComputeCapability(major, minor, _device);
        final int[] result = new int[2];
        result[0] = major[0];
        result[1] = minor[0];

        return result;
    }

    public long getGlobalMemorySize() {
        final long[] amountOfBytes = new long[1];
        JCudaDriver.cuDeviceTotalMem(amountOfBytes, _device);
        return amountOfBytes[0];
    }

    public int getSharedMemorySize() {
        return getProperties().sharedMemPerBlock;
    }

    public long getMemPitch() {
        return getProperties().memPitch;
    }

    public int getThreads() {
        return getProperties().maxThreadsPerBlock;
    }

    public int getMaxGridSize() {
        return getProperties().maxGridSize[0];
    }

    public int getMaxBlockSize() {
        return getProperties().maxThreadsDim[0];
    }

    public int getMaxRegisters() {
        return getProperties().regsPerBlock;
    }

    public int getWarpSize() {
        return getProperties().SIMDWidth;
    }

    private CUdevprop getProperties() {
        final CUdevprop prop = new CUdevprop();
        JCudaDriver.cuDeviceGetProperties(prop, _device);
        return prop;
    }

    @Override
    public String toString() {
        return _device + "[name=" + getName() + "]";
    }

    public String getName() {
        final byte[] name = new byte[4096];
        if (cuDeviceGetName(name, name.length, _device) == CUDA_SUCCESS) {
            for (int i = 0; i < 4096; i++) {
                if (name[i] == 0) {
                    return new String(name, 0, i);
                }
            }
        }
        return "<unnamed>";
    }
}
