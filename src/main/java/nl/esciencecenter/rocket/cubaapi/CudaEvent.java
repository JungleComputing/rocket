/*
 * Copyright (c) 2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.esciencecenter.rocket.cubaapi;

import static jcuda.driver.JCudaDriver.cuEventCreate;
import static jcuda.driver.JCudaDriver.cuEventDestroy;
import static jcuda.driver.JCudaDriver.cuEventRecord;
import static jcuda.driver.JCudaDriver.cuEventSynchronize;

import jcuda.driver.CUevent;
import jcuda.driver.CUevent_flags;

public class CudaEvent {
    private final CUevent _event;

    public CudaEvent() {
        _event = new CUevent();
        cuEventCreate(_event, CUevent_flags.CU_EVENT_DISABLE_TIMING);
    }

    public void record(final CudaStream stream) {
        cuEventRecord(_event, stream.cuStream());
    }

    public CUevent cuEvent() {
        return _event;
    }

    public void synchronize() {
        cuEventSynchronize(_event);
    }

    public void destroy() {
        cuEventDestroy(_event);
    }

    @Override
    public String toString() {
        return _event.toString();
    }
}
