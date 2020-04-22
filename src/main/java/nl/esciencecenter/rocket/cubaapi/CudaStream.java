/*
 * Copyright (c) 2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.esciencecenter.rocket.cubaapi;

import static jcuda.driver.JCudaDriver.cuStreamAddCallback;
import static jcuda.driver.JCudaDriver.cuStreamCreate;
import static jcuda.driver.JCudaDriver.cuStreamDestroy;
import static jcuda.driver.JCudaDriver.cuStreamSynchronize;
import static jcuda.driver.JCudaDriver.cuStreamWaitEvent;

import jcuda.driver.CUstream;
import jcuda.driver.CUstreamCallback;
import jcuda.driver.CUstream_flags;

public final class CudaStream {
    static final private CUstreamCallback DEFAULT_RUNNABLE_CALLBACK = new CUstreamCallback() {
        @Override
        public void call(CUstream hStream, int status, Object userData) {
            ((Runnable) userData).run();
        }
    };

    private final CUstream _stream;


    public CudaStream() {
        _stream = new CUstream();
        cuStreamCreate(_stream, CUstream_flags.CU_STREAM_NON_BLOCKING);
    }

    public CUstream cuStream() {
        return _stream;
    }

    public void addCallback(Runnable r) {
        cuStreamAddCallback(_stream, DEFAULT_RUNNABLE_CALLBACK, r, 0);
    }

    public void synchronize() {
        cuStreamSynchronize(_stream);
    }

    public void waitEvent(CudaEvent event) {
        cuStreamWaitEvent(_stream, event.cuEvent(), 0);
    }

    public void destroy() {
        cuStreamDestroy(_stream);
    }
}
