/*
* Copyright 2015 Netherlands eScience Center, VU University Amsterdam, and Netherlands Forensic Institute
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance withSupplier the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
package nl.esciencecenter.common_source_identification.kernels.compare;

import jcuda.CudaException;
import jcuda.Sizeof;
import nl.esciencecenter.rocket.cubaapi.*;
import jcuda.runtime.cudaStream_t;
import jcuda.driver.*;
import jcuda.jcufft.*;
import jcuda.Pointer;

import java.io.IOException;

/**
 * Class for comparing PRNU patterns using Peak to Correlation Energy ratio on the GPU
 *
 * This class also contains some routines of the Java code for computing PCE scores on the CPU given to me by the NFI.
 * 
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 */
public class PeakToCorrelationEnergy implements PatternComparator {

    protected CudaModule _module;
    protected CudaContext _context;
    protected CudaStream _stream;
    protected CudaStream _stream2;
    protected CudaMemFloat _d_input;
    protected CudaEvent _event;

    //handles to CUDA kernels
    protected CudaFunction _computeEnergy;
    protected CudaFunction _sumDoubles;
    protected CudaFunction _computeCrossCorr;
    protected CudaFunction _findPeak;
    protected CudaFunction _maxlocFloats;
    protected CudaFunction _computePCE;

    //handle for CUFFT plan
    protected cufftHandle _plan1;

    //handles to device memory arrays
    protected CudaMemFloat _d_inputx;
    protected CudaMemFloat _d_inputy;
    protected CudaMemFloat _d_x;
    protected CudaMemFloat _d_y;
    protected CudaMemFloat _d_c;
    protected CudaMemInt _d_peakIndex;
    protected CudaMemFloat _d_peakValues;
    protected CudaMemFloat _d_peakValue;
    protected CudaMemDouble _d_energy;
    protected CudaMemDouble _d_pce;

    //parameterlists for kernel invocations
    protected Pointer toComplex;
    protected Pointer toComplexAndFlip;
    protected Pointer computeEnergy;
    protected Pointer sumDoubles;
    protected Pointer findPeak;
    protected Pointer maxlocFloats;

    protected int h;
    protected int w;
    protected boolean useRealPeak;

    /**
     * Constructor for the PeakToCorrelationEnergy
     * 
     * @param h - the image height in pixels
     * @param w - the image width in pixels
     * @param context - the CudaContext as created by the factory
     * @param usePeak - a boolean for using the real peak value or using the last pixel value
     */
    public PeakToCorrelationEnergy(int h, int w, CudaContext context, boolean usePeak,
                                   String... compileArgs) throws CudaException, IOException {
        JCudaDriver.setExceptionsEnabled(true);
        _module = context.compileModule(getClass().getResource("peaktocorrelationenergy.cu"));

        _context = context;
        _stream = new CudaStream();
        _stream2 = new CudaStream();
        _event = new CudaEvent();
        this.h = h;
        this.w = w;

        //setup CUDA functions
        int threads_x = 32;
        int threads_y = 16;
        _computeCrossCorr = _module.getFunction("computeCrossCorr");
        _computeCrossCorr.setDim(    (int)Math.ceil((float)w / (float)threads_x), (int)Math.ceil((float)h / (float)threads_y), 1,
                threads_x, threads_y, 1);

        //dimensions for reducing kernels        
        int threads = 256;
        int reducing_thread_blocks = 1024;

        //int num_sm =_context.getDevice().getComputeModules();
        //System.out.println("detected " + num_sm + " SMs on GPU");
        //reducing_thread_blocks = num_sm;

        _findPeak = _module.getFunction("findPeak");
        _findPeak.setDim(    reducing_thread_blocks, 1, 1,
                threads, 1, 1);

        _maxlocFloats = _module.getFunction("maxlocFloats");
        _maxlocFloats.setDim( 1, 1, 1,
                threads, 1, 1);    

        _computeEnergy = _module.getFunction("computeEnergy");
        _computeEnergy.setDim( reducing_thread_blocks, 1, 1,
                threads, 1, 1);

        _sumDoubles = _module.getFunction("sumDoubles");
        _sumDoubles.setDim( 1, 1, 1,
                threads, 1, 1);

        _computePCE = _module.getFunction("computePCE");
        _computePCE.setDim(1, 1, 1,
                1, 1, 1);

        long free[] = new long[1];
        long total[] = new long[1];

        //JCuda.cudaMemGetInfo(free, total);
        //System.out.println("Before allocations in PCE free GPU mem: " + free[0]/1024/1024 + " MB total: " + total[0]/1024/1024 + " MB ");

        //allocate local variables in GPU memory
        _d_inputx = _context.allocFloats(h*w);
        _d_inputy = _context.allocFloats(h*w);
        _d_x = _context.allocFloats(h*w*2);
        _d_y = _context.allocFloats(h*w*2);
        _d_c = _context.allocFloats(h*w*2);
        _d_peakIndex = _context.allocInts(reducing_thread_blocks);
        _d_peakValues = _context.allocFloats(reducing_thread_blocks);
        _d_peakValue = _context.allocFloats(1);
        _d_energy = _context.allocDoubles(reducing_thread_blocks);
        _d_pce = _context.allocDoubles(1);

        //JCuda.cudaMemGetInfo(free, total);
        //System.out.println("After allocations in PCE free GPU mem: " + free[0]/1024/1024 + " MB total: " + total[0]/1024/1024 + " MB ");

        //initialize CUFFT
        JCufft.initialize();
        JCufft.setExceptionsEnabled(true);
        _plan1 = new cufftHandle();

        //create CUFFT plan and associate withSupplier stream
        int res;
        res = JCufft.cufftPlan2d(_plan1, h, w, cufftType.CUFFT_C2C);
        if (res != cufftResult.CUFFT_SUCCESS) {
            throw new CudaException(cufftResult.stringFor(res));
        }
        res = JCufft.cufftSetStream(_plan1, new cudaStream_t(_stream.cuStream()));
        if (res != cufftResult.CUFFT_SUCCESS) {
            throw new CudaException(cufftResult.stringFor(res));
        }

        //construct parameter lists for the CUDA kernels
        computeEnergy = Pointer.to(
                Pointer.to(new int[]{h}),
                Pointer.to(new int[]{w}),
                Pointer.to(_d_energy.asDevicePointer()),
                Pointer.to(_d_peakIndex.asDevicePointer()),
                Pointer.to(_d_c.asDevicePointer())
                );
        sumDoubles = Pointer.to(
                Pointer.to(_d_energy.asDevicePointer()),
                Pointer.to(_d_energy.asDevicePointer()),
                Pointer.to(new int[]{reducing_thread_blocks})
                );
        findPeak = Pointer.to(
                Pointer.to(new int[]{h}),
                Pointer.to(new int[]{w}),
                Pointer.to(_d_peakValue.asDevicePointer()),
                Pointer.to(_d_peakValues.asDevicePointer()),
                Pointer.to(_d_peakIndex.asDevicePointer()),
                Pointer.to(_d_c.asDevicePointer())
                );
        maxlocFloats = Pointer.to(
                Pointer.to(_d_peakIndex.asDevicePointer()),
                Pointer.to(_d_peakValues.asDevicePointer()),
                Pointer.to(_d_peakIndex.asDevicePointer()),
                Pointer.to(_d_peakValues.asDevicePointer()),
                Pointer.to(new int[]{reducing_thread_blocks})
                );
    }

    public void compare(CudaMemFloat d_x, CudaMemFloat d_y, CudaMemDouble result) {
        if (d_x.sizeInBytes() < getInputSize() ||
                d_y.sizeInBytes() < getInputSize() ||
                result.sizeInBytes() < getOutputSize()) {
            throw new IllegalArgumentException();
        }

        Pointer computeCrossCorrParams = Pointer.to(
                Pointer.to(new int[]{h}),
                Pointer.to(new int[]{w}),
                Pointer.to(_d_c.asDevicePointer()),
                Pointer.to(d_x.asDevicePointer()),
                Pointer.to(d_y.asDevicePointer()));

        Pointer computePCE = Pointer.to(
                Pointer.to(result.asDevicePointer()),
                Pointer.to(_d_peakValue.asDevicePointer()),
                Pointer.to(_d_energy.asDevicePointer()));

        try {
            _computeCrossCorr.launch(_stream, computeCrossCorrParams);
            JCufft.cufftExecC2C(_plan1, _d_c.asDevicePointer(), _d_c.asDevicePointer(), JCufft.CUFFT_INVERSE);
            _findPeak.launch(_stream, findPeak);
            _maxlocFloats.launch(_stream, maxlocFloats);
            _computeEnergy.launch(_stream, computeEnergy);
            _sumDoubles.launch(_stream, sumDoubles);
            _computePCE.launch(_stream, computePCE);
        } finally {
            _stream.synchronize();
        }
    }

    public long getInputSize() {
        int n = 2 * h * (w / 2 + 1);
        return Sizeof.FLOAT * n;
    }

    public long getOutputSize() {
        return  Sizeof.DOUBLE;
    }

    public void applyGPU(CudaMem left, CudaMem right, CudaMem result) {
        compare(left.asFloats(), right.asFloats(), result.asDoubles());
    }

    /**
     * Cleans up GPU memory and destroys FFT plan
     */
    public void cleanup() {
        _d_inputx.free();
        _d_inputy.free();
        _d_x.free();
        _d_y.free();
        _d_c.free();
        _d_peakIndex.free();
        _d_peakValue.free();
        _d_peakValues.free();
        _d_energy.free();
        _module.cleanup();
        JCufft.cufftDestroy(_plan1);
    }
}
