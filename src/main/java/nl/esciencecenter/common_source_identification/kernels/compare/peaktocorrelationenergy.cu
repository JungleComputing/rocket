/*
* Copyright 2015 Netherlands eScience Center, VU University Amsterdam, and Netherlands Forensic Institute
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
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

/**
 * This file contains CUDA kernels for comparing two PRNU noise patterns
 * using Peak To Correlation Energy.
 *
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 */

#ifndef block_size_x
#define block_size_x 32
#endif

#ifndef block_size_y
#define block_size_y 16
#endif


//function interfaces to prevent C++ garbling the kernel keys
extern "C" {
    __global__ void toComplex(int h, int w, float* x, float* input_x);
    __global__ void toComplexAndFlip(int h, int w, float* y, float *input_y);
    __global__ void toComplexAndFlip2(int h, int w, float* x, float* y, float* input_x, float *input_y);
    __global__ void computeEnergy(int h, int w, double *energy, int *peakIndex, float *input);
    __global__ void computeCrossCorr(int h, int w, float *c, float *x, float *y);
    __global__ void findPeak(int h, int w, float *peakValue, float *peakValues, int *peakIndex, float *input);

    __global__ void sumDoubles(double *output, double *input, int n);
    __global__ void maxlocFloats(int *output_loc, float *output_float, int *input_loc, float *input_float, int n);
    __global__ void computePCE(double *pce, float *peak, double *energy);
}


/**
 * Simple helper kernel to convert an array of real values to an array of complex values
 */
__global__ void toComplex(int h, int w, float* x, float *input_x) {
    int i = threadIdx.y + blockIdx.y * block_size_y;
    int j = threadIdx.x + blockIdx.x * block_size_x;

    if (i < h && j < w) {
        x[i * w * 2 + 2 * j] = input_x[i * w + j];
        x[i * w * 2 + 2 * j + 1] = 0.0f;
    }
}

/**
 * Simple helper kernel to convert an array of real values to a flipped array of complex values
 */
__global__ void toComplexAndFlip(int h, int w, float *y, float* input_y) {
    int i = threadIdx.y + blockIdx.y * block_size_y;
    int j = threadIdx.x + blockIdx.x * block_size_x;

    if (i < h && j < w) {
        //y is flipped vertically and horizontally
        int yi = h - i -1;
        int yj = w - j -1;
        y[yi* w * 2 + 2 * yj] = input_y[i * w + j];
        y[yi* w * 2 + 2 * yj + 1] = 0.0f;
    }
}

/**
 * Two-in-one kernel that puts x and y to Complex, but flips y
 */
__global__ void toComplexAndFlip2(int h, int w, float *x, float *y, float *input_x, float *input_y) {
    int i = threadIdx.y + blockIdx.y * block_size_y;
    int j = threadIdx.x + blockIdx.x * block_size_x;

    if (i < h && j < w) {
        x[i * w * 2 + 2 * j] = input_x[i * w + j];
        x[i * w * 2 + 2 * j + 1] = 0.0f;

        //y is flipped vertically and horizontally
        int yi = h - i -1;
        int yj = w - j -1;
        y[yi* w * 2 + 2 * yj] = input_y[i * w + j];
        y[yi* w * 2 + 2 * yj + 1] = 0.0f;

    }
}


/*
 * This method computes a cross correlation in frequency space
 */
__global__ void computeCrossCorr(int h, int w, float *c, float *x, float *y) {
    int i = threadIdx.y + blockIdx.y * block_size_y;
    int j = threadIdx.x + blockIdx.x * block_size_x;

    if (i < h && j < w) {
        int oindex = i * w + j;
        int iindex;
        bool xConj = false;
        bool yConj = false;

        if (j  < w / 2 + 1) {
            iindex = i * (w / 2 + 1) + j;
            xConj = true;
        } else {
            iindex = i * (w / 2 + 1) + (w - j);
            yConj = true;
        }

        // Load inputs
        float xRe = x[2 * iindex + 0];
        float xIm = x[2 * iindex + 1];
        float yRe = y[2 * iindex + 0];
        float yIm = y[2 * iindex + 1];

        // Take complex conjugate if in mirrored part
        if (xConj) xIm = -xIm;
        if (yConj) yIm = -yIm;

        // Store results
        c[2 * oindex + 0] = (xRe * yRe) - (xIm * yIm);
        c[2 * oindex + 1] = (xRe * yIm) + (xIm * yRe);
    }
}


/* ----------- kernels below this line are reducing kernels ------------ */
#ifndef grid_size_x   //hack to check if kernel tuner is being used
 #undef block_size_x
 #define block_size_x 256
#endif



/* 
 * This method searches for the peak value in a cross correlated signal and outputs the index
 * input is assumed to be a complex array of which only the real component contains values that
 * contribute to the peak
 *
 * Thread block size should be power of two because of the reduction.
 * The implementation currently assumes only one thread block is used for the entire input array
 * 
 * In case of multiple thread blocks initialize output to zero and use atomic add or another kernel
 */
__global__ void findPeak(int h, int w, float *peakValue, float *peakValues, int *peakIndex, float *input) {

    int x = blockIdx.x * block_size_x + threadIdx.x;
    int ti = threadIdx.x;
    int step_size = gridDim.x * block_size_x;
    int n = h*w;
    __shared__ float shmax[block_size_x];
    __shared__ int shind[block_size_x];

    //compute thread-local sums
    float max = -1.0f;
    float val = 0.0f;
    int index = -1;
    for (int i=x; i < n; i+=step_size) {
        val = fabsf(input[i*2]); //input is a complex array, only using real value 
        if (val > max) {
            max = val;
            index = i;
        }
    }

    //store local sums in shared memory
    shmax[ti] = max;
    shind[ti] = index;
    __syncthreads();
        
    //reduce local sums
    for (unsigned int s=block_size_x/2; s>0; s>>=1) {
        if (ti < s) {
            float v1 = shmax[ti];
            float v2 = shmax[ti + s];
            if (v1 < v2) {
                shmax[ti] = v2;
                shind[ti] = shind[ti + s];
            }
        }
        __syncthreads();
    }
        
    //write result
    if (ti == 0) {
        peakValues[blockIdx.x] = shmax[0];
        peakIndex[blockIdx.x] = shind[0];
        if (blockIdx.x == 0) {
            peakValue[0] = input[n*2-2]; //instead of using real peak use last real value
        }
    }

}


/* 
 * This method computes the energy of the signal minus an area around the peak
 *
 * input is assumed to be a complex array of which only the real component
 * contains values that contribute to the energy
 *
 * Thread block size should be power of two because of the reduction.
 * The implementation currently assumes only one thread block is used for the entire input array
 * 
 * In case of multiple thread blocks run kernel twice, with 1 thread block the second time
 */
#define SQUARE_SIZE 11
#define RADIUS 5
__global__ void computeEnergy(int h, int w, double *energy, int *peakIndex, float *input) {

    int x = blockIdx.x * block_size_x + threadIdx.x;
    int ti = threadIdx.x;
    int step_size = gridDim.x * block_size_x;
    int n = h*w;
    __shared__ double shmem[block_size_x];

    int peak_i = peakIndex[0];
    int peak_y = peak_i / w;
    int peak_x = peak_i - (peak_y * w);

    double sum = 0.0f;

    if (ti < n) {
        //compute thread-local sums
        for (int i=x; i < n; i+=step_size) {
            int row = i / w;
            int col = i - (row*w);

            //exclude area around the peak from sum
            int peakrow = (row > peak_y - RADIUS && row < peak_y + RADIUS);
            int peakcol = (col > peak_x - RADIUS && col < peak_x + RADIUS);
            if (peakrow && peakcol) {
                continue;
            } else {
                double val = input[row*w*2+col*2];
                sum += val * val;
            }
        }
    }
        
    //store local sums in shared memory
    shmem[ti] = sum;
    __syncthreads();
        
    //reduce local sums
    for (unsigned int s=block_size_x/2; s>0; s>>=1) {
        if (ti < s) {
            shmem[ti] += shmem[ti + s];
        }
        __syncthreads();
    }
        
    //write result
    if (ti == 0) {
        energy[blockIdx.x] = shmem[0] / (double)((w*h) - (SQUARE_SIZE * SQUARE_SIZE));
    }

}


/*
 * Simple CUDA Helper function to reduce the output of a
 * reduction kernel with multiple thread blocks to a single value
 * 
 * This function performs a sum of an array of doubles
 *
 * This function is to be called with only a single thread block
 */
__global__ void sumDoubles(double *output, double *input, int n) {
    int ti = threadIdx.x;
    __shared__ double shmem[block_size_x];

    //compute thread-local sums
    double sum = 0.0;
    for (int i=ti; i < n; i+=block_size_x) {
        sum += input[i];
    }
        
    //store local sums in shared memory
    shmem[ti] = sum;
    __syncthreads();
        
    //reduce local sums
    for (unsigned int s=block_size_x/2; s>0; s>>=1) {
        if (ti < s) {
            shmem[ti] += shmem[ti + s];
        }
        __syncthreads();
    }
        
    //write result
    if (ti == 0) {
        output[0] = shmem[0];
    }
}


/*
 * Simple CUDA helper functions to reduce the output of a reducing kernel with multiple
 * thread blocks to a single value
 *
 * This function performs a reduction for the max and the location of the max
 *
 * This function is to be called with only one thread block
 */
__global__ void maxlocFloats(int *output_loc, float *output_float, int *input_loc, float *input_float, int n) {

    int ti = threadIdx.x;
    __shared__ float shmax[block_size_x];
    __shared__ int shind[block_size_x];

    //compute thread-local variables
    float max = -1.0f;
    float val = 0.0f;
    int loc = -1;
    for (int i=ti; i < n; i+=block_size_x) {
         val = input_float[i];
         if (val > max) {
             max = val;
             loc = input_loc[i];
         }
    }
        
    //store local variables in shared memory
    shmax[ti] = max;
    shind[ti] = loc;
    __syncthreads();
        
    //reduce local variables
    for (unsigned int s=block_size_x/2; s>0; s>>=1) {
        if (ti < s) {
            float v1 = shmax[ti];
            float v2 = shmax[ti + s];
            if (v1 < v2) {
                shmax[ti] = v2;
                shind[ti] = shind[ti + s];
            }
        }
        __syncthreads();
    }
        
    //write result
    if (ti == 0) {
        output_float[0] = shmax[0]; 
        output_loc[0] = shind[0]; 
    }

}


/*
 * Simple kernel to calculate the final PCE given the peak and energy of the cross correlation.
 */
__global__ void computePCE(double *pce, float *peak, double *energy) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *pce = ((*peak) * (*peak)) / (*energy);
  }
}
