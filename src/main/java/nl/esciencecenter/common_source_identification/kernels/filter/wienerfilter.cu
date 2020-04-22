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
 * This file contains CUDA kernels for applying a Wiener filter to a 
 * PRNU pattern, as proposed by:
 * M. Chen et al. "Determining image origin and integrity using sensor
 * noise", IEEE Trans. Inf. Forensics Secur. 3 (2008) 74-90.
 *
 * The Wiener filter is used to remove JPEG artifacts from a PRNU pattern.
 *
 * To apply the complete filter:
 *  apply Fourier transform to the input image
 *     call computeSquaredMagnitudes() on the frequencies
 *     call computeVarianceEstimates() on the squared magnitudes
 *     call computeVarianceZeroMean() on the squared magnitudes
 *     call scaleWithVariances() scaling the frequencies using the local and global variance
 *  apply inverse Fourier transform
 *  normalize result by calling normalizeToReal()
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

#ifndef reuse_computation
#define reuse_computation 1
#endif

//set the number and size of filters, also adjust max_border
#define FILTERS 4
#define FILTER_SIZES {3, 5, 7, 9}
#define MAX_BORDER 4   //the largest (filter size/2)

#define FLT_MAX 3.40282347e+38f

//function interfaces to prevent C++ garbling the kernel keys
extern "C" {
    __global__ void computeSquaredMagnitudes(int h, int w, float* output, float* frequencies);
    __global__ void scaleWithVariances(int h, int w, float* output, float* input, float* varianceEstimates, float* variance);
    __global__ void toComplex(int h, int w, float* complex, float* input);
    __global__ void toReal(int h, int w, float* output, float* complex);
    __global__ void computeVarianceZeroMean(int n, float* output, float *input);
    __global__ void computeVarianceEstimates(int h, int w, float* varest, float* input);
    __global__ void computeVarianceEstimates_naive(int h, int w, float* varest, float* input);
    __global__ void normalizeToReal(int h, int w, float* output, float* complex);
    __global__ void normalize(int h, int w, float* output, float* complex);
    __global__ void sumFloats(float *output, float *input, int n);
}


/**
 * Computes the square of each frequency and stores the result as a real.
 */
__global__ void computeSquaredMagnitudes(int h, int w, float* output, float* frequencies) {
    int i = threadIdx.y + blockIdx.y * block_size_y;
    int j = threadIdx.x + blockIdx.x * block_size_x;

    if (j < w && i < h) {
        float re = frequencies[i*2*w+(2 * j)];
        float im = frequencies[i*2*w+(2 * j + 1)];
        output[i*w+j] = (re * re) + (im * im);
    }
}

/**
 * This kernel scales the frequencies in input with a combination of the global variance and an estimate
 * for the local variance at that position. Effectively this cleans the input pattern from low frequency
 * noise.
 */
__global__ void scaleWithVariances(int h, int w, float* output, float* input, float* varianceEstimates, float* variance) {
    int i = threadIdx.y + blockIdx.y * block_size_y;
     int j = threadIdx.x + blockIdx.x * block_size_x;

    float var = variance[0];

    if (j < w && i < h) {
        float scale = var / max(var, varianceEstimates[i*w+j]);
        output[i*2*w+(j * 2)] = input[i*2*w+(j*2)] * scale;
        output[i*2*w+(j * 2 + 1)] = input[i*2*w+(j * 2 + 1)] * scale;
    }
}

/**
 * Simple helper kernel to convert an array of real values to an array of complex values
 */
__global__ void toComplex(int h, int w, float* complex, float* input) {
    int i = threadIdx.y + blockIdx.y * block_size_y;
    int j = threadIdx.x + blockIdx.x * block_size_x;

    if (i < h && j < w) {
        complex[i * w * 2 + 2 * j] = input[i * w + j];
        complex[i * w * 2 + (2 * j + 1)] = 0.0f;
    }
}

/**
 * Simple helper kernel to convert a complex array to an array of real values
 */
__global__ void toReal(int h, int w, float* output, float* complex) {
    int i = threadIdx.y + blockIdx.y * block_size_y;
    int j = threadIdx.x + blockIdx.x * block_size_x;

    if (i < h && j < w) {
        output[i*w+j] = complex[i * w * 2 + 2 * j];
    }
}

/**
 * This kernel normalizes the input by dividing it by the number of pixels in the image.
 * It takes an array of complex numbers as input, but only stores the real values.
 */
__global__ void normalizeToReal(int h, int w, float* output, float* complex) {
    int i = threadIdx.y + blockIdx.y * block_size_y;
    int j = threadIdx.x + blockIdx.x * block_size_x;

    if (i < h && j < w) {
        output[i*w+j] = (complex[i * w * 2 + 2 * j] / (float)(w * h));
    }
}

/**
 * This kernel normalizes the complex input by dividing it by the number of pixels in the image.
 */
__global__ void normalize(int h, int w, float* complex_out, float* complex_in) {
    int i = threadIdx.y + blockIdx.y * block_size_y;
    int j = threadIdx.x + blockIdx.x * block_size_x;

    if (i < h && j < w) {
        complex_out[i*w*2+2*j  ] = (complex_in[i*w*2+2*j  ] / (float)(w*h));
        complex_out[i*w*2+2*j+1] = (complex_in[i*w*2+2*j+1] / (float)(w*h));
    }
}

/**
 * computeVarianceEstimates uses a number of simple filters to compute a minimum local variance
 *
 * Instead of using multiple arrays with zeroed borders around them, the loading phase of this
 * kernel writes a zero to shared memory instead of loading a border value from global memory.
 * The filters can then be performed as normal on the data in shared memory. Because of this
 * MAX_BORDER needs to be set accordingly.
 *
 */
__global__ void computeVarianceEstimates(int h, int w, float* varest, float* input) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int i = blockIdx.y * block_size_y;
    int j = blockIdx.x * block_size_x;

    __shared__ float shinput[block_size_y+2*MAX_BORDER][block_size_x+2*MAX_BORDER];
    
    //the loading phase of the kernel, which writes 0.0f to shared memory if the index
    //is outside the input
    int yEnd = block_size_y+2*MAX_BORDER;
    int xEnd = block_size_x+2*MAX_BORDER;
    for (int y=ty; y < yEnd; y+= block_size_y) {
        for (int x=tx; x < xEnd; x+= block_size_x) {
            float in = 0.0f;
            int indexy = i+y-MAX_BORDER;
            int indexx = j+x-MAX_BORDER; 
            if (indexy >= 0 && indexy < h) {
                if (indexx >= 0 && indexx < w) {
                    in = input[indexy*w+indexx];
                }
            }
            shinput[y][x] = in;
        }
    }
    __syncthreads();

    const int filter[FILTERS] = FILTER_SIZES;
    float res = FLT_MAX;

    #if reuse_computation == 0
    //perform filtering without reusing the sum from smaller filters
    for (int f = 0; f < FILTERS; f++) {
        int filterSize = filter[f];
        int offset = MAX_BORDER-(filterSize/2);
        
        //do a convolution
        float sum = 0.0f;
        for (int fi = 0; fi < filterSize; fi++) {
            for (int fj = 0; fj < filterSize; fj++) {
                sum += shinput[ty+fi+offset][tx+fj+offset]; 
            }
        }
        sum /= (float)(filterSize * filterSize);
        
        //store minimum
        res = sum < res ? sum : res; 
    }

    #elif reuse_computation == 1
    //perform filtering while reusing the sum from smaller filters

    //start from center pixel
    float sum = shinput[ty+MAX_BORDER][tx+MAX_BORDER]; 

    //add sides of the square filter to sum and store minimum average
    for (int f = 0; f < FILTERS; f++) {
        int filterSize = filter[f];
        int offset = MAX_BORDER-(filterSize/2);
    
        //top and bottom row
        for (int fj=0; fj<filterSize; fj++) {
            sum += shinput[ty+0+offset][tx+fj+offset];
            sum += shinput[ty+filterSize-1+offset][tx+fj+offset];
        }
        //two sides (between top and bottom rows)
        for (int fi=1; fi<filterSize-1; fi++) {
            sum += shinput[ty+fi+offset][tx+0+offset]; 
            sum += shinput[ty+fi+offset][tx+filterSize-1+offset]; 
        }

        //store minimum
        float avg = sum / (filterSize*filterSize);
        res = avg < res ? avg : res;
    }
    #endif

    //write output
    if (i + ty < h) {
      if (j + tx < w) {
	varest[(i+ty)*w + (j+tx)] = res;
      }
    }
}

/**
 * This method is a naive implementation of computeVarianceEstimates used for correctness checks
 */
__global__ void computeVarianceEstimates_naive(int h, int w, float* varest, float* input) {
    int i = threadIdx.y + blockIdx.y * block_size_y;
    int j = threadIdx.x + blockIdx.x * block_size_x;

    float res = FLT_MAX;
    if (i < h && j < w) {
    
    const int filter[FILTERS] = FILTER_SIZES;
    for (int f = 0; f < FILTERS; f++) {
        int filterSize = filter[f];
        int border = filterSize/2;
        
        //do a convolution
        float sum = 0.0f;
        for (int fi = 0; fi < filterSize; fi++) {
            for (int fj = 0; fj < filterSize; fj++) {
                //original
                //sum += input[(i + fi)*(w+border*2)+(j + fj)];
        
                int row = i+fi-border;
                int col = j+fj-border;
                //the following ifs are a hack to save redundant copying
                if (row >= 0 && row < h) {
                    if (col >= 0 && col < w) {
                          sum += input[row*w + col];
                    }
                }
            }
        }
        sum /= (float)(filterSize * filterSize);
        
        if (sum < res) {
            res = sum;
        }
    }
    
    //write output
    varest[i*w+j] = res;
    }
}

/* 
 * This method computes the variance of an input array, assuming the mean is equal to zero
 *
 * Thread block size should be power of two because of the reduction.
 * The implementation currently assumes only one thread block is used for the entire input array
 * 
 * In case of multiple thread blocks initialize output to zero and use atomic add or another kernel
 *
 * block_size_x power of 2
 */

#ifndef grid_size_x //hack to see if the Kernel Tuner is being used
 #undef block_size_x
 #define block_size_x 128
#endif
__global__ void computeVarianceZeroMean(int n, float *output, float *input) {

    int x = blockIdx.x * block_size_x + threadIdx.x;
    int ti = threadIdx.x;
    int step_size = block_size_x * gridDim.x;

    float sum = 0.0f;
    if (x < n) {
        //compute thread-local sums of squares
        for (int i=x; i < n; i+=step_size) {
            sum += input[i]*input[i];
        }
    }
        
    //store local sums in shared memory
    __shared__ float shmem[block_size_x];
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
        output[blockIdx.x] = ( shmem[0] * n ) / ( n - 1 ); //in case of multiple threadblocks write back using atomicAdd
    }

}

/*
 * Simple CUDA Helper function to reduce the output of a
 * reduction kernel with multiple thread blocks to a single value
 * 
 * This function performs a sum of an array of floats
 *
 * This function is to be called with only a single thread block
 */
__global__ void sumFloats(float *output, float *input, int n) {
    int ti = threadIdx.x;
    __shared__ float shmem[block_size_x];

    //compute thread-local sums
    float sum = 0.0f;
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

