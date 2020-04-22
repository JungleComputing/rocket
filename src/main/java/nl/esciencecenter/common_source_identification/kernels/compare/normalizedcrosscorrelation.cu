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
 * @author Hanno Spreeuw <h.spreeuw@esciencecenter.nl>
 * @version 0.2
 */

// Should be a power of two!!
#ifndef block_size_x
#define block_size_x 256
#endif




//function interfaces to prevent C++ garbling the kernel keys
extern "C" {
    __global__ void computeSums(
            int n,
            const float *x,
            const float *y,
            double *partialXX,
            double *partialX,
            double *partialYY,
            double *partialY,
            double *partialXY
    );

    __global__ void computeNCC(
            int n,
            const double* partialXX,
            const double* partialX,
            const double* partialYY,
            const double* partialY,
            const double* partialXY,
            double* output
    );
}


__device__ __forceinline__ double sumDoublesSharedMem(double input, double *shmem) {
    int ti = threadIdx.x;

    //store local sums in shared memory
    shmem[ti] = input;
    __syncthreads();

    //reduce local sums
    for (unsigned int s=blockDim.x / 2; s>0; s>>=1) {
        if (ti < s) {
            shmem[ti] += shmem[ti + s];
        }
        __syncthreads();
    }

    return shmem[0];
}

/*
 * Simple CUDA Helper function to reduce the output of a
 * reduction kernel with multiple thread blocks to a single value
 *
 * This function performs a sum of an array of doubles
 *
 * This function is to be called with only a single thread block
 */
__global__ void computeNCC(
        int n,
        const double* __restrict__ partialXX,
        const double* __restrict__ partialX,
        const double* __restrict__ partialYY,
        const double* __restrict__ partialY,
        const double* __restrict__ partialXY,
        double* __restrict__ output
) {
    int ti = threadIdx.x;
    __shared__ double shmem[block_size_x];

    //compute thread-local sums
    double sumx = 0.0;
    double sumy = 0.0;
    double sumxx = 0.0;
    double sumyy = 0.0;
    double sumxy = 0.0;

    for (int i=ti; i < n; i+=block_size_x) {
        sumx += partialX[i];
        sumxx += partialXX[i];
        sumy += partialY[i];
        sumyy += partialYY[i];
        sumxy += partialXY[i];
    }

    //reduce local sums
    sumx = sumDoublesSharedMem(sumx, shmem);
    sumxx = sumDoublesSharedMem(sumxx, shmem);
    sumy = sumDoublesSharedMem(sumy, shmem);
    sumyy = sumDoublesSharedMem(sumyy, shmem);
    sumxy = sumDoublesSharedMem(sumxy, shmem);

    //write result
    if (ti == 0) {
        double stddev_x = sqrt(sumxx - sumx * sumx);
        double stddev_y = sqrt(sumyy - sumy * sumy);
        double ncc = sumxy / (stddev_x * stddev_y);

        output[0] = ncc;
    }
}

 
__global__ void computeSums(
        int n,
        const float *__restrict x,
        const float *__restrict y,
        double *__restrict partialXX,
        double *__restrict partialX,
        double *__restrict partialYY,
        double *__restrict partialY,
        double *__restrict partialXY
) {
    int _x = blockIdx.x * block_size_x + threadIdx.x;
    int ti = threadIdx.x;
    int step_size = gridDim.x * block_size_x;

    __shared__ double shmem[block_size_x];

    //compute thread-local sums
    double sumx = 0.0;
    double sumy = 0.0;
    double sumxx = 0.0;
    double sumyy = 0.0;
    double sumxy = 0.0;

    for (int i=_x; i < n; i+=step_size) {
        float v = x[i];
        float w = y[i];
        sumx += v;
        sumxx += v * v;
        sumy += w;
        sumyy += w * w;
        sumxy += v * w;
    }

    //reduce local sums
    sumx = sumDoublesSharedMem(sumx, shmem);
    sumxx = sumDoublesSharedMem(sumxx, shmem);
    sumy = sumDoublesSharedMem(sumy, shmem);
    sumyy = sumDoublesSharedMem(sumyy, shmem);
    sumxy = sumDoublesSharedMem(sumxy, shmem);

    //write result
    if (ti == 0) {
        partialX[blockIdx.x]  = sumx;
        partialXX[blockIdx.x] = sumxx;
        partialY[blockIdx.x]  = sumy;
        partialYY[blockIdx.x] = sumyy;
        partialXY[blockIdx.x] = sumxy;
    }
}   


