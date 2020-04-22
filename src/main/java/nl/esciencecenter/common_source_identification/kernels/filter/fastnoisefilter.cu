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
 * This file contains the CUDA kernels for extracting a PRNU pattern from a
 * grayscaled image using Fist Step Total Variation (FSTV) as described in:
 *
 * "Improving source camera identification using a simplified total variation
 * based noise removal algorithm" by F. Gisolf et al. In: Digital Investigation,
 * Volume 10, Issue 3, October 2013, Pages 207-214
 *
 * To apply the complete filter call both convolveVertically() and convolveHorizontally()
 * on the input and store the extracted gradients separately. Normalize these gradients
 * using normalize() and call convolveVertically() and convolveHorizontally() again on a
 * zeroed array using the normalized gradients as inputs to accumulate the PRNU pattern.
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
    __global__ void convolveVertically(int h, int w, float* output, float* input);
    __global__ void convolveHorizontally(int h, int w, float* output, float* input);
    __global__ void normalize(int h, int w, float* dxs, float* dys);
    __global__ void zeroMem(int h, int w, float* array);
    __global__ void normalized_gradient(int h, int w, float *output1, float *output2, float *input);
    __global__ void gradient(int h, int w, float *output, float *input1, float *input2);
}

/**
 * Vertically computes a local gradient for each pixel in an image.
 * Takes forward differences for first and last row.
 * Takes centered differences for interior points.
 */
__global__ void convolveVertically(int h, int w, float* output, float* input) {
    int i = threadIdx.y + blockIdx.y * block_size_y;
    int j = threadIdx.x + blockIdx.x * block_size_x;

    if (j < w && i < h) {

        float res = output[i*w+j];

        if (i == 0) {
            res += input[1*w+j] - input[0*w+j];
        } 
        else if (i == h-1) {
            res += input[i*w+j] - input[(i-1)*w+j];
        } 
        else if (i > 0 && i < h-1) {
            res += 0.5f * (input[(i+1)*w+j] - input[(i-1)*w+j]);
        }

        output[i*w+j] = res;

    }
}

/**
 * Horizontally computes a local gradient for each pixel in an image.
 * Takes forward differences for first and last element.
 * Takes centered differences for interior points.
 */
__global__ void convolveHorizontally(int h, int w, float* output, float* input) {
    int i = threadIdx.y + blockIdx.y * block_size_y;
    int j = threadIdx.x + blockIdx.x * block_size_x;

    if (i < h && j < w) {

        float res = output[i*w+j];

        if (j == 0) {
            res += input[i*w+1] - input[i*w+0];
        } 
        else if (j == w-1) {
            res += input[i*w+j] - input[i*w+j-1];
        }
        else if (j > 0 && j < w-1) {
            res += 0.5f * (input[i*w+j+1] - input[i*w+j-1]);
        }

        output[i*w+j] = res;

    }
}

/**
 * Normalizes gradient values in place.
 */
__global__ void normalize(int h, int w, float* dxs, float* dys) {
    int i = threadIdx.y + blockIdx.y * block_size_y;
    int j = threadIdx.x + blockIdx.x * block_size_x;

    if (i < h && j < w) {
        float dx = dxs[i*w+j];
        float dy = dys[i*w+j];

        float norm = sqrt((dx * dx) + (dy * dy));
        float scale = 1.0f / (1.0f + norm);

        dxs[i*w+j] = scale * dx;
        dys[i*w+j] = scale * dy;
    }
}





/**
 * Helper kernel to zero an array.
 */
__global__ void zeroMem(int h, int w, float* array) {
    int i = threadIdx.y + blockIdx.y * block_size_y;
    int j = threadIdx.x + blockIdx.x * block_size_x;

    if (i < h && j < w) {
        array[i*w+j] = 0.0f;
    }
}




/*
 * The following contains a simplified re-implementation by Ben (Jan 2017).
 * Where the previously six kernel launches have been reduced to two.
 */
__device__ float horizontal_gradient(int h, int w, int i, int j, float *input) {
    float res = 0.0f;
    if (j == 0) {
        res += input[i*w+1] - input[i*w+0];
    }
    else if (j == w-1) {
        res += input[i*w+j] - input[i*w+j-1];
    }
    else {
        res += 0.5f * (input[i*w+j+1] - input[i*w+j-1]);
    }
    return res;
}

__device__ float vertical_gradient(int h, int w, int i, int j, float *input) {
   float res = 0.0f;
    if (i == 0) {
        res += input[1*w+j] - input[0*w+j];
    }
    else if (i == h-1) {
        res += input[i*w+j] - input[(i-1)*w+j];
    }
    else {
        res += 0.5f * (input[(i+1)*w+j] - input[(i-1)*w+j]);
    }
    return res;
}

__global__ void normalized_gradient(int h, int w, float *output1, float *output2, float *input) {
    int i = threadIdx.y + blockIdx.y * block_size_y;
    int j = threadIdx.x + blockIdx.x * block_size_x;

    if (i < h && j < w) {
        float dx = horizontal_gradient(h,w,i,j,input);
        float dy = vertical_gradient(h,w,i,j,input);

        float norm = sqrtf((dx * dx) + (dy * dy));
        float scale = 1.0f / (1.0f + norm);
    
        output1[i*w+j] = (scale * dx);
        output2[i*w+j] = (scale * dy);
    }
}

__global__ void gradient(int h, int w, float *output, float *input1, float *input2) {
    int i = threadIdx.y + blockIdx.y * block_size_y;
    int j = threadIdx.x + blockIdx.x * block_size_x;

    if (i < h && j < w) {
        float dx = horizontal_gradient(h,w,i,j,input1);
        float dy = vertical_gradient(h,w,i,j,input2);

        output[i*w+j] = dx + dy;
    }
}



