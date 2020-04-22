/*
Copyright (C) 2009 Rob van Nieuwpoort & John Romein
Astron
P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, nieuwpoort@astron.nl

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/
#include <cub/cub.cuh>
#include <cuComplex.h>

struct float8 {
  float a;
  float b;
  float c;
  float d;
  float e;
  float f;
  float g;
  float h;
};

__forceinline__ __host__ __device__ float8 make_float8(float a, float b, float c, float d, float e, float f, float g, float h) {
  float8 output = { a, b, c, d, e, f, g, h };
  return output;
}

__forceinline__ __host__ __device__ float8 operator+(float8 x, float8 y) {
    return make_float8(x.a+y.a, x.b+y.b, x.c+y.c, x.d+y.d, x.e+y.e, x.f+y.f, x.g+y.g, x.h+y.h);
}

extern "C"
__global__ void correlate(
        unsigned numTimes,
        const float4 *__restrict__ left,
        const float4 *__restrict__ right,
        float2 *__restrict__ result
) {
    typedef cub::BlockReduce<float8, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    unsigned myBlock   = blockIdx.x;
    unsigned myThread  = threadIdx.x;

    cuFloatComplex xx = make_cuFloatComplex(0.0, 0.0);
    cuFloatComplex xy = make_cuFloatComplex(0.0, 0.0);
    cuFloatComplex yx = make_cuFloatComplex(0.0, 0.0);
    cuFloatComplex yy = make_cuFloatComplex(0.0, 0.0);

    for (unsigned t = myThread; t < numTimes; t += BLOCK_SIZE) {
		float4 sample0 = left[numTimes * myBlock + t];
		float4 sample1 = right[numTimes * myBlock + t];

		cuFloatComplex lx = make_cuFloatComplex(sample0.x, sample0.y);
		cuFloatComplex ly = make_cuFloatComplex(sample0.z, sample0.w);
		cuFloatComplex rx = make_cuFloatComplex(sample1.x, sample1.y);
		cuFloatComplex ry = make_cuFloatComplex(sample1.z, sample1.w);

        xx = cuCaddf(xx, cuCmulf(lx, cuConjf(rx)));
        xy = cuCaddf(xy, cuCmulf(lx, cuConjf(ry)));
        yx = cuCaddf(yx, cuCmulf(ly, cuConjf(rx)));
        yy = cuCaddf(yy, cuCmulf(ly, cuConjf(ry)));
    }

    float8 out = BlockReduce(temp_storage).Sum(make_float8(
        cuCrealf(xx), cuCimagf(xx),
        cuCrealf(xy), cuCimagf(xy),
        cuCrealf(yx), cuCimagf(yx),
        cuCrealf(yy), cuCimagf(yy)
    ));

    if (myThread == 0) {
        result[4 * myBlock + 0] = make_cuFloatComplex(out.a, out.b);
        result[4 * myBlock + 1] = make_cuFloatComplex(out.c, out.d);
        result[4 * myBlock + 2] = make_cuFloatComplex(out.e, out.f);
        result[4 * myBlock + 3] = make_cuFloatComplex(out.g, out.h);
    }
}
