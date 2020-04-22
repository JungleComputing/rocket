#include <cub/cub.cuh>

#ifndef block_size_x
    #define block_size_x 128
#endif

#ifndef tile_size_x
    #define tile_size_x 1
#endif

#ifndef use_shared_mem
    #define use_shared_mem 0
#endif

__device__ __forceinline__ double4 operator+(double4 a, double4 b) {
  return make_double4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}


template <int tile_size, int stride, typename T>
__device__ __forceinline__ void fill_shared_mem_tiled_1D(T (&sh_mem)[tile_size*stride], const T *d_mem, int sh_offset, int d_offset, int N) {
    #pragma unroll
    for (int ti=0; ti<tile_size; ti++) {
        if (d_offset+ti*stride < N) {
            sh_mem[sh_offset+ti*stride] = d_mem[d_offset+ti*stride];
        }
    }
}


/*
 * This function performs the main body of work for computing the Gauss transform
 * The parallelization is such that one thread block is created
 * for each item in A, which is of size m. This implies that each thread block
 * does n (size of B) work.
 * The gradient computed in this function is reduced to a single value within the
 * thread block. The same is done for the cross term, which then needs to be
 * reduced in a second kernel.
 */
template<typename T, int dim>
__device__ __forceinline__ void GaussTransform_blocked_i(
        const T *A,
        const T *B,
        const int m,
        const int n,
        const T scale_sq,
        T *partial_gradient,
        T *partial_cross_term) {

    int tx = threadIdx.x;

    // Specialize BlockReduce for a 1D block of block_size_x threads on type T
    typedef cub::BlockReduce<T, block_size_x> BlockReduce;
    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;

    T cross_term = 0.0;
    T grad_i[dim];
    for (int d = 0; d < dim; d++) {
        grad_i[d] = 0.0;
    }

    int i = blockIdx.x;

    #if use_shared_mem == 1
    __shared__ T sh_A[dim][block_size_x*tile_size_x];

    #pragma unroll
    for (int d=0; d<dim; d++) {
        fill_shared_mem_tiled_1D<tile_size_x, block_size_x>(sh_A[d], A+d*m, tx, i, m);
    }
    __syncthreads();
    #endif

    //loop parallelized over threads within thread block
    for (int j = tx; j<n; j+=block_size_x) {

        T dist_ij = 0;
        #pragma unroll
        for (int d = 0; d < dim; ++d) {
            dist_ij += (A[i * dim + d] - B[j * dim + d])*(A[i * dim + d] - B[j * dim + d]);
        }
        T cost_ij = exp(-dist_ij/scale_sq);

        #pragma unroll
        for (int d = 0; d < dim; ++d) {
            grad_i[d] -= cost_ij * 2.0 * (A[i * dim + d] - B[j * dim + d]);
        }

        cross_term += cost_ij;
    }

    //reduce grad_i for each d, within the block
    #pragma unroll
    for (int d = 0; d < dim; d++) {
        grad_i[d] = BlockReduce(temp_storage).Sum(grad_i[d]);
        __syncthreads();
    }

    //reduce cross_term within the block, (division by m*n on CPU)
    cross_term = BlockReduce(temp_storage).Sum(cross_term);

    if (tx == 0 && blockIdx.x < m) {
        #pragma unroll
        for (int d = 0; d < dim; d++) {
            partial_gradient[blockIdx.x * dim + d] = grad_i[d] / (scale_sq * m * n);
        }

        partial_cross_term[blockIdx.x] = cross_term;
    }
}


extern "C"
__global__ void
GaussTransform(const double* A, const double* B,
                 int m, int n, double scale_sq, double *grad, double *cross_term) {

    //2-dimensional with double precision
    GaussTransform_blocked_i<double, 2>(A, B, m, n, scale_sq, grad, cross_term);

}

/*
 * Reduce the per thread block cross terms computed in the GaussTransform kernel to single value
 * and divide by (m*n)
 *
 * This kernel is designed to run as single-thread block, because the number of terms to reduce is
 * of size n or m, which is expected to be around 2000 or so. The number of items to reduce
 * is passed as the last argument 'nblocks', which corresponds to the number of thread blocks used
 * by the first kernel.
 */
extern "C"
__global__ void reduce_terms(
        double *__restrict__ d_cross_term,
        double *__restrict__ d_gradient,
        const int m,
        const int n,
        const double *__restrict__ partial_cross_term,
        const double *__restrict__ partial_gradient,
        const double *__restrict__ model,
        const double theta
    ) {

    int tx = threadIdx.x;
    // Specialize BlockReduce for a 1D block of block_size_x threads on type T
    typedef cub::BlockReduce<double4, block_size_x> BlockReduce;
    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double cross_term = 0.0;
    double gx = 0.0;
    double gy = 0.0;
    double gr = 0.0;
    double m00 = 0.0;
    double m01 = 0.0;
    double m10 = 0.0;
    double m11 = 0.0;

    for (int i = tx; i < m; i += block_size_x) {
        cross_term += partial_cross_term[i];

        gx += partial_gradient[2 * i + 0];
        gy += partial_gradient[2 * i + 1];

        m00 += partial_gradient[2 * i + 0] * model[2 * i + 0];
        m01 += partial_gradient[2 * i + 0] * model[2 * i + 1];
        m10 += partial_gradient[2 * i + 1] * model[2 * i + 0];
        m11 += partial_gradient[2 * i + 1] * model[2 * i + 1];
    }

    gr = m00 * -sin(theta) + m01 * -cos(theta) + m10 * cos(theta) + m11 * -sin(theta);
    cross_term = cross_term / (m*n);

    double4 out = BlockReduce(temp_storage).Sum(make_double4(gx, gy, gr, cross_term));
    gx = out.x;
    gy = out.y;
    gr = out.z;
    cross_term = out.w;


    if (tx == 0) {
      *d_cross_term = cross_term;
      d_gradient[0] = gx;
      d_gradient[1] = gy;
      d_gradient[2] = gr;
    }
}