#include <cub/cub.cuh>

//defaults tuned for K40
#ifndef block_size_x
    #define block_size_x 32
#endif
#ifndef block_size_y
    #define block_size_y 4
#endif

#ifndef tile_size_x
    #define tile_size_x 4
#endif
#ifndef tile_size_y
    #define tile_size_y 2
#endif

#ifndef use_shared_mem
    #define use_shared_mem 1
#endif

template <int tile_size, int sh_stride, int d_stride, typename T>
__device__ __forceinline__ void fill_shared_mem_tiled_1D(T* sh_mem, const T *d_mem, int sh_offset, int d_offset, int N) {
    #pragma unroll
    for (int ti=0; ti<tile_size; ti++) {
        if (d_offset+ti*d_stride < N) {
            sh_mem[sh_offset+ti*sh_stride] = d_mem[d_offset+ti*d_stride];
        }
    }
}


/*
 * This function performs the main body of work for computing the Bhattacharya
 * cost function for two given point sets.
 * The parallelization is such that a 2D array of 2D thread blocks are created
 * to match the m*n iteration space. The amount of work per thread is controlled
 * through tiling factors tile_size_x and tile_size_y.
 * The cross term is reduced to a single value per thread block, which then needs
 * to be reduced to a single value in a second kernel.
 */
template<typename T, int dim>
__device__ __forceinline__ void ExpDist_tiled(const T *A, const T *B,
                 int m, int n, const T *scale_A, const T *scale_B, T *d_cross_term) {

    // Specialize BlockReduce for a 1D block of block_size_x threads on type T
    typedef cub::BlockReduce<T, block_size_x, cub::BLOCK_REDUCE_WARP_REDUCTIONS, block_size_y> BlockReduce;

    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;

    T cross_term = 0.0;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = tx + blockIdx.x * block_size_x * tile_size_x;
    int j = ty + blockIdx.y * block_size_y * tile_size_y;

    #if use_shared_mem == 1
    __shared__ T sh_A[dim][block_size_x*tile_size_x];
    __shared__ T sh_B[dim][block_size_y*tile_size_y];
    __shared__ T sh_scale_A[block_size_x*tile_size_x];
    __shared__ T sh_scale_B[block_size_y*tile_size_y];

    #pragma unroll
    for (int d=0; d<dim; d++) {
        fill_shared_mem_tiled_1D<tile_size_x, block_size_x, dim*block_size_x>(sh_A[d], A, tx, d + dim * i, dim * m);
        fill_shared_mem_tiled_1D<tile_size_y, block_size_y, dim*block_size_y>(sh_B[d], B, ty, d + dim * j, dim * n);
    }
    fill_shared_mem_tiled_1D<tile_size_x, block_size_x, block_size_x>(sh_scale_A, scale_A, tx, i, m);
    fill_shared_mem_tiled_1D<tile_size_y, block_size_y, block_size_y>(sh_scale_B, scale_B, ty, j, n);
    __syncthreads();
    #endif

    #pragma unroll
    for (int ti=0; ti<tile_size_x; ti++) {
        #pragma unroll
        for (int tj=0; tj<tile_size_y; tj++) {

            if ((i+ti*block_size_x < m) && (j+tj*block_size_y < n)) {

                T dist_ij = 0;

                #if use_shared_mem == 0
                    #pragma unroll
                    for (int d=0; d<dim; d++) {
                        int id = dim * (i+ti*block_size_x) + d;
                        int jd = dim * (j+tj*block_size_y) + d;
                        dist_ij += (A[id]-B[jd])*(A[id]-B[jd]);
                    }
                    cross_term += exp(-dist_ij/(scale_A[i+ti*block_size_x] + scale_B[j+tj*block_size_y]));
                #elif use_shared_mem == 1
                    #pragma unroll
                    for (int d=0; d<dim; d++) {
                        dist_ij += (sh_A[d][tx+ti*block_size_x]-sh_B[d][ty+tj*block_size_y])*(sh_A[d][tx+ti*block_size_x]-sh_B[d][ty+tj*block_size_y]);
                    }
                    cross_term += exp(-dist_ij/(sh_scale_A[tx+ti*block_size_x] + sh_scale_B[ty+tj*block_size_y]));
                #endif

            }
        }
    }

    //reduce cross_term within the block
    cross_term = BlockReduce(temp_storage).Sum(cross_term);

    //write back the per-thread block partial cross term
    if (tx == 0 && ty == 0) {
        d_cross_term[blockIdx.y*gridDim.x+blockIdx.x] = cross_term;
    }
}


extern "C"
__global__ void
ExpDist(const double *A, const double *B,
                 int m, int n, const double *scale_A, const double *scale_B, double *cross_term) {

    //2-dimensional with double precision
    ExpDist_tiled<double, 2>(A, B, m, n, scale_A, scale_B, cross_term);

}


template<typename T, int dim>
__device__ __forceinline__ T compute_expdist_block_shared(int tx, int ty, int i, int j,
                                                          T (&sh_A)[dim][block_size_x*tile_size_x],
                                                          T (&sh_B)[dim][block_size_y*tile_size_y],
                                                          T (&sh_scale_A)[block_size_x*tile_size_x],
                                                          T (&sh_scale_B)[block_size_y*tile_size_y], int m, int n) {
    T cross_term = 0;

    #pragma unroll
    for (int ti=0; ti<tile_size_x; ti++) {
        #pragma unroll
        for (int tj=0; tj<tile_size_y; tj++) {

            if ((i+ti*block_size_x < m) && (j+tj*block_size_y < n)) {

                T dist_ij = 0;

                #pragma unroll
                for (int d=0; d<dim; d++) {
                     dist_ij += (sh_A[d][tx+ti*block_size_x]-sh_B[d][ty+tj*block_size_y])*(sh_A[d][tx+ti*block_size_x]-sh_B[d][ty+tj*block_size_y]);
                }
                cross_term += exp(-dist_ij/(sh_scale_A[tx+ti*block_size_x] + sh_scale_B[ty+tj*block_size_y]));

            }
        }
    }

    return cross_term;
}


template<typename T, int dim>
__device__ __forceinline__ T compute_expdist_block(int i, int j, const T *A, const T *B,
                                                   const T *scale_A, const T* scale_B, int m, int n) {

    T cross_term = 0;

    #pragma unroll
    for (int ti=0; ti<tile_size_x; ti++) {
        #pragma unroll
        for (int tj=0; tj<tile_size_y; tj++) {

            if ((i+ti*block_size_x < m) && (j+tj*block_size_y < n)) {

                T dist_ij = 0;

                #pragma unroll
                for (int d=0; d<dim; d++) {
                    int id = dim*(i+ti*block_size_x) + d;
                    int jd = dim*(j+tj*block_size_y) + d;
                    dist_ij += (A[id]-B[jd])*(A[id]-B[jd]);
                }
                cross_term += exp(-dist_ij/(scale_A[i+ti*block_size_x] + scale_B[j+tj*block_size_y]));

            }
        }
    }

    return cross_term;
}



/*
 * This function performs the main body of work for computing the Bhattacharya
 * cost function for two given point sets.
 * The parallelization is such that a 1D array of 2D thread blocks is created over
 * the m-dimension. The thread blocks then iterate over n, to process the entire
 * m*n iteration space. The amount of work per thread is controlled
 * through tiling factors tile_size_x and tile_size_y.
 * The cross term is reduced to a single value per thread block, which then needs
 * to be reduced to a single value in a second kernel.
 */
template<typename T, int dim>
__device__ __forceinline__ void ExpDist_tiled_column(const T *A, const T *B,
                 int m, int n, const T *scale_A, const T *scale_B, T *d_cross_term) {

    // Specialize BlockReduce for a 1D block of block_size_x threads on type T
    typedef cub::BlockReduce<T, block_size_x, cub::BLOCK_REDUCE_WARP_REDUCTIONS, block_size_y> BlockReduce;

    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;

    T cross_term = 0.0;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = tx + blockIdx.x * block_size_x * tile_size_x;
    int j = ty + blockIdx.y * block_size_y * tile_size_y;

    #if use_shared_mem == 1
    __shared__ T sh_A[dim][block_size_x*tile_size_x];
    __shared__ T sh_B[dim][block_size_y*tile_size_y];
    __shared__ T sh_scale_A[block_size_x*tile_size_x];
    __shared__ T sh_scale_B[block_size_y*tile_size_y];

    #pragma unroll
    for (int d=0; d<dim; d++) {
        fill_shared_mem_tiled_1D<tile_size_x, dim*block_size_x, dim*block_size_x>(sh_A[d], A+d, tx, i, m);
    }
    fill_shared_mem_tiled_1D<tile_size_x, block_size_x, block_size_x>(sh_scale_A, scale_A, tx, i, m);
    __syncthreads();
    #endif

    int step_size = gridDim.y * block_size_y * tile_size_y;
    for (int sj = j; sj < n; sj += step_size) {

        #if use_shared_mem == 1
        for (int d=0; d<dim; d++) {
            fill_shared_mem_tiled_1D<tile_size_y, dim*block_size_y, dim*block_size_y>(sh_B[d], B+d, ty, sj, n);
        }
        fill_shared_mem_tiled_1D<tile_size_y, block_size_y, block_size_y>(sh_scale_B, scale_B, ty, sj, n);
        __syncthreads();
        #endif

        #if use_shared_mem == 0
            cross_term += compute_expdist_block<double, 2>(i, sj, A, B, scale_A, scale_B, m, n);
        #elif use_shared_mem == 1
            cross_term += compute_expdist_block_shared<double, 2>(tx, ty, i, sj, sh_A, sh_B, sh_scale_A, sh_scale_B, m, n);
            __syncthreads();
        #endif

    }

    //reduce cross_term within the block
    cross_term = BlockReduce(temp_storage).Sum(cross_term);

    //write back the per-thread block partial cross term
    if (tx == 0 && ty == 0) {
        d_cross_term[blockIdx.y*gridDim.x+blockIdx.x] = cross_term;
    }
}


extern "C"
__global__ void
ExpDist_column(const double *A, const double *B,
                 int m, int n, const double *scale_A, const double *scale_B, double *cross_term) {

    //2-dimensional with double precision
    ExpDist_tiled_column<double, 2>(A, B, m, n, scale_A, scale_B, cross_term);

}







#ifdef reduce_block_size
 #define block_size reduce_block_size
#else
 #define block_size block_size_x
#endif

/*
 * Reduce the per thread block cross terms computed in the GaussTransform kernel to single value
 *
 * This kernel is designed to run as single-thread block, because the number of terms to reduce is
 * of size n or m, which is expected to be around 2000 or so. The number of items to reduce
 * is passed as the last argument 'nblocks', which corresponds to the number of thread blocks used
 * by the first kernel.
 */
extern "C"
__global__ void reduce_cross_term(double *output, double *d_cross_term, int m, int n, int nblocks) {

    int tx = threadIdx.x;
    // Specialize BlockReduce for a 1D block of block_size threads on type double
    typedef cub::BlockReduce<double, block_size> BlockReduce;
    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double cross_term = 0.0;
    for (int i=tx; i<nblocks; i+=block_size) {
        cross_term += d_cross_term[i];
    }

    //reduce to single value within thread block
    cross_term = BlockReduce(temp_storage).Sum(cross_term);

    //thread 0 writes output
    if (tx == 0) {
        output[0] = cross_term;
    }

}
