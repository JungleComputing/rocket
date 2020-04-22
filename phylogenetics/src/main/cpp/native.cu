#include <iostream>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

typedef uint32_t kmer_t;
const kmer_t INVALID_KMER = ~0;

template <typename T>
__host__ __device__ T powi(T a, T b) {
    T result = 1;

    while (b > 0) {
        result *= a;
        b--;
    }

    return result;
}

template <typename T>
T div_ceil(T a, T b) {
    return (a / b) + (a % b == 0 ? 0 : 1);
}

/**
 * Calculates the scratch memory required to store n elements of type T.
 * Memory is aligned 256-byte segments for good performance on the GPU.
 */
template <typename T>
size_t allocate_scratch(size_t n) {
    return (n * sizeof(T) + 255) / 256 * 256;
}

/**
 * Takes memory for n elements of type T from scratch and advance
 * the scratch memory pointer forward to the next available bytes.
 */
template <typename T>
T* advance_scratch(void **scratch, size_t n) {
    void* current = *scratch;

    size_t bytes = allocate_scratch<T>(n);
    *scratch = (void*)(((char*) *scratch) + bytes);

    return (T*) current;
}

__constant__ uint8_t DNA2DIGIT_MAPPING[26] = {
    0, // A
    255, // B
    1, // C
    255, // D
    255, // E
    255, // F
    2, // G
    255, // H
    255, // I
    255, // J
    255, // K
    255, // L
    255, // M
    255, // N
    255, // O
    255, // P
    255, // Q
    255, // R
    255, // S
    3, // T
    255, // U
    255, // V
    255, // W
    255, // X
    255, // Y
    255 // Z
};

struct DNAAlphabet {
    static const size_t NUM_SYMBOLS = 4;


    /**
     * convert an ASCII DNA representation to its 2-bit symbol
     * Based on nvbio/dna.h:78
     */
    __device__ static uint8_t char_to_digit(const char c) {
        if (c >= 'A' && c <= 'Z') {
            return DNA2DIGIT_MAPPING[c - 'A'];
        } else {
            return 0xff;
        }
    }
};

__constant__ uint8_t PROTEIN2DIGIT_MAPPING[26] = {
    0, // A
    255, // B
    1, // C
    2, // D
    3, // E
    4, // F
    5, // G
    6, // H
    7, // I
    255, // J
    8, // K
    9, // L
    10, // M
    11, // N
    255, // O
    12, // P
    13, // Q
    14, // R
    15, // S
    16, // T
    255, // U
    17, // V
    18, // W
    20, // X (Uknown protein)
    19, // Y
    255 // Z
};


struct ProteinAlphabet {
    static const size_t NUM_SYMBOLS = 21;


    /**
     * convert an ASCII character to a 5-bit symbol
     * Based on nvbio/alphabet_inl.h:90
     */
    __device__ static uint8_t char_to_digit(const char c) {
        if (c >= 'A' && c <= 'Z') {
            return PROTEIN2DIGIT_MAPPING[c - 'A'];
        } else {
            return 0xff;
        }
    }
};


template <typename Alphabet>
size_t scratch_build_composition_vector(int k) {
    size_t k0 = powi(Alphabet::NUM_SYMBOLS, (size_t) k - 0);
    size_t k1 = powi(Alphabet::NUM_SYMBOLS, (size_t) k - 1);
    size_t k2 = powi(Alphabet::NUM_SYMBOLS, (size_t) k - 2);

    size_t scratch_size = 0;
    scratch_size += allocate_scratch<uint32_t>(k0);
    scratch_size += allocate_scratch<uint32_t>(k1);
    scratch_size += allocate_scratch<uint32_t>(k2);

    scratch_size += allocate_scratch<uint32_t>(1);
    scratch_size += allocate_scratch<uint32_t>(1);
    scratch_size += allocate_scratch<uint32_t>(1);
    scratch_size += allocate_scratch<uint32_t>(1);

    scratch_size += allocate_scratch<double>(1);

    size_t cub_scratch = 0;
    cudaError_t err = cub::DeviceReduce::Sum(
            (void*) NULL,
            cub_scratch,
            (uint32_t*) NULL,
            (uint32_t*) NULL,
            k0);
    if (err != cudaSuccess) return err;

    size_t x = 0;
    err = cub::DeviceReduce::Sum(
            (void*) NULL,
            x,
            (double*) NULL,
            (double*) NULL,
            k0);
    if (err != cudaSuccess) return err;
    if (x > cub_scratch) cub_scratch = x;

    err =  cub::DeviceSelect::If(
            (void*) NULL,
            x,
            (uint32_t*) NULL,
            (uint32_t*) NULL,
            (uint32_t*) NULL,
            k0,
            [=] __device__ (uint32_t i) { return false; });
    if (err != cudaSuccess) return err;
    if (x > cub_scratch) cub_scratch = x;

    scratch_size += allocate_scratch<uint8_t>(cub_scratch);

    return scratch_size;
}

template <int i, int j, bool flag = i < j>
struct unroll_helper {
    template <typename F>
    __device__ static void call(F fun) {
        if (fun(i)) {
            unroll_helper<i + 1, j>::call(fun);
        }
    }
};

template <int i, int j>
struct unroll_helper<i, j, false> {
    template <typename F>
    __device__ static void call(F fun) {
        //
    }
};

template <int N, typename F>
__device__ void unroll(F fun) {
    unroll_helper<0, N>::call(fun);
}

template <typename Alphabet, int K>
cudaError_t count_kmers(
        cudaStream_t stream,
        const char* d_string,
        uint32_t string_len,
        uint32_t *d_k0_count,
        uint32_t *d_k1_count,
        uint32_t *d_k2_count
) {
    auto exec = thrust::cuda::par.on(stream);

    thrust::for_each(
            exec, 
            thrust::make_counting_iterator<uint32_t>(0u),
            thrust::make_counting_iterator<uint32_t>(string_len),

            [=] __device__ (uint32_t i) {
                kmer_t w = 0;

                unroll<K + 1>([=, &w](uint32_t j) {
                    // Compiler should remove the conditionals when unrolling
                    // the loop (fingers crossed!).
                    if (j == K - 2) atomicAdd(&d_k2_count[w], 1);
                    if (j == K - 1) atomicAdd(&d_k1_count[w], 1);
                    if (j == K - 0) atomicAdd(&d_k0_count[w], 1);

                    if (i + j >= string_len) return false;
                    uint8_t c = Alphabet::char_to_digit(d_string[i + j]);
                    if (c == 0xff) return false;

                    w = w * Alphabet::NUM_SYMBOLS + c;
                    return true;
                });
            });

    return cudaSuccess;
}

template <typename Alphabet>
cudaError_t build_composition_vector(
    cudaStream_t stream,
    void *d_scratch,
    size_t scratch_size,
    int k,
    const char* d_string,
    uint32_t string_len,
    kmer_t *d_keys,
    float *d_values,
    uint32_t *num_unique,
    uint32_t max_size
) {
    cudaError_t err = cudaSuccess;

    size_t k0 = powi(Alphabet::NUM_SYMBOLS, (size_t) k - 0);
    size_t k1 = powi(Alphabet::NUM_SYMBOLS, (size_t) k - 1);
    size_t k2 = powi(Alphabet::NUM_SYMBOLS, (size_t) k - 2);

    uint32_t *d_k0_count = advance_scratch<kmer_t>(&d_scratch, k0);
    uint32_t *d_k1_count = advance_scratch<kmer_t>(&d_scratch, k1);
    uint32_t *d_k2_count = advance_scratch<kmer_t>(&d_scratch, k2);

    uint32_t *d_n0 = advance_scratch<uint32_t>(&d_scratch, 1);
    uint32_t *d_n1 = advance_scratch<uint32_t>(&d_scratch, 1);
    uint32_t *d_n2 = advance_scratch<uint32_t>(&d_scratch, 1);
    uint32_t *d_num_unique = advance_scratch<uint32_t>(&d_scratch, 1);

    double *d_norm = advance_scratch<double>(&d_scratch, 1);

    // initialize k-mer count table with zeros
    auto exec = thrust::cuda::par.on(stream);
    thrust::fill_n(exec, d_k0_count, k0, 0);
    thrust::fill_n(exec, d_k1_count, k1, 0);
    thrust::fill_n(exec, d_k2_count, k2, 0);


    // count k-mers of length k, k-1, and k-2
#define SPECIALIZE(K) \
    if (k == K) err = count_kmers<Alphabet, K>( \
            stream, d_string, string_len, d_k0_count, d_k1_count, d_k2_count)

    if (k < 3 || k > 10) {
        fprintf(stderr, "error: k=%d should be in range 3-10\n", k);
        return cudaErrorUnknown;
    }
    SPECIALIZE(3);
    SPECIALIZE(4);
    SPECIALIZE(5);
    SPECIALIZE(6);
    SPECIALIZE(7);
    SPECIALIZE(8);
    SPECIALIZE(9);
    SPECIALIZE(10);

    if (err != cudaSuccess) return err;

#undef SPECIALIZE

    // sum the number of k-mers of length k
    err = cub::DeviceReduce::Sum(
            d_scratch,
            scratch_size,
            d_k0_count,
            d_n0,
            k0,
            stream);
    if (err != cudaSuccess) return err;

    // sum the number of k-mers of length k-1
    err = cub::DeviceReduce::Sum(
            d_scratch,
            scratch_size,
            d_k1_count,
            d_n1,
            k1,
            stream);
    if (err != cudaSuccess) return err;

    // sum the number of k-mers of length k-2
    err = cub::DeviceReduce::Sum(
            d_scratch,
            scratch_size,
            d_k2_count,
            d_n2,
            k2,
            stream);
    if (err != cudaSuccess) return err;

    // following function implements formulas (2) and (3) from "Whole Proteome 
    // Prokaryote Phylogeny Without Sequence Alignment:AK-String Composition 
    // Approach" (2004) by Qi et al.
    thrust::for_each(
            exec, 
            thrust::make_counting_iterator<uint32_t>(0u),
            thrust::make_counting_iterator<uint32_t>(k0),
            [=] __device__ (uint32_t i) {
                // full = a_1 a_2 ... a_{k-1} a_k (complete k-mer)
                // prefix = a_1 a_2 ... a_{k-1} ({k-1}-mer without last character)
                // suffix = a_2 ... a_{k-1} a_k ({k-1}-mer without first character)
                // middle = a_2 ... a_{k-1} ({k-2}-mer without first and last char.)
                kmer_t full = i;
                kmer_t prefix = full / Alphabet::NUM_SYMBOLS;
                kmer_t suffix = full % powi((int) Alphabet::NUM_SYMBOLS, k - 1);
                kmer_t middle = suffix / Alphabet::NUM_SYMBOLS;

                // You are probably wonder what this strange do-while statement is doing
                // here. Formula (3) gives that alpha = 0 if either middle_count == 0 or
                // prefix_count == 0 or suffix_count == 0. This means we read each of these
                // counts in any order and immediately break if any of them is zero.
                // The counts are read in decending order of locality (i.e., increasing 
                // order of cost): first middle_count, then prefix_count, then suffix_count.
                float alpha = 0.0;
                do {
                    uint32_t middle_count = d_k2_count[middle];
                    if (middle_count == 0) break;

                    uint32_t prefix_count = d_k1_count[prefix];
                    if (prefix_count == 0) break;

                    uint32_t suffix_count = d_k1_count[suffix];
                    if (suffix_count == 0) break;

                    uint32_t full_count = d_k0_count[full];

                    uint32_t n0 = *d_n0;
                    uint32_t n1 = *d_n1;
                    uint32_t n2 = *d_n2;

                    // Formula (1) = count / n
                    double p = double(full_count) / n0;

                    // Formula (2) = P(prefix) * P(suffix) / P(middle)
                    double p0 = ((double(prefix_count) / n1) * (double(suffix_count) / n1)) / (double(middle_count) / n2);

                    // Formula (3) = (p - p0) / p0
                    alpha = float((p - p0) / p0);

                } while (0);

                // Overwrite int by taking the int32 interpretation of float.
                d_k0_count[i] = __float_as_int(alpha);
            });

    // count entries where alpha != 0
    err = cub::DeviceReduce::Sum(
            d_scratch,
            scratch_size,
            thrust::make_transform_iterator(
                d_k0_count, 
                [=] __device__ (uint32_t p) { return p != 0; }),
            d_num_unique,
            k0,
            stream);
    if (err != cudaSuccess) return err;


    // sum p**2 for all entries
    err = cub::DeviceReduce::Sum(
            d_scratch,
            scratch_size,
            thrust::make_transform_iterator(
                d_k0_count, 
                [=] __device__ (uint32_t p) { 
                    float f = __int_as_float(p);
                    return f * f;
                }),
            d_norm,
            k0,
            stream);
    if (err != cudaSuccess) return err;

    err = cudaMemcpyAsync(
            num_unique,
            d_num_unique,
            sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            stream);
    if (err != cudaSuccess) return err;

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return err;

    if (*num_unique > max_size) {
        fprintf(stderr,
                "error: size of composition vector (size: %d) exceeds the given maximum (size: %d)\n",
                *num_unique, 
                max_size);
        return cudaErrorUnknown;
    }

    err =  cub::DeviceSelect::If(
            d_scratch,
            scratch_size,
            thrust::make_counting_iterator<uint32_t>(0u),
            d_keys,
            d_num_unique,
            k0,
            [=] __device__ (uint32_t i) { return d_k0_count[i] != 0; },
            stream);
    if (err != cudaSuccess) return err;

    thrust::for_each(
            exec,
            thrust::make_counting_iterator<uint32_t>(0u),
            thrust::make_counting_iterator<uint32_t>(*num_unique),
            [=] __device__ (uint32_t i) {
                kmer_t index = d_keys[i];
                float f = __int_as_float(d_k0_count[index]);
                d_values[i] = f / sqrt(*d_norm);
            });

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return err;

    thrust::fill(exec, d_keys + *num_unique, d_keys + max_size, INVALID_KMER);
    thrust::fill(exec, d_values + *num_unique, d_values + max_size, 0.0f);

    return cudaSuccess;
}



#ifdef TUNE_THREADS_PER_BLOCK
#define THREADS_PER_BLOCK (TUNE_THREADS_PER_BLOCK)
#define ITEMS_PER_THREAD (TUNE_ITEMS_PER_THREAD)
#define USE_SMEM (TUNE_USE_SMEM)
#else
#define THREADS_PER_BLOCK (256)
#define ITEMS_PER_THREAD (1)
#define USE_SMEM (0)
#endif
#define ITEMS_PER_BLOCK (THREADS_PER_BLOCK * ITEMS_PER_THREAD)

size_t scratch_calculate_cosine_similarity(uint32_t max_vector_size) {
    size_t n = 0;
    size_t num_blocks = div_ceil(
            max_vector_size + max_vector_size, 
            (uint32_t) ITEMS_PER_BLOCK);

    n += allocate_scratch<double>(num_blocks);
    n += allocate_scratch<uint2>(num_blocks + 1);

    size_t x = 0;
    cub::DeviceReduce::Sum(
            (void*) NULL,
            x,
            (double*) NULL,
            (double*) NULL,
            num_blocks);
    n += allocate_scratch<uint8_t>(x);

    return n;
}

template <typename K>
uint2 __device__ __forceinline__ merge_path(
        uint32_t diag,
        const K *__restrict__ left,
        uint32_t left_size,
        const K *__restrict__ right,
        uint32_t right_size
    ) {
    uint32_t begin = diag < right_size ? 0 : diag - right_size;
    uint32_t end = diag < left_size ? diag : left_size;

    while (begin < end) {
        uint32_t mid = (begin + end) / 2 + 1;
        K a = left[mid - 1];
        K b = right[diag - mid];

        if (a <= b) {
            begin = mid;
        } else {
            end = mid - 1;
        }
    }

    int i = min(begin, left_size);
    int j = min(diag - begin, right_size);
    return make_uint2(i, j);
}

__global__ void set_merge_by_key_and_reduce_cosine(
        const uint2 *__restrict__ ranges,
        const kmer_t *__restrict__ left_keys,
        const float *__restrict__ left_values,
        const uint32_t left_size,
        const kmer_t *__restrict__ right_keys,
        const float *__restrict__ right_values,
        const uint32_t right_size,
        double *results
) {
    typedef cub::BlockReduce<double, THREADS_PER_BLOCK> BlockReduce;

    __shared__ union {
        typename BlockReduce::TempStorage temp;
        kmer_t keys[2 * (ITEMS_PER_BLOCK + 1) * !!USE_SMEM];
    } shared;

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // YYY
    uint left_begin = ranges[bid].x;
    uint right_begin = ranges[bid].y;
    uint left_end = ranges[bid + 1].x;
    uint right_end = ranges[bid+ 1].y;
    uint left_span = left_end - left_begin;
    uint right_span = right_end - right_begin;

#if USE_SMEM
#pragma unroll
    for (int i = tid; i < 2 * (ITEMS_PER_BLOCK + 2); i += THREADS_PER_BLOCK) {
        kmer_t key = !0;

        if (left_begin + i <= left_end) {
            key = left_keys[left_begin + i];
        } else {
            int j = i - (left_end - left_begin + 1);

            if (right_begin + j <= right_end) {
                key = right_keys[right_begin + j];
            }
        }

        shared.keys[i] = key;
    }

    __syncthreads();
#endif

#if USE_SMEM
    uint2 mp = merge_path(
            tid * ITEMS_PER_THREAD,
            shared.keys,
            left_span,
            shared.keys + left_span,
            right_span);
#else
    uint2 mp = merge_path(
            tid * ITEMS_PER_THREAD,
            left_keys + left_begin,
            left_span,
            right_keys + right_begin,
            right_span);
#endif

    uint i = mp.x + left_begin;
    uint j = mp.y + right_begin;
    double result = 0.0;

#pragma unroll
    for (int it = 0; it < ITEMS_PER_THREAD; it++) {
        if ((i >= left_end && j >= right_end) || i >= left_size || j >= right_size) {
            break;
        }

        kmer_t p = left_keys[i];
        kmer_t q = right_keys[j];

        if (p == q) {
            double a = left_values[i];
            double b = right_values[j];
            result += a * b;

            //printf("GPU found %d %d (%d == %d): %f * %f == %f\n",
            //        i, j, p, q, a, b, a * b);
        }

        if (p <= q) {
            i++;
        } else {
            j++;
        }
    }


    // Reduce
    result = BlockReduce(shared.temp).Sum(result);

    if (tid == 0) {
        results[bid] = result;
    }
}


cudaError_t calculate_cosine_similarity(
    cudaStream_t stream,
    void *d_scratch,
    size_t scratch_size,
    const kmer_t *d_left_keys,
    const float *d_left_values,
    const uint32_t left_size,
    const kmer_t *d_right_keys,
    const float *d_right_values,
    const uint32_t right_size,
    double *d_result
) {
    cudaError_t err = cudaSuccess;
    size_t num_blocks = div_ceil(left_size + right_size, (uint32_t) ITEMS_PER_BLOCK);

    uint2 *d_ranges = advance_scratch<uint2>(&d_scratch, num_blocks);
    double *d_partial_results = advance_scratch<double>(&d_scratch, num_blocks);

    thrust::transform(
            thrust::cuda::par.on(stream),
            thrust::make_counting_iterator<uint32_t>(0),
            thrust::make_counting_iterator<uint32_t>(num_blocks + 1),
            d_ranges,
            [=] __device__ (uint32_t bid) {
                return merge_path(
                        bid * ITEMS_PER_BLOCK,
                        d_left_keys,
                        left_size,
                        d_right_keys,
                        right_size);
            });

    set_merge_by_key_and_reduce_cosine<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
            d_ranges,
            d_left_keys,
            d_left_values,
            left_size,
            d_right_keys,
            d_right_values,
            right_size,
            d_partial_results);

    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    err = cub::DeviceReduce::Sum(
            d_scratch,
            scratch_size,
            d_partial_results,
            d_result,
            num_blocks,
            stream);
    if (err != cudaSuccess) return err;

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

extern "C" long estimateScratchMemory(
        char *alphabet,
        int k,
        int max_vector_size
) {
    size_t a = strcmp(alphabet, "DNA") == 0 ?
        scratch_build_composition_vector<DNAAlphabet>(k) :
        scratch_build_composition_vector<ProteinAlphabet>(k);
    size_t b = scratch_calculate_cosine_similarity(max_vector_size);

    return std::max(a, b);
}

extern "C" int buildCompositionVector(
    uintptr_t stream,
    uintptr_t d_temp_storage_ptr,
    long temp_storage_size,
    char *alphabet,
    int k, 
    uintptr_t d_string_ptr, 
    long string_len, 
    uintptr_t d_set_keys_ptr,
    uintptr_t d_set_values_ptr, 
    uintptr_t set_size_ptr,
    int max_vector_size
) {
#define SPECIALIZE(name, A) \
    if (strcmp(alphabet, name) == 0) { \
        cudaError_t err = build_composition_vector<A>( \
                (cudaStream_t) stream, \
                (void*) d_temp_storage_ptr, \
                temp_storage_size, \
                k, \
                (const char*) d_string_ptr, \
                (uint32_t) string_len, \
                (kmer_t*) d_set_keys_ptr, \
                (float*) d_set_values_ptr, \
                (uint32_t*) set_size_ptr, \
                max_vector_size); \
        return (int) err; \
    }

    SPECIALIZE("DNA", DNAAlphabet);
    SPECIALIZE("protein", ProteinAlphabet);

    fprintf(stderr, "error: invalid alphabet '%s'", alphabet);
    return cudaErrorUnknown;
}


extern "C" double compareCompositionVectors(
    uintptr_t stream,
    uintptr_t d_temp_storage_ptr,
    long temp_storage_size,
    uintptr_t d_left_keys_ptr,
    uintptr_t d_left_values_ptr,
    uint32_t left_size,
    uintptr_t d_right_keys_ptr,
    uintptr_t d_right_values_ptr,
    uint32_t right_size,
    uintptr_t d_output_ptr
) {
    cudaError_t err = calculate_cosine_similarity(
            (cudaStream_t) stream,
            (void*) d_temp_storage_ptr,
            temp_storage_size,
            (const kmer_t*) d_left_keys_ptr,
            (const float*) d_left_values_ptr,
            left_size,
            (const kmer_t*) d_right_keys_ptr,
            (const float*) d_right_values_ptr,
            right_size,
            (double*) d_output_ptr);

    return (int) err;
}

extern "C" float tuneCalculateCosineSimilarity(
    const kmer_t *left_keys,
    const float *left_values,
    const uint32_t left_size,
    const kmer_t *right_keys,
    const float *right_values,
    const uint32_t right_size,
    double *result
) {
    size_t scratch_size = scratch_calculate_cosine_similarity(left_size + right_size);
    cudaStream_t stream;
    cudaEvent_t event_before;
    cudaEvent_t event_after;

    uint32_t max_size = std::max(left_size, right_size);

    thrust::device_vector<double> d_result(1);
    thrust::device_vector<uint8_t> d_scratch(scratch_size);

    thrust::device_vector<kmer_t> d_left_keys(left_keys, left_keys + left_size);
    thrust::device_vector<float> d_left_values(left_values, left_values + left_size);
    thrust::device_vector<uint32_t> d_left_size(1, left_size);
    thrust::device_vector<kmer_t> d_right_keys(right_keys, right_keys + right_size);
    thrust::device_vector<float> d_right_values(right_values, right_values + right_size);
    thrust::device_vector<uint32_t> d_right_size(1, right_size);

    cudaStreamCreate(&stream);
    cudaEventCreate(&event_before);
    cudaEventCreate(&event_after);

    cudaEventRecord(event_before, stream);

    cudaError_t err = calculate_cosine_similarity(
            stream,
            (void*) thrust::raw_pointer_cast(d_scratch.data()),
            scratch_size,
            thrust::raw_pointer_cast(d_left_keys.data()),
            thrust::raw_pointer_cast(d_left_values.data()),
            left_size,
            thrust::raw_pointer_cast(d_right_keys.data()),
            thrust::raw_pointer_cast(d_right_values.data()),
            right_size,
            thrust::raw_pointer_cast(d_result.data()));

    cudaEventRecord(event_after, stream);
    cudaStreamSynchronize(stream);

    float elapsed = 0.0;
    cudaEventElapsedTime(&elapsed, event_before, event_after);

    cudaStreamDestroy(stream);
    cudaEventDestroy(event_before);
    cudaEventDestroy(event_after);

    *result = d_result[0];
    return elapsed;
}

