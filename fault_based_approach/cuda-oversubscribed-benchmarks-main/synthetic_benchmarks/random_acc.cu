/*
 * From Tyler Allen's code artifact
 *
 * https://github.com/tallendev/uvm-eval.git
 *
 * Cleaned up by PROSPAR GROUP.
 * Performance - thread block size does not appear to matter.
 * I have commented out the clflush (cache line flush) loop.
 */

#include <stdio.h>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <assert.h>
#include <random>
#include <omp.h>
#include "cuda-macros-v1.h"

// num float in 4k page
#define PSIZE 1024lu

#ifndef THREADS
#define THREADS 64
#endif

#ifndef TASKS_PER_THREAD
#define TASKS_PER_THREAD 100
#endif

#ifdef FLUSH_CPU_CACHE
static inline void clflush2(volatile void *__p)
{
    asm volatile("clflush (%0)" :: "r" (__p));
}
#endif


#if 0
__inline__ __device__ void prefetch_l1 (const void* addr)
{
    asm(" prefetch.global.L1 [ %1 ];": "=l"(addr) : "l"(addr));
}

__inline__ __device__ void prefetch_l2 (const void* addr)
{

    asm(" prefetch.global.L2 [ %1 ];": "=l"(addr) : "l"(addr));
}
#endif

/*
 * One thread (idx) per page.
 * Read a random element of a[]. Random index chosen from b: b[idx].
 * Read-mostly workload.
 */
extern "C"
__global__ void uvmer2(volatile float* a, const __restrict__ size_t* b,
                       unsigned long num_pgs)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pgs)
    {
        float ele = a[b[idx] * PSIZE];
        if (ele == 0.35)
        {
            a[idx] = b[idx + 73];
        }
    }
}

extern "C"
__global__ void uvmer(volatile float* a, float* b, unsigned long num_pgs)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pgs)
    {
        for (int i = 0; i < TASKS_PER_THREAD; i++)
        {
            float ele = a[PSIZE * (idx + (i * 10) * (blockDim.x * gridDim.x))];
            if (ele == 0.35)
            {
                b[idx] = ele;
            }
        }
    }
}

inline void copyIndexData(size_t* b, unsigned long num_pgs)
{
    std::vector<size_t> indexes;
    indexes.reserve(num_pgs);
#pragma omp simd
    for (size_t i = 0; i < num_pgs; i++)
    {
        //indexes[i] = i;
        indexes.push_back(i);
    }
    std::random_shuffle(indexes.begin(), indexes.end());
    cudaMemcpy(b, &indexes[0], num_pgs * sizeof(size_t), cudaMemcpyHostToDevice);
}

int main(int argc, char *argv[])
{
    float *array;
    typedef std::chrono::high_resolution_clock Clock;
    unsigned long data_size = (1 << 23);
    unsigned long data_mb = 0;
    unsigned long thread_block = THREADS;
    unsigned long num_blocks;
    unsigned iters = 1;

    for (int i = 0; i < argc; i++) {
        GET_INT_FLAG(i, "-data", data_size);
        GET_INT_FLAG(i, "-mb",   data_mb);
        GET_INT_FLAG(i, "-thread-block", thread_block);
        GET_INT_FLAG(i, "-iters", iters);
        get_hints(i, argv);
        if (strcmp(argv[i], "-h") == 0)
            return -1;
        UNRECOGNIZED_ARGUMENT(i);
    }

    if (data_mb)
        data_size = data_mb << 20;

    unsigned long arr_len = data_size / sizeof(float);
    unsigned long num_pgs = data_size >> 12;

    num_blocks = (num_pgs + thread_block - 1) / thread_block;

    CHECK_CUDA_ERROR();
    cudaDeviceSynchronize();

    for (int i = 0; i < iters; i++)
    {
        size_t* b;
        NONUVM_ALLOC_ARR(size_t, b, num_pgs);
        copyIndexData(b, num_pgs);

        CHECK_CUDA_ERROR();
        UVM_ALLOC_ARR(float, array, arr_len);

        if (i == iters - 1) {
            mickey_clear();
            mickey_register_va((char *)array + (1UL << 25));
            mickey_register_va((char *)array + (1UL << 32));
        }

        double t_before_init = gettime();
#pragma simd
        for (size_t i = 0; i < arr_len; i++)
        {
            array[i] = 0.0;
        }
#ifdef FLUSH_CPU_CACHE
#pragma omp simd
        for (size_t i = 0; i < arr_len; i++)
        {
            clflush2(array + i);
        }
#endif
        double t_after_init = gettime();

        printf("# init array[]: %.3lf s\n", t_after_init - t_before_init);
        cudaEvent_t start;
        cudaEventCreate(&start);

        cudaEvent_t stop;
        cudaEventCreate(&stop);

        double t_start = gettime();

        // Record the start event
        cudaEventRecord(start, NULL);

        uvmer2<<<num_blocks, thread_block>>>(array, b, num_pgs);
        // Record the stop event
        cudaEventRecord(stop, NULL);

        // Wait for the stop event to complete
        cudaEventSynchronize(stop);
        cudaDeviceSynchronize();
        double t_end = gettime();

        CHECK_CUDA_ERROR();
        float msecTotal = 0.0f;
        cudaEventElapsedTime(&msecTotal, start, stop);

        printf("# kernel (%ld threads, %ld MB): %.3lf s\n",
               thread_block, data_size >> 20, t_end - t_start);

        // should be pages / sec
        printf("perf,%lf\n", (num_blocks * thread_block) / (msecTotal/1000.0));

        CHECK_CUDA_ERROR();
        cudaFree(array);
        cudaFree(b);
        CHECK_CUDA_ERROR();
    }
    CHECK_CUDA_ERROR();
}
