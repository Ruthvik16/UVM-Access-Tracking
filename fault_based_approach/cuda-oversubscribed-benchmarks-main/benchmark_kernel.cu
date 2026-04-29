#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>


#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Helper to get global thread ID and grid stride
// Index: $i = blockIdx.x \cdot blockDim.x + threadIdx.x$
// Stride: $s = blockDim.x \cdot gridDim.x$

// 1. Coalesced: Ideal pattern with Grid-Stride Loop
__global__ void test_coalesced(int* data, int n) {
    int grid_stride = blockDim.x * gridDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += grid_stride) {
        data[i] = i;
    }
}

// 2. Stride: Forces multiple cache line fetches
__global__ void test_stride(int* data, int n, int mem_stride) {
    int grid_stride = blockDim.x * gridDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += grid_stride) {
        // We use 'i' as the logic index and 'mem_stride' for the memory gap
        data[i * mem_stride] = i;
    }
}

// 3. Random: High latency, no spatial locality
__global__ void test_random(int* data, int* indices, int n) {
    int grid_stride = blockDim.x * gridDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += grid_stride) {
        data[indices[i]] = i;
    }
}

// 4. Stencil: Reads neighbors (L1/L2 reuse)
__global__ void test_stencil(int* in, int* out, int n) {
    int grid_stride = blockDim.x * gridDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += grid_stride) {
        // Boundary check remains necessary within the loop
        if (i > 0 && i < n - 1) {
            out[i] = (in[i-1] + in[i] + in[i+1]) / 3;
        }
    }
}

// 5. Atomics: Serialization overhead
__global__ void test_atomic(int* data, int n) {
    int grid_stride = blockDim.x * gridDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += grid_stride) {
        atomicAdd(&data[0], 1);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <test_id>\n", argv[0]);
        return 1;
    }
    int test_id = atoi(argv[1]);
    int n = 1048576; // 1M elements

    int *d_in, *d_out, *d_indices;
    // Over-allocate for stride (mem_stride * n)
    CUDA_CHECK(cudaMallocManaged(&d_in, n * 64 * sizeof(int))); 
    CUDA_CHECK(cudaMallocManaged(&d_out, n * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&d_indices, n * sizeof(int)));

    std::vector<int> h_indices(n);
    for(int i=0; i<n; i++) h_indices[i] = i;
    std::shuffle(h_indices.begin(), h_indices.end(), std::mt19937(42));
    cudaMemcpy(d_indices, h_indices.data(), n * sizeof(int), cudaMemcpyHostToDevice);

#ifdef TRACKING_ENABLED
    void*** d_l1;
#endif

    // Execution Configuration: Using a fixed grid size
    // 256 blocks is usually enough to saturate most modern GPUs
    dim3 block(256);
    dim3 grid(256); 

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::cout << "Warming up GPU..." << std::endl;
    // Warmup & Select Kernel
    auto launch_kernel = [&](int id) {
        switch(id) {
            case 0: printf("Selected coalesced");test_coalesced<<<grid, block>>>(d_in, n); break;
            case 1: printf("Selected stride");test_stride<<<grid, block>>>(d_in, n, 32); break;
            case 2: printf("Selected random");test_random<<<grid, block>>>(d_in, d_indices, n); break;
            case 3: printf("Selected stencil");test_stencil<<<grid, block>>>(d_in, d_out, n); break;
            case 4: printf("Selected atomic");test_atomic<<<grid, block>>>(d_out, n); break;
        }
    };

    launch_kernel(test_id); // Warmup
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    launch_kernel(test_id); // Timed run
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "BENCHMARK_TIME: " << ms << " ms" << std::endl;

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_indices);
#ifdef TRACKING_ENABLED
    cudaFree(d_l1);
#endif

    
    return 0;
}
