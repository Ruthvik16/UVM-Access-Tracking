#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void vectorAdd(float *a, float *b, float *c, size_t n) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        c[id] = a[id] + b[id];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <size of vectors in M>\n", argv[0]);
        return -1;
    }
    size_t n = atol(argv[1]);
    n=n*1024*1024; // * 1024 for KB * 1024 for MB
    if (n <= 0) {
        fprintf(stderr, "Error: The size of vectors should be a positive integer.\n");
        return -1;
    }
    printf("Size of float: %ld\n", sizeof(float));
    size_t bytes = n * sizeof(float);
    printf("Will allocate 3 vectors of %ld MiB\n", (bytes >> 20));
    
    float *a,*b,*c;

    cudaError_t err = cudaMallocManaged(&a, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaMallocManaged failed for a: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaMallocManaged(&b, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaMallocManaged failed for b: %s\n", cudaGetErrorString(err));
        cudaFree(a);
        return -1;
    }

    err = cudaMallocManaged(&c, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaMallocManaged failed for c: %s\n", cudaGetErrorString(err));
        cudaFree(a);
        cudaFree(b);
        return -1;
    }
    
    clock_t start, after_array_init, after_kernel, end;
    start = clock();

    for (size_t i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * 2.0f;
    }
    after_array_init = clock();

    int blockSize = 256;

    size_t gridSize = (int)ceil((float)n / blockSize);
    // Execute the kernel
    vectorAdd<<<gridSize, blockSize>>>(a, b, c, n);

    err = cudaDeviceSynchronize();
    after_kernel = clock();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaDeviceSynchronize returned error code %d after launching the kernel!\n", err);
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
        return -1;
    }

    // For CPU faults again
    for (size_t i = 0; i < n; i++) {
        if (fabs(c[i] - (a[i] + b[i])) > 1e-5) { // < 0.00001
            printf("Error: value mismatch at index %ld\n", i);
            return -1;
        }
    }
    end = clock();

    printf("Vector addition completed successfully.\n");
    printf("%s: array_init: %.3f s, compute: %.3f s, check: %.3f s\n",
        __FILE__,
        ((float)(after_array_init - start))/CLOCKS_PER_SEC,
        ((float)(after_kernel - after_array_init))/CLOCKS_PER_SEC,
        ((float)(end - after_kernel))/CLOCKS_PER_SEC
        );


    // Free them
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}
