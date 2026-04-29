#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void matrixMul(float *a, float *b, float *c, size_t n) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (size_t k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <size of matrices (n for n x n)>\n", argv[0]);
        return -1;
    }
    size_t n = atol(argv[1]);
    if (n <= 0) {
        fprintf(stderr, "Error: The size of matrices should be a positive integer.\n");
        return -1;
    }

    printf("Size of float: %ld\n", sizeof(float));
    size_t bytes = n * n * sizeof(float);

    float *a, *b, *c;

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

    // Initialize matrices a and b
    for (size_t i = 0; i < n * n; i++) {
        a[i] = static_cast<float>(rand()) / RAND_MAX;
        b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    int blockSize = 16; // Define block size (16x16)
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((n + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize);

    // Execute the kernel
    matrixMul<<<numBlocks, threadsPerBlock>>>(a, b, c, n);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaDeviceSynchronize returned error code %d after launching the kernel!\n", err);
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
        return -1;
    }

    // Verification
    for (size_t row = 0; row < n; row++) {
        for (size_t col = 0; col < n; col++) {
            float expected_value = 0.0f;
            for (size_t k = 0; k < n; k++) {
                expected_value += a[row * n + k] * b[k * n + col];
            }
            if (fabs(c[row * n + col] - expected_value) > 1e-5) {
                printf("Error: value mismatch at row %ld, col %ld\n", row, col);
                cudaFree(a);
                cudaFree(b);
                cudaFree(c);
                return -1;
            }
        }
    }

    printf("Matrix multiplication completed successfully.\n");

    // Free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}
