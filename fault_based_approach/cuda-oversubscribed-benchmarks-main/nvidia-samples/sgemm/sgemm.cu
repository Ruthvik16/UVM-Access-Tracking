/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
* Matrix multiplication: C = A * B.
* Host code.
*
* This sample implements matrix multiplication as described in Chapter 3
* of the programming guide and uses the CUBLAS library to demonstrate
* the best performance.

* SOME PRECAUTIONS:
* IF WE WANT TO CALCULATE ROW-MAJOR MATRIX MULTIPLY C = A * B,
* WE JUST NEED CALL CUBLAS API IN A REVERSE ORDER: cublasSegemm(B, A)!
* The reason is explained as follows:

* CUBLAS library uses column-major storage, but C/C++ use row-major storage.
* When passing the matrix pointer to CUBLAS, the memory layout alters from
* row-major to column-major, which is equivalent to an implicit transpose.

* In the case of row-major C/C++ matrix A, B, and a simple matrix multiplication
* C = A * B, we can't use the input order like cublasSgemm(A, B)  because of
* implicit transpose. The actual result of cublasSegemm(A, B) is A(T) * B(T).
* If col(A(T)) != row(B(T)), equal to row(A) != col(B), A(T) and B(T) are not
* multipliable. Moreover, even if A(T) and B(T) are multipliable, the result C
* is a column-based cublas matrix, which means C(T) in C/C++, we need extra
* transpose code to convert it to a row-based C/C++ matrix.

* To solve the problem, let's consider our desired result C, a row-major matrix.
* In cublas format, it is C(T) actually (because of the implicit transpose).
* C = A * B, so C(T) = (A * B) (T) = B(T) * A(T). Cublas matrice B(T) and A(T)
* happen to be C/C++ matrice B and A (still because of the implicit transpose)!
* We don't need extra transpose code, we only need alter the input order!
*
* CUBLAS provides high-performance matrix multiplication.
* See also:
* V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
* in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
* Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
*/

// CUDA runtime
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cuda-macros-v1.h"

#ifndef min
#define min(a, b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a, b) ((a > b) ? a : b)
#endif

unsigned do_random_init = 0;
unsigned long mat_size  = 100;
unsigned copy_back_output = 1;
unsigned compare_with_cpu = 0;
bool     accby_gpu_after_init_all = false;
bool     accby_gpu_after_init_right_matrix = false;
unsigned long pin_b_partially_mb = 0;

// Optional Command-line multiplier for matrix sizes
typedef struct _matrixSize
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on CPU
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            double sum = 0;

            for (unsigned int k = 0; k < wA; ++k) {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }

            C[i * wB + j] = (float)sum;
        }
}

// Allocates a matrix with random float entries.
void randomInit(float *data, int size)
{
    if (do_random_init)
        for (int i = 0; i < size; ++i)
            data[i] = rand() / (float)RAND_MAX;
    else {
        float rand_seed = rand() / (float)RAND_MAX;
        for (int i = 0; i < size; ++i)
            data[i] = (rand_seed + i) * (i - 1);
    }
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i, j, k;
    int error_count = 0;

    for (j = 0; j < height; j++) {
        if (error_count < iListLength) {
            printf("\n  Row %d:\n", j);
        }

        for (i = 0; i < width; i++) {
            k           = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);

            if (fDiff > fListTol) {
                if (error_count < iListLength) {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }

                error_count++;
            }
        }
    }

    printf(" \n  Total Errors = %d\n", error_count);
}

/*
 * iSizeMultiple is the array dimension.
 */
void initializeCUDA(int argc, char **argv, int &devID, int iSizeMultiple, sMatrixSize &matrix_size)
{
    // By default, we use device 0, otherwise we override the device ID based on
    // what is provided at the command line
    iSizeMultiple = min(iSizeMultiple, 10);
    iSizeMultiple = max(iSizeMultiple, 1);

    int block_size = 32;

    matrix_size.uiWA = 3 * block_size * iSizeMultiple;
    matrix_size.uiHA = 4 * block_size * iSizeMultiple;
    matrix_size.uiWB = 2 * block_size * iSizeMultiple;
    matrix_size.uiHB = 3 * block_size * iSizeMultiple;
    matrix_size.uiWC = 2 * block_size * iSizeMultiple;
    matrix_size.uiHC = 4 * block_size * iSizeMultiple;

    printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n",
           matrix_size.uiHA,
           matrix_size.uiWA,
           matrix_size.uiHB,
           matrix_size.uiWB,
           matrix_size.uiHC,
           matrix_size.uiWC);

    if (matrix_size.uiWA != matrix_size.uiHB || matrix_size.uiHA != matrix_size.uiHC
        || matrix_size.uiWB != matrix_size.uiWC) {
        printf("ERROR: Matrix sizes do not match!\n");
        exit(-1);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test matrix multiply using CUBLAS
////////////////////////////////////////////////////////////////////////////////
void matrixMultiply(unsigned long arr_dim) {

    int block_size = 32;
    sMatrixSize matrix_size;

    // set seed for rand()
    srand(2006);

    matrix_size.uiWA = arr_dim;
    matrix_size.uiHA = arr_dim;
    matrix_size.uiWB = arr_dim;
    matrix_size.uiHB = arr_dim;
    matrix_size.uiWC = arr_dim;
    matrix_size.uiHC = arr_dim;

    printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n",
           matrix_size.uiHA,
           matrix_size.uiWA,
           matrix_size.uiHB,
           matrix_size.uiWB,
           matrix_size.uiHC,
           matrix_size.uiWC);

    // allocate host memory for matrices A and B
    unsigned long size_A     = matrix_size.uiWA * matrix_size.uiHA;
    unsigned long size_B     = matrix_size.uiWB * matrix_size.uiHB;
    float *u_A, *u_B, *u_C;

    UVM_ALLOC_ARR(float, u_A, size_A);
    UVM_ALLOC_ARR(float, u_B, size_B);
    mickey_clear();
    mickey_register_va(u_A + (3 << 21));
    mickey_register_va(u_B + (3 << 21));

    // set seed for rand()
    srand(2006);

    unsigned long size_C     = matrix_size.uiWC * matrix_size.uiHC;
    unsigned long mem_size_C = sizeof(float) * size_C;
    UVM_ALLOC_ARR(float, u_C, size_C);
    mickey_register_va(u_C + (3 << 21));

    clock_t start, end;
    start = clock();
    randomInit(u_A, size_A);
    randomInit(u_B, size_B);
    end = clock();
    printf("%s: init A[] B[]: %.3f s\n", __FILE__, ((float)(end - start)) / CLOCKS_PER_SEC); 

#ifdef CUDA_CLI_HINTS
    HINTS_POST_INIT(u_A, size_A * sizeof(u_A[0]), 1);
    HINTS_POST_INIT(u_B, size_A * sizeof(u_A[0]), 2);
    HINTS_POST_INIT(u_C, size_A * sizeof(u_A[0]), 3);
#endif

    if (accby_gpu_after_init_right_matrix) {
        CUDA_ACCESSED_BY_GPU_HINT(u_B, size_B * sizeof(u_A[0]));
    } else if (accby_gpu_after_init_all) {
        CUDA_ACCESSED_BY_GPU_HINT(u_A, size_A * sizeof(u_A[0]));
        CUDA_ACCESSED_BY_GPU_HINT(u_B, size_B * sizeof(u_A[0]));
        CUDA_ACCESSED_BY_GPU_HINT(u_C, size_C * sizeof(u_A[0]));
    }

    if (pin_b_partially_mb) {
        long nbytes = pin_b_partially_mb << 21;
        char *next = (char *) u_B + nbytes;
        CUDA_HINT(u_B, nbytes,
                  cudaMemAdviseUnsetReadMostly, 0, "u_B");
        CUDA_HINT(u_B, nbytes,
                  cudaMemAdviseUnsetPreferredLocation, 0, "u_B");
        CUDA_ACCESSED_BY_GPU_HINT(u_B, nbytes);

        squidward_report_buf_name(u_B, nbytes, "u_B1", SQUIDWARD_DEFAULT);
        squidward_report_buf_name(next, nbytes, "u_B2", SQUIDWARD_DEFAULT);
    }

    // setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(matrix_size.uiWC / threads.x, matrix_size.uiHC / threads.y);

    const float    alpha = 1.0f;
    const float    beta  = 0.0f;
    cublasHandle_t handle;
    // cudaEvent_t    start, stop;

    cublasStatus_t st = cublasCreate(&handle);
    assert(st == CUBLAS_STATUS_SUCCESS);


    cublasStatus_t status;
    start = clock();
    status = cublasSgemm(handle,
                         CUBLAS_OP_N,
                         CUBLAS_OP_N,
                         matrix_size.uiWB,
                         matrix_size.uiHA,
                         matrix_size.uiWA,
                         &alpha,
                         u_B,
                         matrix_size.uiWB,
                         u_A,
                         matrix_size.uiWA,
                         &beta,
                         u_C,
                         matrix_size.uiWB);
    assert(status == CUBLAS_STATUS_SUCCESS);
    st = cublasDestroy(handle);
    assert(st == CUBLAS_STATUS_SUCCESS);

    CHECK_RETURN_VALUE(cudaDeviceSynchronize());
    end = clock();
    printf("%s: GPU run time: %.3f s\n", __FILE__, ((float)(end - start)) / CLOCKS_PER_SEC); 

#ifdef CUDA_CLI_HINTS
    HINTS_POST_COMPUTE(u_C, size_A * sizeof(u_A[0]), 3);
#endif

    if (copy_back_output)
        TOUCH_ARRAY(u_C, mem_size_C);

    // clean up memory
    cudaFree(u_A);
    cudaFree(u_B);
    cudaFree(u_C);

}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("[Matrix Multiply CUBLAS] - Starting...\n");

    // int         devID = 0, sizeMult = 5;
    unsigned long mem_footprint = (1 << 21);
    unsigned long data_mb = 0;

    for (int i = 0; i < argc; i++) {
        GET_INT_FLAG(i, "-data", mem_footprint);
        GET_INT_FLAG(i, "-mb", data_mb);
        GET_BOOL_FLAG(i, "-copy-back", copy_back_output, true);
        GET_BOOL_FLAG(i, "-random", do_random_init, 1);
        GET_BOOL_FLAG(i, "-acc-by-all", accby_gpu_after_init_all, true);
        GET_BOOL_FLAG(i, "-acc-by-b", accby_gpu_after_init_right_matrix, true);
        GET_INT_FLAG(i, "-pin-mb-b", pin_b_partially_mb);
        get_hints(i, argv);
        if (strcmp(argv[i], "-h") == 0) {
            return -1;
        }
        UNRECOGNIZED_ARGUMENT(i);
    }
    if (data_mb)
        mem_footprint = data_mb * 1000000;
    unsigned long arr_size = mem_footprint / (3 * sizeof(float));
    unsigned long arr_dim  = (unsigned long) (sqrt(arr_size));
    printf("%s: array is %lu * %lu | memory %lu MB\n", __FILE__, arr_dim, arr_dim,
        mem_footprint / 1000000);

    matrixMultiply(arr_dim);

    return 0;
}
