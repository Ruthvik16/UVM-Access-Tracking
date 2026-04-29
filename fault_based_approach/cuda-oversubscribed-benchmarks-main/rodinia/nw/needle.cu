/*
 * Needleman-Wunsch Protein Alignment Algorithm
 *
 * Sources: Rodinia, UVMBench
 */

#define LIMIT -999
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "needle.h"
#include <cuda.h>
#include <sys/time.h>

// includes, kernels
#include "needle_kernel.cu"
#include "tracking.h"

#if 1 /* Two pairs of kernels, 16 and 64 threads. */

#undef BLOCK_SIZE
#define BLOCK_SIZE 64
#include "needle_kernel.cu"
#undef BLOCK_SIZE

#endif

#include "cuda-macros-v1.h"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
int runTest(int argc, char** argv);


int blosum62[24][24] = {
    { 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
    {-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
    {-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
    {-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
    { 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
    {-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
    {-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
    { 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
    {-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
    {-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
    {-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
    {-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
    {-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
    {-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
    {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
    { 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
    { 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
    {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
    {-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
    { 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
    {-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
    {-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
    { 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
    {-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    return runTest(argc, argv);
}

int runTest(int argc, char** argv)
{
    long max_rows, max_cols, penalty = 10;
    long size;
    long data_size = (1 << 22), data_mb = 0;
    bool accessed_by_gpu = false;
    bool accessed_by_gpu_input = false;
    bool accessed_by_gpu_ref = false;
    bool pin_to_cpu = false;
    int  *reference_cuda, *input_itemsets;
    double t1, t2, t3, t4;
    bool do_optimal_10G = false, block_size_64 = false;
    long block_size = 16;

    for (int i = 1; i < argc; i++) {
        GET_INT_FLAG(i, "-data", data_size);
        GET_INT_FLAG(i, "-penalty", penalty);
        GET_INT_FLAG(i, "-mb", data_mb);
        GET_BOOL_FLAG(i, "-accessed-by", accessed_by_gpu, true);
        GET_BOOL_FLAG(i, "-pin-cpu", pin_to_cpu, true);
        GET_BOOL_FLAG(i, "-accessed-by-input", accessed_by_gpu_input, true);
        GET_BOOL_FLAG(i, "-accessed-by-ref", accessed_by_gpu_ref, true);
        GET_BOOL_FLAG(i, "-optimal", do_optimal_10G, true);
        GET_BOOL_FLAG(i, "-threads-64", block_size_64, true);
        get_hints(i, argv);
        if (strncmp(argv[i], "-h", 2) == 0)
            return 1;
        UNRECOGNIZED_ARGUMENT(i);
    }
#ifdef TRACKING_ENABLED
	void*** d_l1;
	init_tracking(&d_l1);
#endif

    if (argv[1] && (argv[1][0] != '-')) {
        data_size = atol(argv[1]);
        assert(data_size);
    }
    if (data_mb) {
        assert((data_mb >> 15) == 0);
        data_size = data_mb * 1000000;
    }
    if (block_size_64 || do_optimal_10G) {
        block_size = 64;
        printf("# Changed block size to 64 (originally 16)\n");
    }
    assert(block_size == 16 || block_size == 64);
    printf("WG size of kernel = %lu\n", block_size);

    long arr_dim = get_data_size(data_size/2, sizeof(int), 2);
    arr_dim -= (arr_dim % block_size);

    max_rows = arr_dim + 1;
    max_cols = arr_dim + 1;
    printf("# Array dims: %ld * %ld\n", max_rows, max_cols);

    size = max_cols * max_rows;

    UVM_ALLOC_ARR(int, reference_cuda, size);
    UVM_ALLOC_ARR(int, input_itemsets, size);
    if (accessed_by_gpu_ref || do_optimal_10G) {
        if (do_optimal_10G && (data_size > (11UL << 30)))
            fprintf(stderr, "Warning: input_itemsets[] exceeds GPU size\n");
        CUDA_ACCESSED_BY_GPU_HINT(reference_cuda, size * sizeof(int));
    } else if (accessed_by_gpu_input) {
        CUDA_ACCESSED_BY_GPU_HINT(input_itemsets, size * sizeof(int));
    } else if (accessed_by_gpu) {
        CUDA_ACCESSED_BY_GPU_HINT(reference_cuda, size * sizeof(int));
        CUDA_ACCESSED_BY_GPU_HINT(input_itemsets, size * sizeof(int));
    } else if (pin_to_cpu) {
        CUDA_PIN_CPU_HINT(reference_cuda, size * sizeof(int));
        CUDA_PIN_CPU_HINT(input_itemsets, size * sizeof(int));
    }

    mickey_clear();
    mickey_register_va(input_itemsets + (1 << 25));
    mickey_register_va(input_itemsets + (1 << 31));
    mickey_register_va(reference_cuda + (1 << 25));
    mickey_register_va(reference_cuda + (1 << 31));

    srand(7);

    t1 = gettime();
    for (int i = 0 ; i < max_cols; i++)
    {
        for (int j = 0 ; j < max_rows; j++)
        {
            input_itemsets[i*max_cols+j] = 0;
        }
    }

    printf("Start Needleman-Wunsch\n");

    for(int i=1; i< max_rows ; i++) {    //please define your own sequence.
        input_itemsets[i*max_cols] = rand() % 10 + 1;
    }
    for(int j=1; j< max_cols ; j++) {    //please define your own sequence.
        input_itemsets[j] = rand() % 10 + 1;
    }


    for (int i = 1 ; i < max_cols; i++){
        for (int j = 1 ; j < max_rows; j++){
            reference_cuda[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
        }
    }

    for(int i = 1; i< max_rows ; i++)
        input_itemsets[i*max_cols] = -i * penalty;
    for(int j = 1; j< max_cols ; j++)
        input_itemsets[j] = -j * penalty;

    t2 = gettime();
    printf("# init input_itemsets[] reference_cuda[]: %.3lf s\n", t2 - t1);

#ifdef CUDA_CLI_HINTS
    HINTS_POST_INIT(reference_cuda, size * sizeof(int), 1);
    HINTS_POST_INIT(input_itemsets, size * sizeof(int), 2);
#endif

    dim3 dimGrid;
    dim3 dimBlock(block_size, 1);
    int block_width = (max_cols - 1)/block_size;

    void (*kernel_1)(int *, int *, int, int, int, int);
    void (*kernel_2)(int *, int *, int, int, int, int);

    if (block_size == 16) {
        kernel_1 = needle_cuda_shared_1_16;
        kernel_2 = needle_cuda_shared_2_16;
    } else if (block_size == 64) {
        kernel_1 = needle_cuda_shared_1_64;
        kernel_2 = needle_cuda_shared_2_64;
    } else {
        assert(0);
    }

    printf("Processing top-left matrix\n");
    //process top-left matrix
    for(int i = 1 ; i <= block_width ; i++){
        dimGrid.x = i;
        dimGrid.y = 1;
        (*kernel_1)<<<dimGrid, dimBlock>>>(reference_cuda, input_itemsets
                ,max_cols, penalty, i, block_width);
    }

    CHECK_RETURN_VALUE(cudaDeviceSynchronize());
    t3 = gettime();
    printf("# kernel1 took %.3lf s\n", t3 - t2);

    printf("Processing bottom-right matrix\n");
    //process bottom-right matrix
    for(int i = block_width - 1 ; i >= 1 ; i--){
        dimGrid.x = i;
        dimGrid.y = 1;
        (*kernel_2)<<<dimGrid, dimBlock>>>(reference_cuda, input_itemsets
                ,max_cols, penalty, i, block_width);
    }
    CHECK_RETURN_VALUE(cudaDeviceSynchronize());

    t4 = gettime();
    printf("# kernel2 took %.3lf s\n", t4 - t3);

//#define TRACEBACK
#ifdef TRACEBACK

    FILE *fpo = fopen("result.txt","w");
    fprintf(fpo, "print traceback value GPU:\n");

    for (int i = max_rows - 2, j = max_rows - 2; i>=0, j>=0;){
        int nw, n, w, traceback;
        if (i == max_rows - 2 && j == max_rows - 2)
            fprintf(fpo, "%d ", input_itemsets[ i * max_cols + j]); //print the first element
        if (i == 0 && j == 0)
            break;
        if (i > 0 && j > 0){
            nw = input_itemsets[(i - 1) * max_cols + j - 1];
            w  = input_itemsets[ i * max_cols + j - 1 ];
            n  = input_itemsets[(i - 1) * max_cols + j];
        }
        else if (i == 0){
            nw = n = LIMIT;
            w  = input_itemsets[ i * max_cols + j - 1 ];
        }
        else if (j == 0){
            nw = w = LIMIT;
            n  = input_itemsets[(i - 1) * max_cols + j];
        }
        else{
        }

        //traceback = maximum(nw, w, n);
        int new_nw, new_w, new_n;
        new_nw = nw + reference_cuda[i * max_cols + j];
        new_w = w - penalty;
        new_n = n - penalty;

        traceback = maximum(new_nw, new_w, new_n);
        if(traceback == new_nw)
            traceback = nw;
        if(traceback == new_w)
            traceback = w;
        if(traceback == new_n)
            traceback = n;

        fprintf(fpo, "%d ", traceback);

        if(traceback == nw)
        {i--; j--; continue;}

        else if(traceback == w)
        {j--; continue;}

        else if(traceback == n)
        {i--; continue;}

        else
            ;
    }

    fclose(fpo);

#endif

    cudaFree(reference_cuda);
    cudaFree(input_itemsets);
#ifdef TRACKING_ENABLED
    export_binary(d_l1, "access_log.bin");
    cudaFree(d_l1);
#endif

    return EXIT_SUCCESS;
}
