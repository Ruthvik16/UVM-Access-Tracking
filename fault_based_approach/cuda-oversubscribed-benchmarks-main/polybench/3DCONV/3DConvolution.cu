/**
 * 3DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#if __has_include("../../../common/polybenchUtilFuncts.h")
// Because we've copied from UVMBench
#include "../../../common/polybenchUtilFuncts.h"
#endif

#if __has_include("../polybenchUtilFuncts.h")
#include "../polybenchUtilFuncts.h"
#endif

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

/* Problem size */
unsigned long NI, NJ, NK;

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;


void conv3D(DATA_TYPE* A, DATA_TYPE* B, unsigned long NI, unsigned long NJ, unsigned long NK)
{
	long i, j, k;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +2;  c21 = +5;  c31 = -8;
	c12 = -3;  c22 = +6;  c32 = -9;
	c13 = +4;  c23 = +7;  c33 = +10;

	for (i = 1; i < NI - 1; ++i) // 0
	{
		for (j = 1; j < NJ - 1; ++j) // 1
		{
			for (k = 1; k < NK -1; ++k) // 2
			{
				//printf("i:%d\nj:%d\nk:%d\n", i, j, k);
				B[i*(NK * NJ) + j*NK + k] = c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c21 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c23 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c31 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c33 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c12 * A[(i + 0)*(NK * NJ) + (j - 1)*NK + (k + 0)]  +  c22 * A[(i + 0)*(NK * NJ) + (j + 0)*NK + (k + 0)]   
					     +   c32 * A[(i + 0)*(NK * NJ) + (j + 1)*NK + (k + 0)]  +  c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]  
					     +   c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]  +  c21 * A[(i - 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]  
					     +   c23 * A[(i + 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]  +  c31 * A[(i - 1)*(NK * NJ) + (j + 1)*NK + (k + 1)]  
					     +   c33 * A[(i + 1)*(NK * NJ) + (j + 1)*NK + (k + 1)];
			}
		}
	}
}


void init(DATA_TYPE* A, DATA_TYPE* A_gpu, unsigned long NI, unsigned long NJ, unsigned long NK)
{
	long i, j, k;
	double t_start, t_end;
	t_start = gettime();

	for (i = 0; i < NI; ++i)
	{
		for (j = 0; j < NJ; ++j)
		{
			for (k = 0; k < NK; ++k)
			{
				if (compare_with_cpu)
					A[i*(NK * NJ) + j*NK + k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
				A_gpu[i*(NK * NJ) + j*NK + k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
			}
		}
	}
	t_end = gettime();

	printf("init A[]: %.3lf s\n", t_end - t_start);
}


void compareResults(DATA_TYPE* B, DATA_TYPE* B_outputFromGpu, unsigned long NI, unsigned long NJ, unsigned long NK)
{
	long i, j, k, fail;
	fail = 0;
	
	// Compare result from cpu and gpu...
	for (i = 1; i < NI - 1; ++i) // 0
	{
		for (j = 1; j < NJ - 1; ++j) // 1
		{
			for (k = 1; k < NK - 1; ++k) // 2
			{
				if (percentDiff(B[i*(NK * NJ) + j*NK + k], B_outputFromGpu[i*(NK * NJ) + j*NK + k]) > PERCENT_DIFF_ERROR_THRESHOLD)
				{
					fail++;
				}
			}	
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %ld\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}


__global__ void convolution3D_kernel(DATA_TYPE *A, DATA_TYPE *B, int i, unsigned long NI, unsigned long NJ, unsigned long NK)
{
	long k = blockIdx.x * blockDim.x + threadIdx.x;
	long j = blockIdx.y * blockDim.y + threadIdx.y;

	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +2;  c21 = +5;  c31 = -8;
	c12 = -3;  c22 = +6;  c32 = -9;
	c13 = +4;  c23 = +7;  c33 = +10;


	if ((i < (NI-1)) && (j < (NJ-1)) &&  (k < (NK-1)) && (i > 0) && (j > 0) && (k > 0))
	{
		B[i*(NK * NJ) + j*NK + k] = c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c21 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c23 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c31 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c33 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c12 * A[(i + 0)*(NK * NJ) + (j - 1)*NK + (k + 0)]  +  c22 * A[(i + 0)*(NK * NJ) + (j + 0)*NK + (k + 0)]   
					     +   c32 * A[(i + 0)*(NK * NJ) + (j + 1)*NK + (k + 0)]  +  c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]  
					     +   c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]  +  c21 * A[(i - 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]  
					     +   c23 * A[(i + 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]  +  c31 * A[(i - 1)*(NK * NJ) + (j + 1)*NK + (k + 1)]  
					     +   c33 * A[(i + 1)*(NK * NJ) + (j + 1)*NK + (k + 1)];
	}
}


void convolution3DCuda(DATA_TYPE* A_gpu, DATA_TYPE* B_gpu, unsigned long NI, unsigned long NJ, unsigned long NK)
{
	double t_start, t_end;
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil( ((float)NK) / ((float)block.x) )), (size_t)(ceil( ((float)NJ) / ((float)block.y) )));
	
	t_start = rtclock();
	long i;
	for (i = 1; i < NI - 1; ++i) // 0
	{
		convolution3D_kernel<<< grid, block >>>(A_gpu, B_gpu, i, NI, NJ, NK);
	}

	cudaDeviceSynchronize();
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
}


int main(int argc, char *argv[])
{
	double t_start, t_end;
	double t_after_array_init;
	long target = 1 << 20; // Default size: 1 MB
	long data_mb = 0;
	if (argc > 1 && argv[1][0] != '-')
		target = atol(argv[1]);

	// See ../../../common/polybenchUtilFuncts.h
	compare_with_cpu = COMPARE_WITH_CPU_DEFAULT;
	copy_back_gpu_results = COPY_BACK_CPU_DEFAULT;

	for (int i = 0; i < argc; i++) {
		GET_BOOL_FLAG(i, "-compare", compare_with_cpu, true);
		GET_BOOL_FLAG(i, "-copy-back", copy_back_gpu_results, true);
		GET_INT_FLAG(i, "-mb", data_mb);
		if (__get_hints(i, argv)) {
			i++;
			continue;
		}
		if (strcmp(argv[i], "-h") == 0)
			return 1;
		UNRECOGNIZED_ARGUMENT(i);
	}

	if (data_mb)
		target = data_mb * 1000000;
	NI = get_data_size(target/2, sizeof(DATA_TYPE), 3);
	NJ = NI;
	NK = NI;

	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;

	A = (DATA_TYPE*)malloc(NI*NJ*NK*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(NI*NJ*NK*sizeof(DATA_TYPE));
	UVM_ALLOC_ARR(DATA_TYPE, A_gpu, NI * NJ * NK);
	UVM_ALLOC_ARR(DATA_TYPE, B_gpu, NI * NJ * NK);

	init(A, A_gpu, NI, NJ, NK);

	GPU_argv_init();
	t_after_array_init = rtclock();

	convolution3DCuda(A_gpu, B_gpu, NI, NJ, NK);

	if (!compare_with_cpu)
		goto skip_comparison;
	t_start = rtclock();
	conv3D(A, B, NI, NJ, NK);
	t_end = rtclock();
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	compareResults(B, B_gpu, NI, NJ, NK);

skip_comparison:
	if (copy_back_gpu_results && compare_with_cpu == 0) {
		TOUCH_ARRAY(B_gpu, sizeof(DATA_TYPE)*NI*NJ*NK);
	}
	t_end = rtclock();
	fprintf(stdout, "CPU + GPU: %lf s\n",
	        t_end - t_after_array_init);

	free(A);
	free(B);
	cudaFree(A_gpu);
	cudaFree(B_gpu);

	return 0;
}

