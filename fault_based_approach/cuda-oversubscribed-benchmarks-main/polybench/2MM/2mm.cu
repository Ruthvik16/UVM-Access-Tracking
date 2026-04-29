/**
 * 2mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#if __has_include("../../../common/polybenchUtilFuncts.h")
// Because we've copied from UVMBench
#include "../../../common/polybenchUtilFuncts.h"
#endif

#if __has_include("../polybenchUtilFuncts.h")
#include "../polybenchUtilFuncts.h"
#endif

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size. */

/* pranjal 
   Operation: C = A * B
              E = C * D
   For square matrices, data size is 5 N^2
 */
unsigned long NI;
unsigned long NJ;
unsigned long NK;
unsigned long NL;

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;


void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* A_gpu, DATA_TYPE* B_gpu, DATA_TYPE* C_gpu, DATA_TYPE* D_gpu, unsigned long NI, unsigned long NJ, unsigned long NK, unsigned long NL)
{
	int i, j;
	double t_start, t_end;
	t_start = rtclock();

	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NK; j++)
		{
                        if (compare_with_cpu)
                                A[i*NI + j] = ((DATA_TYPE) i*j) / NI;
			A_gpu[i*NI + j] = ((DATA_TYPE) i*j) / NI;
		}
	}

	for (i = 0; i < NK; i++)
	{
		for (j = 0; j < NJ; j++)
		{
                        if (compare_with_cpu)
                                B[i*NK + j] = ((DATA_TYPE) i*(j+1)) / NJ;
			B_gpu[i*NK + j] = ((DATA_TYPE) i*(j+1)) / NJ;
		}
	}

	/* C is NOT an input!
	for (i = 0; i < NL; i++)
	{
		for (j = 0; j < NJ; j++)
		{
                        if (compare_with_cpu)
                                C[i*NL + j] = ((DATA_TYPE) i*(j+3)) / NL;
			C_gpu[i*NL + j] = ((DATA_TYPE) i*(j+3)) / NL;
		}
	}
	*/

	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NL; j++)
		{
                        if (compare_with_cpu)
                                D[i*NL + j] = ((DATA_TYPE) i*(j+2)) / NK;
			D_gpu[i*NL + j] = ((DATA_TYPE) i*(j+2)) / NK;
		}
	}
	t_end = rtclock();
	printf("Init A[] B[] C[] D[]: %lf s\n", t_end - t_start);
}


void compareResults(DATA_TYPE *E, DATA_TYPE *E_outputFromGpu, unsigned long NI, unsigned long NJ, unsigned long NK, unsigned long NL)
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < NL; i++)
	{
		for (j=0; j < NI; j++)
		{
			if (percentDiff(E[i*NI + j], E_outputFromGpu[i*NI + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}


__global__ void mm2_kernel1(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, unsigned long NI, unsigned long NJ, unsigned long NK, unsigned long NL)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NJ))
	{ 
		int k;
		for (k = 0; k < NK; k++)
		{
			C[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
		}
	}
}


__global__ void mm2_kernel2(DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *E, unsigned long NI, unsigned long NJ, unsigned long NK, unsigned long NL)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NL))
	{ 
		int k;
		for (k = 0; k < NJ; k++)
		{
			E[i * NL + j] += C[i * NJ + k] * D[k * NL + j];
		}
	}
}


void mm2_cpu(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E, unsigned long NI, unsigned long NJ, unsigned long NK, unsigned long NL)
{
	int i, j, k;
	
  	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			C[i*NJ + j] = 0.0;
			for (k = 0; k < NK; ++k)
			{
				C[i*NJ + j] += A[i*NK + k] * B[k*NJ + j];
			}
		}
	}
	
	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NL; j++)
		{
			E[i*NL + j] = 0.0;
			for (k = 0; k < NJ; ++k)
			{
				E[i*NL + j] += C[i*NJ + k] * D[k*NL + j];
			}
		}
	}
}


void mm2Cuda(DATA_TYPE* A_gpu, DATA_TYPE* B_gpu, DATA_TYPE* C_gpu, DATA_TYPE* D_gpu, DATA_TYPE* E_gpu, unsigned long NI, unsigned long NJ, unsigned long NK, unsigned long NL)
{
	double t_start, t_end;
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)ceil( ((float)NJ) / ((float)block.x) ), (size_t)ceil( ((float)NI) / ((float)block.y)) );
	dim3 grid2((size_t)ceil( ((float)NL) / ((float)block.x) ), (size_t)ceil( ((float)NI) / ((float)block.y)) );
	t_start = rtclock();
	mm2_kernel1<<<grid1,block>>>(A_gpu, B_gpu, C_gpu, NI, NJ, NK, NL);
	cudaDeviceSynchronize();
	mm2_kernel2<<<grid2,block>>>(C_gpu, D_gpu, E_gpu, NI, NJ, NK, NL);
	cudaDeviceSynchronize();
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

	// the one-line output at the end
	fprintf(stdout, __FILE__ " GPU: %lf s ; ", t_end - t_start);
	// In main(), print CPU+GPU right after this
}


int main(int argc, char** argv)
{
	long target = 1 << 20; // Default size: 1 MB
	long data_mb = 0;
	if (argc > 1 && argv[1][0] != '-')
		target = atol(argv[1]);

	// See ../polybenchUtilFuncts.h
	compare_with_cpu = COMPARE_WITH_CPU_DEFAULT;
	copy_back_gpu_results = COPY_BACK_CPU_DEFAULT;

	for (int i = 0; i < argc; i++) {
		GET_BOOL_FLAG(i, "-compare", compare_with_cpu, true);
		GET_BOOL_FLAG(i, "-copy-back", copy_back_gpu_results, true);
		GET_INT_FLAG(i,  "-mb", data_mb);
		if (strncmp(argv[i], "-h", 2) == 0)
			return 1;
	}

	if (data_mb)
		target = data_mb * 1000000;

	NI = get_data_size(target/5, sizeof(DATA_TYPE), 2);
	printf("Matrix dimension: %lu * %lu\n", NI, NI);
	NJ = NI;
	NK = NI;
	NL = NI;

	double t_start, t_end, t_after_array_init;

	DATA_TYPE* C;
	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* D;
	DATA_TYPE* E;

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;
	DATA_TYPE *D_gpu;
	DATA_TYPE *E_gpu;

	C = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
	A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));
	D = (DATA_TYPE*)malloc(NJ*NL*sizeof(DATA_TYPE));
	E = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE));


	UVM_ALLOC_ARR(DATA_TYPE, A_gpu, NI * NK)
	UVM_ALLOC_ARR(DATA_TYPE, B_gpu, NK * NJ)
	UVM_ALLOC_ARR(DATA_TYPE, C_gpu, NI * NJ)
	UVM_ALLOC_ARR(DATA_TYPE, D_gpu, NJ * NL)
	UVM_ALLOC_ARR(DATA_TYPE, E_gpu, NI * NL)

	mickey_clear();
	mickey_register_va(A_gpu + (1 << 21));
	mickey_register_va(B_gpu + (1 << 21));
	mickey_register_va(C_gpu + (1 << 21));
	mickey_register_va(E_gpu + (1 << 21));

	init_array(A, B, C, D, A_gpu, B_gpu, C_gpu, D_gpu, NI, NJ, NK, NL);
	GPU_argv_init();
	t_after_array_init = rtclock();
	mm2Cuda(A_gpu, B_gpu, C_gpu, D_gpu, E_gpu, NI, NJ, NK, NL);

	if (compare_with_cpu) {
		t_start = rtclock();
		mm2_cpu(A, B, C, D, E, NI, NJ, NK, NL);
		t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

		compareResults(E, E_gpu, NI, NJ, NK, NL);
	}


	if (copy_back_gpu_results && compare_with_cpu == 0) {
		// TOUCH GPU pages for D2H migration
		TOUCH_ARRAY(E_gpu, NI * NL * sizeof(DATA_TYPE));
	}
        t_end = rtclock();
	fprintf(stdout, "CPU+GPU: %lf s | Mem %ld MB\n",
	                t_end - t_after_array_init, target/1000000);

	free(C);
	free(A);
	free(B);
	free(D);
	free(E);
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
	cudaFree(D_gpu);
	cudaFree(E_gpu);
	return 0;
}

