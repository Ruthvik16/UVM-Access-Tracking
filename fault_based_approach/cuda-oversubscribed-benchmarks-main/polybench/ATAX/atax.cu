/**
 * atax.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "../cuda-macros-v1.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

/* Problem size. */
#define NX 4096
#define NY 4096

#undef NX
#undef NY
unsigned long NX;
unsigned long NY;

/* Thread block dimensions */
unsigned long DIM_THREAD_BLOCK_X = 256;
unsigned long DIM_THREAD_BLOCK_Y = 1;

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void init_array(DATA_TYPE *x, DATA_TYPE *A, DATA_TYPE *x_gpu, DATA_TYPE *A_gpu, unsigned long NX, unsigned long NY)
{
	int i, j;
	double init_start, init_end;
	init_start = rtclock();

	for (i = 0; i < NX; i++)
	{
		x[i] = i * M_PI;
		x_gpu[i] = i * M_PI;
		for (j = 0; j < NY; j++)
		{
			if (compare_with_cpu)
				A[i*NY + j] = ((DATA_TYPE) i*(j)) / NX;
			A_gpu[i*NY + j] = ((DATA_TYPE) i*(j)) / NX;
		}
	}

	init_end = rtclock();
	printf("Array Dimensions %lu * %lu\n", NX, NY);
	printf("Init A[]: %.2lf s\n", init_end - init_start);
}


void compareResults(DATA_TYPE *z, DATA_TYPE *z_outputFromGpu, unsigned long NX, unsigned long NY)
{
	int i, fail;
	fail = 0;

	for (i=0; i<NY; i++)
	{
		if (percentDiff(z[i], z_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
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


__global__ void atax_kernel1(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp, unsigned long NX, unsigned long NY)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < NX)
	{
		int j;
		for(j=0; j < NY; j++)
		{
			tmp[i] += A[i * NY + j] * x[j];
		}
	}
}

__global__ void atax_kernel2(DATA_TYPE *A, DATA_TYPE *y, DATA_TYPE *tmp, unsigned long NX, unsigned long NY)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < NY)
	{
		int i;
		for(i=0; i < NX; i++)
		{
			y[j] += A[i * NY + j] * tmp[i];
		}
	}
}


void atax_cpu(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp, unsigned long NX, unsigned long NY)
{
	int i,j;
	
	for (i= 0; i < NY; i++)
	{
    	y[i] = 0;
	}
  
	for (i = 0; i < NX; i++)
 	{
      	tmp[i] = 0;

      	for (j = 0; j < NY; j++)
		{
			tmp[i] = tmp[i] + A[i*NY + j] * x[j];
		}
		
      	for (j = 0; j < NY; j++)
		{
			y[j] = y[j] + A[i*NY + j] * tmp[i];
		}
    }
}


void ataxGpu(DATA_TYPE* A_gpu, DATA_TYPE* x_gpu, DATA_TYPE* y_gpu, DATA_TYPE* tmp_gpu, unsigned long NX, unsigned long NY)
{
	double t_start, t_end, t_step;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NX) / ((float)block.x) )), 1);
	dim3 grid2((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);

	t_start = rtclock();
	atax_kernel1<<< grid1, block >>>(A_gpu,x_gpu,tmp_gpu, NX, NY);
	cudaDeviceSynchronize();
	t_step = rtclock();
	printf("# %s: kernel 1: %.3lfs\n", __FILE__, t_step - t_start);
	t_step = rtclock();
	atax_kernel2<<< grid2, block >>>(A_gpu,y_gpu,tmp_gpu, NX, NY);
	cudaDeviceSynchronize();
	t_end = rtclock();
	printf("# %s: kernel 2: %.3lfs\n", __FILE__, t_end - t_step);
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

	// the one-line output at the end
	fprintf(stdout, __FILE__ " GPU: %lf s ; ", t_end - t_start);
	// In main(), print CPU+GPU right after this

}


int main(int argc, char** argv)
{
	double t_start, t_end;
	double t_after_array_init;
	long target = 1 << 25, data_mb = 0;
	if (argc > 1 && argv[1][0] != '-')
		target = atol(argv[1]);

	compare_with_cpu = COMPARE_WITH_CPU_DEFAULT;
	copy_back_gpu_results = COPY_BACK_CPU_DEFAULT;

	for (int i = 0; i < argc; i++) {
		GET_BOOL_FLAG(i, "-compare", compare_with_cpu, true);
		GET_BOOL_FLAG(i, "-copy-back", copy_back_gpu_results, true);
		GET_INT_FLAG(i, "-thread-block", DIM_THREAD_BLOCK_X);
		get_hints(i, argv);
		GET_INT_FLAG(i, "-mb", data_mb);
		if (strncmp(argv[i], "-h", 2) == 0)
			return 1;
		UNRECOGNIZED_ARGUMENT(i);
	}

	if (data_mb)
		target = data_mb * 1000000;
	NX = get_data_size(target, sizeof(DATA_TYPE), 2);
	NY = NX;

	DATA_TYPE* A;
	DATA_TYPE* x;
	DATA_TYPE* y;
	DATA_TYPE* tmp;

	DATA_TYPE *A_gpu;
	DATA_TYPE *x_gpu;
	DATA_TYPE *y_gpu;
	DATA_TYPE *tmp_gpu;

	// DATA_TYPE* tmp;
	A = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
	x = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	y = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	tmp = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));

	UVM_ALLOC_ARR(DATA_TYPE, A_gpu, NX * NY);
	UVM_ALLOC_ARR(DATA_TYPE, x_gpu, NY);
	UVM_ALLOC_ARR(DATA_TYPE, y_gpu, NY);
	UVM_ALLOC_ARR(DATA_TYPE, tmp_gpu, NX);

	mickey_register_va((char *)A_gpu);
	mickey_register_va(((char *)A_gpu) + (1 << 31));

	init_array(x, A, x_gpu, A_gpu, NX, NY);

#ifdef CUDA_CLI_HINTS
	if (cuda_hints[1][HINT_PREFETCH_ASYNC])
		CUDA_PF_ASYNC_GPU(A_gpu, NX * NY * sizeof(DATA_TYPE));
	if (cuda_hints[2][HINT_PREFETCH_ASYNC])
		CUDA_PF_ASYNC_GPU(x_gpu, NY * sizeof(DATA_TYPE));
#endif

	GPU_argv_init();
	t_after_array_init = rtclock();
	ataxGpu(A_gpu, x_gpu, y_gpu, tmp_gpu, NX, NY);
	
	if (compare_with_cpu) {
		t_start = rtclock();
		atax_cpu(A, x, y, tmp, NX, NY);
		t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.3lfs\n", t_end - t_start);
	}

	compareResults(y, y_gpu, NX, NY);

	if (copy_back_gpu_results && compare_with_cpu == 0) {
		TOUCH_ARRAY(y_gpu, sizeof(DATA_TYPE)*NY);
	}
	t_end = rtclock();
	printf("CPU+GPU: %.3lf s | Mem %ld MB\n",
		        t_end - t_after_array_init, target/1000000);

	free(A);
	free(x);
	free(y);
	free(tmp);
	
	cudaFree(A_gpu);
	cudaFree(x_gpu);
	cudaFree(y_gpu);
	cudaFree(tmp_gpu);

  	return 0;
}

