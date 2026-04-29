/**
 * bicg.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#include <sys/time.h>
#include <cuda.h>

#if __has_include("../../../common/polybenchUtilFuncts.h")
// Because we've copied from UVMBench
#include "../../../common/polybenchUtilFuncts.h"
#endif

#if __has_include("../polybenchUtilFuncts.h")
#include "../polybenchUtilFuncts.h"
#endif

//Error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

/*
 * Inputs: A, P, R. Read-only.
 * Outputs: Q, S.
 */
long NX, NY;

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void init_array(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *r, DATA_TYPE *A_gpu, DATA_TYPE *p_gpu, DATA_TYPE *r_gpu)
{
	int i, j;
	double t_start, t_end;
	fprintf(stdout, "Array Dimensions: %ld * %ld\n", NX, NY);
	t_start = rtclock();

	for (i = 0; i < NX; i++)
	{
		if (compare_with_cpu)
			r[i] = i * M_PI;
		r_gpu[i] = i * M_PI;

		for (j = 0; j < NY; j++)
		{
			if (compare_with_cpu)
				A[i*NY + j] = ((DATA_TYPE) i*j) / NX;
			A_gpu[i*NY + j] = ((DATA_TYPE) i*j) / NX;
		}
	}

	for (i = 0; i < NY; i++)
	{
		if (compare_with_cpu)
			p[i] = i * M_PI;
		p_gpu[i] = i * M_PI;
	}
	t_end = rtclock();
	fprintf(stdout, "Init P[] R[] A[]: %.2lf s\n", t_end - t_start);
}


void compareResults(DATA_TYPE* s, DATA_TYPE* s_outputFromGpu, DATA_TYPE* q, DATA_TYPE* q_outputFromGpu)
{
	int i,fail;
	fail = 0;

	// Compare s with s_cuda
	for (i=0; i<NX; i++)
	{
		if (percentDiff(q[i], q_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}
	}

	for (i=0; i<NY; i++)
	{
		if (percentDiff(s[i], s_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
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


//Distributed (split) from initial loop and permuted into reverse order to allow parallelism...
__global__ void bicg_kernel1(DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s, long NX, long NY)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < NY)
	{
		s[j] = 0.0f;

		int i;
		for(i = 0; i < NX; i++)
		{
			s[j] += A[i * NY + j] * r[i];
		}
	}
}


//Distributed (split) from initial loop to allow parallelism
__global__ void bicg_kernel2(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *q, long NX, long NY)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < NX)
	{
		q[i] = 0.0f;

		int j;
		for(j=0; j < NY; j++)
		{
			q[i] += A[i * NY + j] * p[j];
		}
	}
}


void bicg_cpu(DATA_TYPE* A, DATA_TYPE* r, DATA_TYPE* s, DATA_TYPE* p, DATA_TYPE* q)
{
	int i,j;

	for (i = 0; i < NY; i++)
	{
		s[i] = 0.0;
	}

	for (i = 0; i < NX; i++)
	{
		q[i] = 0.0;
		for (j = 0; j < NY; j++)
		{
			s[j] = s[j] + r[i] * A[i*NY + j];
			q[i] = q[i] + A[i*NY + j] * p[j];
		}
	}
}


void bicgCuda(DATA_TYPE* A_gpu, DATA_TYPE* r_gpu, DATA_TYPE* s_gpu, DATA_TYPE* p_gpu, DATA_TYPE* q_gpu)
{
	double t_start, t_end;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);
	dim3 grid2((size_t)(ceil( ((float)NX) / ((float)block.x) )), 1);

	t_start = rtclock();
	bicg_kernel1<<< grid1, block >>>(A_gpu, r_gpu, s_gpu, NX, NY);
	CHECK_RETURN_VALUE(cudaDeviceSynchronize());
	bicg_kernel2<<< grid2, block >>>(A_gpu, p_gpu, q_gpu, NX, NY);
	CHECK_RETURN_VALUE(cudaDeviceSynchronize());
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
}

int main(int argc, char** argv)
{
	double t_start, t_end;
	double t_after_array_init;
	long target = 1 << 28;
	long data_mb = 0;
	if (argc > 1 && argv[1][0] != '-')
		target = atol(argv[1]);

	compare_with_cpu = COMPARE_WITH_CPU_DEFAULT;
	copy_back_gpu_results = COPY_BACK_CPU_DEFAULT;

	for (int i = 0; i < argc; i++) {
		GET_BOOL_FLAG(i, "-compare", compare_with_cpu, true);
		GET_BOOL_FLAG(i, "-copy-back", copy_back_gpu_results, true);
		GET_INT_FLAG(i, "-mb", data_mb);
		if (__get_hints(i, argv))
			continue;
		if (strncmp(argv[i], "-h", 2) == 0)
			return 1;
		UNRECOGNIZED_ARGUMENT(i);
	}

	if (data_mb)
		target = data_mb * 1000000;

	NX = get_data_size(target, sizeof(DATA_TYPE), 2);
	NY = NX;

	DATA_TYPE* A;
	DATA_TYPE* r;
	DATA_TYPE* s;
	DATA_TYPE* p;
	DATA_TYPE* q;

	DATA_TYPE *A_gpu;
	DATA_TYPE *q_gpu;
	DATA_TYPE *p_gpu;
	DATA_TYPE *r_gpu;
	DATA_TYPE *s_gpu;

	A = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
	r = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));
	s = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	p = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	q = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));

	UVM_ALLOC_BUF(DATA_TYPE, A_gpu, sizeof(DATA_TYPE) * NX * NY);
	UVM_ALLOC_BUF(DATA_TYPE, r_gpu, sizeof(DATA_TYPE) * NX);
	UVM_ALLOC_BUF(DATA_TYPE, s_gpu, sizeof(DATA_TYPE) * NY);
	UVM_ALLOC_BUF(DATA_TYPE, p_gpu, sizeof(DATA_TYPE) * NY);
	UVM_ALLOC_BUF(DATA_TYPE, q_gpu, sizeof(DATA_TYPE) * NX);

	init_array(A, p, r, A_gpu, p_gpu, r_gpu);

	GPU_argv_init();
	t_after_array_init = rtclock();

	bicgCuda(A_gpu, r_gpu, s_gpu, p_gpu, q_gpu);

	if (compare_with_cpu) {
		t_start = rtclock();
		bicg_cpu(A, r, s, p, q);
		t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

		compareResults(s, s_gpu, q, q_gpu);
	}

	if (copy_back_gpu_results && compare_with_cpu == 0) {
		TOUCH_ARRAY(s_gpu, NY*sizeof(DATA_TYPE));
		TOUCH_ARRAY(q_gpu, NX*sizeof(DATA_TYPE));
	}
	t_end = rtclock();
	printf("CPU+GPU: %.3lf s | Mem %ld MB\n",
			t_end - t_after_array_init, target/1000000);

	free(A);
	free(r);
	free(s);
	free(p);
	free(q);
	cudaFree(A_gpu);
	cudaFree(r_gpu);
	cudaFree(s_gpu);
	cudaFree(p_gpu);
	cudaFree(q_gpu);
	return 0;
}
