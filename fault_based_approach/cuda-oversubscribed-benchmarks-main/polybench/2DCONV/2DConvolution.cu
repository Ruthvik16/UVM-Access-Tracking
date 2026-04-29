/**
 * 2DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#include <assert.h>

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

/* Problem size */
unsigned long NI;
unsigned long NJ;
unsigned init_with_random = 0;

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void conv2D(DATA_TYPE* A, DATA_TYPE* B, unsigned long NI, unsigned long NJ)
{
	int i, j;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;
	cudaMemAdvise(A, NI*NJ, cudaMemAdviseSetReadMostly, GPU_DEVICE);

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;


	for (i = 1; i < NI - 1; ++i) // 0
	{
		for (j = 1; j < NJ - 1; ++j) // 1
		{
			B[i*NJ + j] = c11 * A[(i - 1)*NJ + (j - 1)]  +  c12 * A[(i + 0)*NJ + (j - 1)]  +  c13 * A[(i + 1)*NJ + (j - 1)]
				+ c21 * A[(i - 1)*NJ + (j + 0)]  +  c22 * A[(i + 0)*NJ + (j + 0)]  +  c23 * A[(i + 1)*NJ + (j + 0)] 
				+ c31 * A[(i - 1)*NJ + (j + 1)]  +  c32 * A[(i + 0)*NJ + (j + 1)]  +  c33 * A[(i + 1)*NJ + (j + 1)];
		}
	}
}



void init(DATA_TYPE* A, DATA_TYPE* A_gpu, unsigned long NI, unsigned long NJ)
{
	int i, j;
	double t_start, t_end;

	t_start = rtclock();
	for (i = 0; i < NI; ++i)
	{
		for (j = 0; j < NJ; ++j)
		{
			float temp =  (init_with_random ? (float)rand()/RAND_MAX : (((i+2) * (j+2)) * (3 + i + j)));
			if (compare_with_cpu)
				A[i*NJ + j] = temp;
			A_gpu[i*NJ + j] = temp;
        	}
    	}
	t_end = rtclock();
	printf("init A[]: %lf s\n", ((t_end - t_start)));
}


void compareResults(DATA_TYPE* B, DATA_TYPE* B_outputFromGpu, unsigned long NI, unsigned long NJ)
{
	long i, j, fail = 0, correct = 0;
	
	// Compare a and b
	for (i=1; i < (NI-1); i++) 
	{
		for (j=1; j < (NJ-1); j++) 
		{
			if (percentDiff(B[i*NJ + j], B_outputFromGpu[i*NJ + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
			else
				correct++;
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %ld\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
	printf("Matching CPU-GPU outputs: %ld\n", correct);
	
}

__global__ void Convolution2D_kernel(DATA_TYPE *A, DATA_TYPE *B, unsigned long NI, unsigned long NJ)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

	if ((i < NI-1) && (j < NJ-1) && (i > 0) && (j > 0))
	{
		B[i * NJ + j] =  c11 * A[(i - 1) * NJ + (j - 1)]  + c21 * A[(i - 1) * NJ + (j + 0)] + c31 * A[(i - 1) * NJ + (j + 1)] 
			+ c12 * A[(i + 0) * NJ + (j - 1)]  + c22 * A[(i + 0) * NJ + (j + 0)] +  c32 * A[(i + 0) * NJ + (j + 1)]
			+ c13 * A[(i + 1) * NJ + (j - 1)]  + c23 * A[(i + 1) * NJ + (j + 0)] +  c33 * A[(i + 1) * NJ + (j + 1)];
	}
}


void convolution2DCuda(DATA_TYPE* A_gpu, DATA_TYPE* B_gpu, unsigned long NI, unsigned long NJ)
{
	double t_start, t_end;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)ceil( ((float)NI) / ((float)block.x) ), (size_t)ceil( ((float)NJ) / ((float)block.y)) );
	t_start = rtclock();

	#ifdef PREF
	cudaStream_t stream1;
	cudaStream_t stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	for (int i = 0; i < 1; i++)
	{
		cudaMemPrefetchAsync(A_gpu,NI*NJ*sizeof(DATA_TYPE), GPU_DEVICE, stream1 );
		cudaStreamSynchronize(stream1);
		cudaMemPrefetchAsync(B_gpu,NI*NJ*sizeof(DATA_TYPE), GPU_DEVICE, stream2 );
		cudaStreamSynchronize(stream2);
		// cudaMemset(B_gpu,0 ,NI*NJ*sizeof(DATA_TYPE));
		Convolution2D_kernel<<<grid,block, 0,stream2>>>(A_gpu,B_gpu, NI, NJ);
		cudaDeviceSynchronize();
	}
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);//);
	#else
		for (int i = 0; i < 1; i++)
		{
			Convolution2D_kernel<<<grid,block>>>(A_gpu,B_gpu, NI, NJ);
			cudaError_t ret = cudaDeviceSynchronize();
			assert(ret == cudaSuccess);
		}
		t_end = rtclock();
		fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);//);
	#endif


	// the one-line output at the end
	fprintf(stdout, __FILE__ " GPU: %lf s ; ", t_end - t_start);
	// In main(), print CPU+GPU right after this
}

// Does A->B and B->A on loop
void convolution2DCuda_phased(DATA_TYPE* A_gpu, DATA_TYPE* B_gpu, unsigned long NI, unsigned long NJ,
	int num_phases, bool cuda_readonly_hint)
{
	double t_start, t_end, t_phase_start;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)ceil( ((float)NI) / ((float)block.x) ), (size_t)ceil( ((float)NJ) / ((float)block.y)) );
	printf("%s: Running with %d phases\n", __FILE__, num_phases);
	t_start = rtclock();

	for (int i = 0; i < num_phases; i++)
	{
		if (cuda_readonly_hint) {
			cudaError_t ret;
			ret = cudaMemAdvise(A_gpu, NI*NJ*sizeof(DATA_TYPE),
				((i%2) ? cudaMemAdviseUnsetReadMostly : cudaMemAdviseSetReadMostly),
				0);
			assert(ret == cudaSuccess);
			ret = cudaMemAdvise(B_gpu, NI*NJ*sizeof(DATA_TYPE),
				((i%2 == 0) ? cudaMemAdviseUnsetReadMostly : cudaMemAdviseSetReadMostly),
				0);
			assert(ret == cudaSuccess);
		}

		t_phase_start = rtclock();
		if (i % 2 == 0)
			Convolution2D_kernel<<<grid, block>>>(A_gpu, B_gpu, NI, NJ);
		else
			Convolution2D_kernel<<<grid, block>>>(B_gpu, A_gpu, NI, NJ);
		cudaError_t ret = cudaDeviceSynchronize();
		assert(ret == cudaSuccess);
		printf("Phase %d run time: %0.6lfs\n", i, rtclock() - t_phase_start);
	}
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
}



int main(int argc, char *argv[])
{
	double t_end, t_after_array_init;
	long target = 1 << 28;
	int num_phases = 1;
	bool pass_cuda_readonly_hints = false;
	bool accessed_by_gpu_hint = false;
	bool pin_cpu_hint = false;
	bool prefetch_h2d_hint = false;
	long data_mb = 0;

	if (argc > 1 && argv[1][0] != '-')
		target = atol(argv[1]);

	compare_with_cpu = COMPARE_WITH_CPU_DEFAULT;
	copy_back_gpu_results = COPY_BACK_CPU_DEFAULT;

	for (int i = 0; i < argc; i++) {
		int iteration_counter = i; // used in CHECK_ARG_AND_SET_PARAM
		CHECK_ARG_AND_SET_VAL("-compare", compare_with_cpu, true);
		CHECK_ARG_AND_SET_VAL("-copy-back", copy_back_gpu_results, true);
		CHECK_ARG_AND_SET_VAL("-no-random", init_with_random, 0);
		GET_INT_FLAG(i, "-phases",  num_phases);
		CHECK_ARG_AND_SET_VAL("-read-mostly-hint", pass_cuda_readonly_hints, true);
		GET_BOOL_FLAG(i, "-accessed-by-gpu", accessed_by_gpu_hint, true);
		GET_BOOL_FLAG(i, "-pin-cpu", pin_cpu_hint, true);
		GET_BOOL_FLAG(i, "-prefetch-h2d", prefetch_h2d_hint, true);
		GET_BOOL_FLAG(i, "-random-data", init_with_random, 1);
		GET_INT_FLAG(i, "-mb", data_mb);
		get_hints(i, argv);

		if (strncmp(argv[i], "-h", 2) == 0)
			return 1;
		UNRECOGNIZED_ARGUMENT(i);
	}
	assert(num_phases > 0);

	DATA_TYPE* A;
	DATA_TYPE* B;  
	if (data_mb)  {
		target = data_mb * 1000000;
	}
	NI = get_data_size(target/2, sizeof(DATA_TYPE), 2);
	NJ = NI;
	printf("Memory footprint: %ld | NI = %ld | NJ = %ld\n", target, NI, NJ);


	long ar_size = NI * NJ * sizeof(DATA_TYPE);
	DATA_TYPE *A_gpu, *B_gpu;
	A = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
	UVM_ALLOC_BUF(DATA_TYPE, A_gpu, NI * NJ * sizeof(DATA_TYPE));
	UVM_ALLOC_BUF(DATA_TYPE, B_gpu, NI * NJ * sizeof(DATA_TYPE));
	assert(A_gpu);
	assert(B_gpu);
	mickey_clear();
	mickey_register_va(((char *)A_gpu) + (1 << 22));
	mickey_register_va(((char *)A_gpu) + (1 << 30));
	mickey_register_va(((char *)B_gpu) + (1 << 30));
	mickey_register_va(((char *)B_gpu) + (1UL << 33));

	//initialize the arrays
	init(A, A_gpu, NI, NJ);

	if (pass_cuda_readonly_hints) {
		CUDA_READ_MOSTLY_HINT(A_gpu, ar_size);
	}

	if (accessed_by_gpu_hint) {
		CUDA_ACCESSED_BY_GPU_HINT(A_gpu, ar_size);
		CUDA_ACCESSED_BY_GPU_HINT(B_gpu, ar_size);
	}

	if (pin_cpu_hint) {
		CUDA_PIN_CPU_HINT(A_gpu, ar_size);
		CUDA_PIN_CPU_HINT(B_gpu, ar_size);
	}

#ifdef CUDA_CLI_HINTS
	if (prefetch_h2d_hint) {
		CUDA_PF_ASYNC_GPU(A_gpu, ar_size);
		if (num_phases != 1)
			fprintf(stderr, "Warning - pf_async with phases\n");
	}
	HINTS_POST_INIT(A_gpu, ar_size, 1);
	HINTS_POST_INIT(B_gpu, ar_size, 2);
#endif

	t_after_array_init = rtclock();

	if (num_phases == 1)
		convolution2DCuda(A_gpu, B_gpu, NI, NJ);
	else
		convolution2DCuda_phased(A_gpu, B_gpu, NI, NJ, num_phases, pass_cuda_readonly_hints);

#ifdef CUDA_CLI_HINTS
	HINTS_POST_COMPUTE(A_gpu, ar_size, 1);
	HINTS_POST_COMPUTE(B_gpu, ar_size, 2);
#endif
	if (compare_with_cpu) {
		conv2D(A, B, NI, NJ);
		compareResults(B, B_gpu, NI, NJ);
	}

	if (copy_back_gpu_results && compare_with_cpu == 0) {
		TOUCH_ARRAY(B_gpu, NI*NJ*sizeof(DATA_TYPE));
	}
	t_end = rtclock();

	fprintf(stdout, "2DC: CPU+GPU: %lf s | Mem %ld MiB\n\n",
		        t_end - t_after_array_init, target >> 20);
	free(A);
	free(B);
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	
	return 0;
}
