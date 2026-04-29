/**
 * mvt.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

/* Problem size */
#define N 4096

// Memory footprint is N*N*sizeof(float)
#undef N
long N;

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1
unsigned dim_thread_block_x = DIM_THREAD_BLOCK_X;
unsigned per_kernel_array = 0;
bool do_optimal_10gb = false;
bool optimal_tb_16_1024 = false;
bool accby_gpu_after_kernel1 = false;
unsigned long prefetchAsync_mb = 0;

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;


__global__ void gpu_init_matrix(DATA_TYPE *a, long N)
{
	long i = (long)blockIdx.y * blockDim.y + threadIdx.y;
	long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N && j < N)
		a[i * N + j] = ((DATA_TYPE)(i * j)) / N;
}

// @a_gpu_k1 and @a_gpu_k2 might be the same pointer.
void init_array(
	DATA_TYPE *A,
	DATA_TYPE *x1,
	DATA_TYPE *x2,
	DATA_TYPE *y1,
	DATA_TYPE *y2,
	DATA_TYPE *a_gpu_k1,
	DATA_TYPE *a_gpu_k2,
	DATA_TYPE *x1_gpu,
	DATA_TYPE *x2_gpu,
	DATA_TYPE *y_1_gpu,
	DATA_TYPE *y_2_gpu)
{
	int i;
	double t_start, t_end;

	fprintf(stdout, "Array Dimensions: %ld * %ld\n", N, N);
	t_start = rtclock();

	/* Initialize small vectors on CPU (negligible memory). */
	for (i = 0; i < N; i++)
	{
		if (compare_with_cpu) {
			x1[i] = ((DATA_TYPE) i) / N;
			x2[i] = ((DATA_TYPE) i + 1) / N;
			y1[i] = ((DATA_TYPE) i + 3) / N;
			y2[i] = ((DATA_TYPE) i + 4) / N;
		}
		x1_gpu[i] = ((DATA_TYPE) i) / N;
		x2_gpu[i] = ((DATA_TYPE) i + 1) / N;
		y_1_gpu[i] = ((DATA_TYPE) i + 3) / N;
		y_2_gpu[i] = ((DATA_TYPE) i + 4) / N;
	}

	/* Initialize matrix on GPU to avoid faulting N*N pages into CPU RAM. */
	dim3 block(16, 16);
	dim3 grid((N + 15) / 16, (N + 15) / 16);
	gpu_init_matrix<<<grid, block>>>(a_gpu_k1, N);
	if (a_gpu_k1 != a_gpu_k2)
		gpu_init_matrix<<<grid, block>>>(a_gpu_k2, N);
	TRY_DEVICE_SYNCHRONIZE();

	t_end = rtclock();
	fprintf(stdout, "init x1[] x2[] y1[] y2[] A[]: %lf s\n", t_end - t_start);
}



void runMvt(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y1, DATA_TYPE* y2)
{
	int i, j;
	
	for (i=0; i<N; i++) 
	{
		for (j=0; j<N; j++) 
		{
       			x1[i] = x1[i] + a[i*N + j] * y1[j];
        	}
    	}
	
	for (i=0; i<N; i++) 
	{
		for (j=0; j<N; j++) 
		{
 		       	x2[i] = x2[i] + a[j*N + i] * y2[j];
      		}
    	}
}


void compareResults(DATA_TYPE* x1, DATA_TYPE* x1_outputFromGpu, DATA_TYPE* x2, DATA_TYPE* x2_outputFromGpu)
{
	int i, fail;
	fail = 0;
	
	for (i=0; i<N; i++) 
	{
		if (percentDiff(x1[i], x1_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}

		if (percentDiff(x2[i], x2_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


/*
 * Each thread takes a row in A. There is a LOT of locality within a thread, but
 * adjacent threads in a warp pick addresses separated by a row.
 * This is the "slow" kernel, which suffers under oversubscription.
 */
__global__ void mvt_kernel1(DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *y_1, long N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		int j;
		for(j=0; j < N; j++)
		{
			x1[i] += a[i * N + j] * y_1[j];
		}
	}
}


/*
 * Thread i takes column i of the matrix. There is near-zero locality within a
 * thread's accesses, tremendous locality in *one* access from a warp, and if the
 * warps run in parallel, a lot of locality overall.
 */
__global__ void mvt_kernel2(DATA_TYPE *a, DATA_TYPE *x2, DATA_TYPE *y_2, long N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		int j;
		for(j=0; j < N; j++)
		{
			x2[i] += a[j * N + i] * y_2[j];	
		}
	}
}

void mvtCuda(DATA_TYPE *a_gpu_k1,
             DATA_TYPE *a_gpu_k2,
             DATA_TYPE *x1_gpu,
             DATA_TYPE *x2_gpu,
             DATA_TYPE *y_1_gpu,
             DATA_TYPE *y_2_gpu
             )
{
	double t_start, t_end, t_total;
	dim3 block(dim_thread_block_x, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)ceil((float)N/ ((float)dim_thread_block_x)), 1);
	t_total = 0;

	t_start = rtclock();
	if (optimal_tb_16_1024) {
		long tb_size = 16;
		printf("# kernel1: %ld threads\n", tb_size);
		long num_tbs = (N + tb_size - 1)/tb_size;
		mvt_kernel1<<<num_tbs, tb_size>>>(a_gpu_k1, x1_gpu, y_1_gpu, N);
	} else {
		mvt_kernel1<<<grid,block>>>(a_gpu_k1, x1_gpu, y_1_gpu, N);
	}

	TRY_DEVICE_SYNCHRONIZE();
	t_end = rtclock();
	printf("# MVT Kernel 1: %0.6lf s\n", t_end - t_start);
	t_total += t_end - t_start;


	if (accby_gpu_after_kernel1)
		CUDA_ACCESSED_BY_GPU_HINT(a_gpu_k2, (N * N * sizeof(DATA_TYPE)));
	t_start = rtclock();

	if (optimal_tb_16_1024) {
		long tb_size = 1024;
		long num_tbs;
		printf("# kernel2: 1024 threads\n");
		num_tbs = (N + tb_size - 1)/tb_size;
		mvt_kernel2<<<num_tbs, tb_size>>>(a_gpu_k2, x2_gpu, y_2_gpu, N);
	} else {
		mvt_kernel2<<<grid,block>>>(a_gpu_k2, x2_gpu, y_2_gpu, N);
	}

	TRY_DEVICE_SYNCHRONIZE();
	t_end = rtclock();
	printf("# MVT Kernel 2: %0.6lf s\n", t_end - t_start);
	t_total += t_end - t_start;

	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_total);

	// the one-line output at the end
	fprintf(stdout, __FILE__ " GPU: %lf s ; ", t_total);
	// In main(), print CPU+GPU right after this
}


int main(int argc, char *argv[])
{
	double t_start, t_end;
	double t_after_array_init;
	bool memAdviseReadMostly_hint = false;
	bool prefetchAsyncHint = false;
	bool pin_input_cpu_hint = false;
	bool accessed_by_gpu_hint = false;
	bool pin_input_cpu_before_init_hint = false;
	bool pin_vectors_gpu = false, pin_vectors_cpu = false;
	unsigned long data_mb = 0;

	long target = 1 << 25;
	if (argc > 1 && argv[1][0] != '-')
		target = atol(argv[1]);

	compare_with_cpu = COMPARE_WITH_CPU_DEFAULT;
	copy_back_gpu_results = COPY_BACK_CPU_DEFAULT;
	for (int i = 0; i < argc; i++) {
		GET_BOOL_FLAG(i, "-compare", compare_with_cpu, true);
		GET_BOOL_FLAG(i, "-copy-back", copy_back_gpu_results, true);
		GET_BOOL_FLAG(i, "-skip-copy-back", copy_back_gpu_results, false);
		GET_BOOL_FLAG(i, "-read-mostly-hint", memAdviseReadMostly_hint, true);
		GET_BOOL_FLAG(i, "-prefetch-h2d", prefetchAsyncHint, true);
		GET_BOOL_FLAG(i, "-pin-input-cpu", pin_input_cpu_hint, true);
		GET_BOOL_FLAG(i, "-accessed-by-gpu", accessed_by_gpu_hint, true);
		GET_BOOL_FLAG(i, "-pin-cpu-before-init", pin_input_cpu_before_init_hint, true);
		GET_BOOL_FLAG(i, "-per-kernel-array", per_kernel_array, true);
		GET_BOOL_FLAG(i, "-pin-vectors-cpu", pin_vectors_cpu, true);
		GET_BOOL_FLAG(i, "-pin-vectors-gpu", pin_vectors_gpu, true);
		GET_INT_FLAG(i,  "-thread-block", dim_thread_block_x);
		GET_INT_FLAG(i, "-mb", data_mb);
		GET_BOOL_FLAG(i, "-optimal", do_optimal_10gb, true);
		GET_BOOL_FLAG(i, "-optimal-tblock", optimal_tb_16_1024, true);
		GET_BOOL_FLAG(i, "-kernel2-accby", accby_gpu_after_kernel1, true);
		GET_INT_FLAG(i, "-pfasync-mb", prefetchAsync_mb);
		get_hints(i, argv);
		if (strncmp(argv[i], "-h", 2) == 0)
			return 1;
		UNRECOGNIZED_ARGUMENT(i);
	}

	if (data_mb) {
		target = data_mb * 1000000;
	}
	N = get_data_size(target, sizeof(DATA_TYPE), 2);

	if (do_optimal_10gb) {
		accby_gpu_after_kernel1 = true;
		optimal_tb_16_1024      = true;
	}

	DATA_TYPE* a;
	DATA_TYPE* x1;
	DATA_TYPE* x2;
	DATA_TYPE* y_1;
	DATA_TYPE* y_2;
	DATA_TYPE* a_gpu;
	DATA_TYPE* a_gpu_k1 = NULL;
	DATA_TYPE* a_gpu_k2 = NULL;
	DATA_TYPE* x1_gpu;
	DATA_TYPE* x2_gpu;
	DATA_TYPE* y_1_gpu;
	DATA_TYPE* y_2_gpu;
	a = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));
	x1 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	x2 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	y_1 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	y_2 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	long a_size = N * N * sizeof(DATA_TYPE);
	UVM_ALLOC_BUF(DATA_TYPE, a_gpu, a_size);
	UVM_ALLOC_ARR(DATA_TYPE, x1_gpu, N);
	UVM_ALLOC_ARR(DATA_TYPE, x2_gpu, N);
	UVM_ALLOC_ARR(DATA_TYPE, y_1_gpu, N);
	UVM_ALLOC_ARR(DATA_TYPE, y_2_gpu, N);

	if (pin_input_cpu_before_init_hint) {
		CUDA_PIN_CPU_HINT(a_gpu, a_size);
	}
	mickey_clear();
	mickey_register_va(a_gpu);
	mickey_register_va(a_gpu + (1 << 29));

	a_gpu_k1 = a_gpu;
	if (per_kernel_array == 0) {
		a_gpu_k2 = a_gpu;
	} else {
		printf("# Using separate buffers for the two kernels\n");
		// The second allocation is slightly larger for easy analysis. We won't
		// be using the extra 2 MiB.
#ifdef SQUIDWARD_ENABLED
		squidward_report_buf_name(a_gpu_k1, a_size, "a_gpu_k1", SQUIDWARD_DEFAULT);
#endif
		UVM_ALLOC_BUF(DATA_TYPE, a_gpu_k2, a_size + (1 << 22));
		mickey_register_va(a_gpu_k2 + (1 << 29));
	}

	init_array(a, x1, x2, y_1, y_2,
	           a_gpu_k1, a_gpu_k2, x1_gpu, x2_gpu, y_1_gpu, y_2_gpu);

	if (prefetchAsync_mb) {
		CUDA_PF_ASYNC_GPU(a_gpu_k1, prefetchAsync_mb << 20);
	}

	t_after_array_init = rtclock();
	if (memAdviseReadMostly_hint) {
		CUDA_READ_MOSTLY_HINT(a_gpu_k1, a_size);
		if (per_kernel_array)
			CUDA_READ_MOSTLY_HINT(a_gpu_k2, a_size);
	}

	if (prefetchAsyncHint) {
		if (per_kernel_array)
			fprintf(stderr, "WARNING: per_kernel_array incompatible with prefetch hint\n");
		CHECK_RETURN_VALUE(cudaMemPrefetchAsync(a_gpu, N * N * sizeof(DATA_TYPE), 0));
		printf("cudaMemPrefetchAsync(%p, %ld MB)\n", a_gpu, N * N * sizeof(DATA_TYPE) / 1000000);
	}

#ifdef CUDA_CLI_HINTS
	if (cuda_hints[1][HINT_PREFETCH_ASYNC])
		CUDA_PF_ASYNC_GPU(a_gpu_k1, N * N * sizeof(DATA_TYPE));
	if (cuda_hints[2][HINT_PREFETCH_ASYNC])
		CUDA_PF_ASYNC_GPU(x1_gpu, N * sizeof(DATA_TYPE));
	if (cuda_hints[2][HINT_PREFETCH_ASYNC])
		CUDA_PF_ASYNC_GPU(x2_gpu, N * sizeof(DATA_TYPE));
	if (cuda_hints[3][HINT_PREFETCH_ASYNC])
		CUDA_PF_ASYNC_GPU(y_1_gpu, N * sizeof(DATA_TYPE));
	if (cuda_hints[4][HINT_PREFETCH_ASYNC])
		CUDA_PF_ASYNC_GPU(y_2_gpu, N * sizeof(DATA_TYPE));
#endif

	if (pin_input_cpu_hint) {
		CUDA_PIN_CPU_HINT(a_gpu_k1, a_size);
		if (per_kernel_array)
			CUDA_PIN_CPU_HINT(a_gpu_k2, a_size);
	}

	if (accessed_by_gpu_hint) {
		CUDA_ACCESSED_BY_GPU_HINT(a_gpu_k1, a_size);
		if (per_kernel_array)
			CUDA_ACCESSED_BY_GPU_HINT(a_gpu_k2, a_size);
	}

	if (pin_vectors_gpu) {
		CUDA_PIN_GPU_HINT(x1_gpu, N * sizeof(DATA_TYPE));
		CUDA_PIN_GPU_HINT(x2_gpu, N * sizeof(DATA_TYPE));
		CUDA_PIN_GPU_HINT(y_1_gpu, N * sizeof(DATA_TYPE));
		CUDA_PIN_GPU_HINT(y_2_gpu, N * sizeof(DATA_TYPE));
	}

	if (pin_vectors_cpu) {
		CUDA_PIN_CPU_HINT(x1_gpu, N * sizeof(DATA_TYPE));
		CUDA_PIN_CPU_HINT(x2_gpu, N * sizeof(DATA_TYPE));
		CUDA_PIN_CPU_HINT(y_1_gpu, N * sizeof(DATA_TYPE));
		CUDA_PIN_CPU_HINT(y_2_gpu, N * sizeof(DATA_TYPE));
	}
	mvtCuda(a_gpu_k1, a_gpu_k2, x1_gpu, x2_gpu, y_1_gpu, y_2_gpu);

	if (compare_with_cpu) {
		t_start = rtclock();

		//run the algorithm on the CPU
		runMvt(a, x1, x2, y_1, y_2);

		t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

		compareResults(x1, x1_gpu, x2, x2_gpu);
	}

	if (copy_back_gpu_results && compare_with_cpu == 0) {
		TOUCH_ARRAY(x1_gpu, sizeof(DATA_TYPE)*N);
		TOUCH_ARRAY(x2_gpu, sizeof(DATA_TYPE)*N);
	}

	t_end = rtclock();
	printf("CPU+GPU: %lf s | Mem %ld MB\n\n",
	                t_end - t_after_array_init, target/1000000);

	free(a);
	free(x1);
	free(x2);
	free(y_1);
	free(y_2);
	cudaFree(a_gpu_k1);
	cudaFree(a_gpu_k2);
	cudaFree(x1_gpu);
	cudaFree(x2_gpu);
	cudaFree(y_1_gpu);
	cudaFree(y_2_gpu);
	return 0;
}
