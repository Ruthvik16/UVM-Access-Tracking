/**
 * fdtd2d.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

#define GPU_DEVICE 0

/* Problem size */
#define tmax 10
long NX, NY;


/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;


void init_arrays(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz, DATA_TYPE* _fict_gpu, DATA_TYPE* ex_gpu, DATA_TYPE* ey_gpu, DATA_TYPE* hz_gpu)
{
	int i, j;
	double t2, t1 = gettime();
	bool compare = (ex != NULL) && (ey != NULL) && (hz != NULL) && (_fict_ != NULL);

	for (i = 0; i < tmax; i++)
	{
		if (compare)
			_fict_[i] = (DATA_TYPE) i;
		_fict_gpu[i] = (DATA_TYPE) i;
	}
	for (i = 0; i < NX; i++)
	{
		for (j = 0; j < NY; j++)
		{
			ex_gpu[i*NY + j] = ((DATA_TYPE) i*(j+1) + 1) / NX;
			ey_gpu[i*NY + j] = ((DATA_TYPE) (i-1)*(j+2) + 2) / NX;
			hz_gpu[i*NY + j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / NX;
			if (compare) {
				ex[i*NY + j] = ((DATA_TYPE) i*(j+1) + 1) / NX;
				ey[i*NY + j] = ((DATA_TYPE) (i-1)*(j+2) + 2) / NX;
				hz[i*NY + j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / NX;
			}
		}
	}

	t2 = gettime();
	printf("init ex[] ey[] hz[]: %.3lf s\n", t2 - t1);
}


void runFdtd(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
	int t, i, j;

	for (t=0; t < tmax; t++)
	{
		for (j=0; j < NY; j++)
		{
			ey[0*NY + j] = _fict_[t];
		}

		for (i = 1; i < NX; i++)
		{
			for (j = 0; j < NY; j++)
			{
				ey[i*NY + j] = ey[i*NY + j] - 0.5*(hz[i*NY + j] - hz[(i-1)*NY + j]);
			}
		}

		for (i = 0; i < NX; i++)
		{
			for (j = 1; j < NY; j++)
			{
				ex[i*(NY+1) + j] = ex[i*(NY+1) + j] - 0.5*(hz[i*NY + j] - hz[i*NY + (j-1)]);
			}
		}

		for (i = 0; i < NX; i++)
		{
			for (j = 0; j < NY; j++)
			{
				hz[i*NY + j] = hz[i*NY + j] - 0.7*(ex[i*(NY+1) + (j+1)] - ex[i*(NY+1) + j] + ey[(i+1)*NY + j] - ey[i*NY + j]);
			}
		}

		if ((t & (t - 1)) == 0)
			printf("completed iteration %d\n", t);
	}
}


void compareResults(DATA_TYPE* hz1, DATA_TYPE* hz2)
{
	int i, j, fail;
	fail = 0;
	
	for (i=0; i < NX; i++)
	{
		for (j=0; j < NY; j++)
		{
			if (percentDiff(hz1[i*NY + j], hz2[i*NY + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}



__global__ void fdtd_step1_kernel(DATA_TYPE* _fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t, long NX, long NY)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NX) && (j < NY))
	{
		if (i == 0)
		{
			ey[i * NY + j] = _fict_[t];
		}
		else
		{
			ey[i * NY + j] = ey[i * NY + j] - 0.5f*(hz[i * NY + j] - hz[(i-1) * NY + j]);
		}
	}
}



__global__ void fdtd_step2_kernel(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t, long NX, long NY)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i < NX) && (j < NY) && (j > 0))
	{
		ex[i * (NY+1) + j] = ex[i * (NY+1) + j] - 0.5f*(hz[i * NY + j] - hz[i * NY + (j-1)]);
	}
}


__global__ void fdtd_step3_kernel(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t, long NX, long NY)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i < NX) && (j < NY))
	{	
		hz[i * NY + j] = hz[i * NY + j] - 0.7f*(ex[i * (NY+1) + (j+1)] - ex[i * (NY+1) + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
	}
}


void fdtdCuda(DATA_TYPE* _fict_gpu, DATA_TYPE* ex_gpu, DATA_TYPE* ey_gpu, DATA_TYPE* hz_gpu)
{
	double t_start, t_end;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid( (size_t)ceil(((float)NY) / ((float)block.x)), (size_t)ceil(((float)NX) / ((float)block.y)));

	t_start = rtclock();

	for(int t = 0; t< tmax; t++)
	{
		fdtd_step1_kernel<<<grid,block>>>(_fict_gpu, ex_gpu, ey_gpu, hz_gpu, t, NX, NY);
		cudaDeviceSynchronize();
		fdtd_step2_kernel<<<grid,block>>>(ex_gpu, ey_gpu, hz_gpu, t, NX, NY);
		cudaDeviceSynchronize();
		fdtd_step3_kernel<<<grid,block>>>(ex_gpu, ey_gpu, hz_gpu, t, NX, NY);
		cudaError_t ret = cudaDeviceSynchronize();
		assert(ret == cudaSuccess);
		CHECK_CUDA_ERROR();

		if ((t & (t - 1)) == 0)
			printf("completed iteration %d\n", t);
	}

	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.3lfs\n", t_end - t_start);
}


int main(int argc, char *argv[])
{
	double t_start, t_end;

	long data_size = 1<<25, data_mb = 0;
	if (argc > 1 && argv[1][0] != '-')
		data_size = atol(argv[1]);

	compare_with_cpu = false;
	copy_back_gpu_results = 1;
	for (int i = 0; i < argc; i++) {
		GET_BOOL_FLAG(i, "-compare", compare_with_cpu, true);
		GET_BOOL_FLAG(i, "-skip-copy-back", copy_back_gpu_results, false);
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
		data_size = data_mb * 1000000;

	NX = get_data_size(data_size/3, sizeof(DATA_TYPE), 2);
	NY = NX;

	DATA_TYPE* _fict_;
	DATA_TYPE* ex;
	DATA_TYPE* ey;
	DATA_TYPE* hz;

	DATA_TYPE *_fict_gpu;
	DATA_TYPE *ex_gpu;
	DATA_TYPE *ey_gpu;
	DATA_TYPE *hz_gpu;

	_fict_ = (DATA_TYPE*)malloc(tmax*sizeof(DATA_TYPE));
	ex = (DATA_TYPE*)malloc(NX*(NY+1)*sizeof(DATA_TYPE));
	ey = (DATA_TYPE*)malloc((NX+1)*NY*sizeof(DATA_TYPE));
	hz = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));

	UVM_ALLOC_ARR(DATA_TYPE, _fict_gpu, tmax);
	UVM_ALLOC_ARR(DATA_TYPE, ex_gpu, (NX * (NY + 1)));
	UVM_ALLOC_ARR(DATA_TYPE, ey_gpu, (NY * (NX + 1)));
	UVM_ALLOC_ARR(DATA_TYPE, hz_gpu, (NX * NY));

	init_arrays(_fict_, ex, ey, hz, _fict_gpu, ex_gpu, ey_gpu, hz_gpu);

	GPU_argv_init();
	fdtdCuda(_fict_gpu, ex_gpu, ey_gpu, hz_gpu);

	if (compare_with_cpu) {
		t_start = rtclock();
		runFdtd(_fict_, ex, ey, hz);
		t_end = rtclock();

		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

		compareResults(hz, hz_gpu);
	} else {
		TOUCH_ARRAY(hz, (NX * NY * sizeof(DATA_TYPE)));
	}

	free(_fict_);
	free(ex);
	free(ey);
	free(hz);

	cudaFree(_fict_gpu);
	cudaFree(ex_gpu);
	cudaFree(ey_gpu);
	cudaFree(hz_gpu);
	return 0;
}

