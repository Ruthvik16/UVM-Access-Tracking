#if __has_include("cuda-macros-v1.h")
#include "cuda-macros-v1.h"
#elif __has_include("../dev/cuda-macros-v1.h")
#include "../dev/cuda-macros-v1.h"
#else
#warning "Can't find cuda-macros-v1.h"
#endif

unsigned copy_back_gpu_results = 0;
unsigned compare_result_with_cpu = 0;
unsigned long access_stride = sizeof(int);

#define ARRAY_CONST_VALUE 4 // Fill the array with this
__global__ void write_stride_2m(int *arr) {
	long tid = threadIdx.x + blockIdx.x * blockDim.x;

	tid = (tid << 19);
	arr[tid] = ARRAY_CONST_VALUE;
	return;
}


__global__ void write_stride_4k(int *arr) {
	long tid = threadIdx.x + blockIdx.x * blockDim.x;

	tid = (tid << 10);
	arr[tid] = ARRAY_CONST_VALUE;
	return;
}

__global__ void write_stride_4b(int *arr) {
	long tid = threadIdx.x + blockIdx.x * blockDim.x;

	arr[tid] = ARRAY_CONST_VALUE;
	return;
}

int main(int argc, char *argv[]) {
	unsigned long data_size = 1 << 12;
	unsigned long block_size = 32;

	FOR_EACH_ARGUMENT {
		CHECK_ARG_AND_SET_VAL("-copy-back", copy_back_gpu_results, 1);
		CHECK_ARG_AND_SET_VAL("-compare", compare_result_with_cpu, 1);
		CHECK_ARG_AND_SET_PARAM("-data", data_size);
		CHECK_ARG_AND_SET_PARAM("-blk-size", block_size);
		CHECK_ARG_AND_SET_VAL("-stride-4k", access_stride, 4096);
		CHECK_ARG_AND_SET_VAL("-stride-2m", access_stride, (1<<21));
		if (strncmp(argv[iteration_counter], "-h", 2) == 0)
			return 1;
	}

	assert(access_stride == 4 || access_stride == 4096 || access_stride == (1 << 21));
	unsigned long arr_len = data_size/4;
	unsigned long num_threads = data_size / access_stride;

	unsigned long num_blks = (num_threads)/block_size; // take floor, not ceil
	if (num_blks == 0)
		printf("&&&&&&& WARNING &&&&&&&&&\nzero blocks: look at %s\n", __FILE__);

	assert(data_size >= (access_stride * block_size)); // won't check this condition in kernel
	arr_len = access_stride * block_size * num_blks;
	printf("Truncated arr_len to %lu to align with thread block size\n", arr_len);

	TRY_ALLOC_NON_UVM(int, d_arr, arr_len);

	int *arr = (int *)malloc(data_size);

	cudaMemcpy(d_arr, arr, data_size, cudaMemcpyHostToDevice);

	if (access_stride == 4)
		write_stride_4b<<<num_blks, block_size>>>(d_arr);
	else if (access_stride == 4096)
		write_stride_4k<<<num_blks, block_size>>>(d_arr);
	else if (access_stride == (1<<21))
		write_stride_2m<<<num_blks, block_size>>>(d_arr);

	TRY_DEVICE_SYNCHRONIZE();
	cudaError_t status = cudaMemcpy(arr, d_arr, data_size, cudaMemcpyDeviceToHost);
	assert(status == cudaSuccess); /* for the memcpy */

	if (compare_result_with_cpu) {
		for (unsigned long i = 0;
				i < arr_len;
				i+=(access_stride/sizeof(arr[0]))
				)

			if (arr[i] != ARRAY_CONST_VALUE) {
				printf("***** ERROR *****\n"
					"Index %ld: GPU %d\n",
					i, arr[i]
					);
				return -1;
			}
	}
	printf("**** COMPARISON DONE ****\n"); 
	return 0;
}
