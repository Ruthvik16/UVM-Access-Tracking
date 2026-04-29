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
	long tid = threadIdx.x + (long) blockIdx.x * blockDim.x;

	tid = (tid << 19);
	arr[tid] = ARRAY_CONST_VALUE;
	return;
}

__global__ void write_stride_4k(int *arr) {
	long tid = threadIdx.x + (long) blockIdx.x * blockDim.x;

	tid = (tid << 10);
	arr[tid] = ARRAY_CONST_VALUE;
	return;
}

__global__ void write_stride_4b(int *arr) {
	long tid = threadIdx.x + (long) blockIdx.x * blockDim.x;

	arr[tid] = ARRAY_CONST_VALUE;
	return;
}

int main(int argc, char *argv[]) {
	unsigned long data_size = 1 << 12;
	unsigned long block_size = 32;
	unsigned pin_to_cpu_hint = 0;
	unsigned pin_to_gpu_hint = 0;
	unsigned read_mostly_hint = 0;
	unsigned accessed_by_gpu_hint = 0;
	unsigned num_iter = 1;
	unsigned sleep_sec = 0;
	unsigned init_on_cpu = 0;
	unsigned long data_size_mb = 0;

	for (int i = 0; i < argc; i++) {
		GET_BOOL_FLAG(i, "-copy-back", copy_back_gpu_results, 1);
		GET_BOOL_FLAG(i, "-compare", compare_result_with_cpu, 1);
		GET_INT_FLAG(i,  "-data", data_size);
		GET_INT_FLAG(i,  "-blk-size", block_size);
		GET_BOOL_FLAG(i, "-stride-4k", access_stride, 4096);
		GET_BOOL_FLAG(i, "-stride-2m", access_stride, (1<<21));
		GET_BOOL_FLAG(i, "-pin-cpu-hint", pin_to_cpu_hint, true);
		GET_BOOL_FLAG(i, "-pin-gpu-hint", pin_to_gpu_hint, true);
		GET_BOOL_FLAG(i, "-read-mostly-hint", read_mostly_hint, true);
		GET_BOOL_FLAG(i, "-accessed-by-gpu-hint", accessed_by_gpu_hint, true);
		GET_INT_FLAG(i,  "-iter", num_iter);
		GET_INT_FLAG(i,  "-sleep", sleep_sec);
		GET_BOOL_FLAG(i, "-cpu-init-first", init_on_cpu, 1);
		GET_INT_FLAG(i, "-mb", data_size_mb);
		get_hints(i, argv);
		if (strcmp(argv[i], "-h") == 0)
			return 1;
		UNRECOGNIZED_ARGUMENT(i);
	}

	if (data_size_mb)
		data_size = (data_size_mb << 20);

	assert(access_stride == 4 || access_stride == 4096 || access_stride == (1 << 21));
	unsigned long arr_len = data_size/4;
	unsigned long num_threads = data_size / access_stride;

	long num_blks = (num_threads)/block_size; // take floor, not ceil
	if (num_blks == 0)
		printf("&&&&&&& WARNING &&&&&&&&&\nzero blocks: look at %s\n", __FILE__);

	assert(data_size >= (access_stride * block_size)); // won't check this condition in kernel
	arr_len = access_stride * block_size * num_blks / sizeof(int);
	printf("Truncated arr_len to %lu to align with thread block size\n", arr_len);

	int *d_arr;
	UVM_ALLOC_BUF(int, d_arr, arr_len * sizeof(int));
	mickey_clear();
	mickey_register_va(d_arr + (1<<30));
	mickey_register_va(d_arr + (1<<28));
	unsigned long arr_size = arr_len * sizeof(d_arr[0]);

	// Now the hints
	if (read_mostly_hint) {
		CHECK_RETURN_VALUE(cudaMemAdvise(d_arr, arr_size,
					cudaMemAdviseSetReadMostly,
					0));
		printf("Setting cudaMemAdvise ReadMostly\n");
	}
	if (pin_to_cpu_hint) {
		CHECK_RETURN_VALUE(cudaMemAdvise(d_arr, arr_size,
					cudaMemAdviseSetPreferredLocation,
					cudaCpuDeviceId));
		printf("Setting cudaMemAdvise PreferredLocation cudaCpuDeviceId\n");
	} else if (pin_to_gpu_hint) {
		CHECK_RETURN_VALUE(cudaMemAdvise(d_arr, arr_size,
					cudaMemAdviseSetPreferredLocation,
					0));
		printf("Setting cudaMemAdvise PreferredLocation 0 (GPU)\n");
	}
	if (accessed_by_gpu_hint) {
		CHECK_RETURN_VALUE(cudaMemAdvise(d_arr, arr_size,
					cudaMemAdviseSetAccessedBy,
					0));
		printf("Setting cudaMemAdvise SetAccessedBy 0\n");
	}

	if (init_on_cpu) {
		double t1, t2;
		t1 = gettime();
		for (long i = 0; i < arr_len; i += access_stride)
			d_arr[i] = 43;
		t2 = gettime();
		printf("# CPU init: %.3lf s\n", t2 - t1);
	}


	for (int i = 0; i < num_iter; i++) {
		assert(d_arr);
		double t1 = gettime();
		if (access_stride == 4)
			write_stride_4b<<<num_blks, block_size>>>(d_arr);
		else if (access_stride == 4096)
			write_stride_4k<<<num_blks, block_size>>>(d_arr);
		else if (access_stride == (1<<21))
			write_stride_2m<<<num_blks, block_size>>>(d_arr);
		if (sleep_sec) {
			printf("[H2D] Entering sleep %u s\n", sleep_sec);
			sleep(sleep_sec);
			printf("Completed sleep\n");
		}
		CHECK_RETURN_VALUE(cudaDeviceSynchronize());
		printf("# completed iteration %d: %.3lf s\n", i + 1, gettime() - t1);
		if (!compare_result_with_cpu)
			continue;

		TOUCH_ARRAY(d_arr, arr_size);
		if (sleep_sec) {
			printf("[D2H] Entering sleep %u s\n", sleep_sec);
			sleep(sleep_sec);
			printf("Completed sleep\n");
		}
	}

	TRY_DEVICE_SYNCHRONIZE();

	if (compare_result_with_cpu) {
		for (unsigned long i = 0;
				i < arr_len;
				i+=(access_stride/sizeof(d_arr[0]))
			) {

			if (d_arr[i] != ARRAY_CONST_VALUE) {
				printf("***** ERROR *****\n"
						"Index %ld: GPU %d\n",
						i, d_arr[i]
					  );
				return -1;
			}
		}
		if (sleep_sec) {
			printf("Entering sleep %u s\n", sleep_sec);
			sleep(sleep_sec);
		}
		printf("**** COMPARISON DONE ****\n");
	} else {
		printf("**** SKIPPING COMPARISON. EXITING. ****\n");
	}
	return 0;
}
