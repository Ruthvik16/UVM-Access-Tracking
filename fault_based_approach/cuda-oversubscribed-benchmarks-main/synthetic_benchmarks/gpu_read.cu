#if __has_include("cuda-macros-v1.h")
#include "cuda-macros-v1.h"
#elif __has_include("../dev/cuda-macros-v1.h")
#include "../dev/cuda-macros-v1.h"
#else
#warning "Can't find cuda-macros-v1.h"
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#define DTYPE uint64_t

/*
 * Synthetic benchmark: if a[i] and a[i+1] have a difference divisible by
 * @difference, then double them.
 */
__global__ void check_neighbours(
		DTYPE *arr,
		long arr_len,
		long difference,
		int access_sparsity
		)
{
	long tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	if ((tid % access_sparsity == 0) && (tid <= arr_len - 1)) {
		if ((arr[tid+1] - arr[tid]) % difference == 0) {
			DTYPE multiplier = (arr[tid] * arr[tid+1]);
			multiplier %= (multiplier ? multiplier * multiplier : 1010);
			// add some complicated code for the branch predictor to fail.
			arr[tid] *= 2;
			arr[tid+1] *= 2;
		}
	}
}


int main(int argc, char *argv[]) {
	unsigned long data_size = 1 << 12;
	unsigned long block_size = 32;
	bool read_from_file = false;
	bool memadvise_read_mostly = false;
	int data_reuse_iters = 1;
	int access_sparsity = 1;
	DTYPE neighbour_delta = 100;

	FOR_EACH_ARGUMENT {
		CHECK_ARG_AND_SET_PARAM("-data", data_size);
		CHECK_ARG_AND_SET_PARAM("-blk-size", block_size);
		CHECK_ARG_AND_SET_PARAM("-neighbour-delta", neighbour_delta);
		CHECK_ARG_AND_SET_VAL("-read-file", read_from_file, true);
		CHECK_ARG_AND_SET_VAL("-read-mostly", memadvise_read_mostly, true);
		CHECK_ARG_AND_SET_PARAM("-iters", data_reuse_iters);
		CHECK_ARG_AND_SET_PARAM("-sparsity", access_sparsity);
		if (strncmp(argv[iteration_counter], "-h", 2) == 0)
			return 1;
	}
	assert(neighbour_delta != 0);

	unsigned long arr_len = data_size/sizeof(DTYPE);

	long num_blks = (arr_len)/block_size; // take floor, not ceil
	if (num_blks == 0)
		printf("&&&&&&& WARNING &&&&&&&&&\nzero blocks: look at %s\n", __FILE__);

	DTYPE *d_arr;
	UVM_ALLOC_ARR(DTYPE, d_arr, arr_len);

	clock_t start, end;
	long duration_ms;

	// get random data
	start = clock();

	FILE *random_fp;
	if (read_from_file)
		random_fp = fopen("/var/cuda-repo-ubuntu2204-12-9-local/libcublas-12-9_12.9.0.13-1_amd64.deb", "r");
	else
		random_fp = fopen("/dev/random", "r");

	assert(random_fp);
	long remaining = data_size;
	char *write_here = (char *)d_arr;
	while (remaining > 0) {
		long retval = fread(write_here, 1, remaining, random_fp);
		assert(retval > 0);
		assert(fseek(random_fp, 0, SEEK_SET) == 0);
		remaining =- retval;
		write_here += retval;
	}
	assert(remaining <= 0);
	fclose(random_fp);

	end = clock();
	duration_ms = (end - start) * 1000 / CLOCKS_PER_SEC;

	printf("Init random data: %.2f s\n", ((float) duration_ms) / 1000);

	if (memadvise_read_mostly) {
		remaining = data_size;
		write_here = (char *) d_arr;
		while (remaining > 0) {
			cudaError_t ret = cudaMemAdvise(
				(void *) write_here,
				((remaining > (1<<30)) ? (1<<30) : remaining),
				cudaMemAdviseSetReadMostly,
				0
				);
			assert(ret == cudaSuccess);
			remaining -= (1<<30);
			write_here += (1<<30);
		}
	}


	// find stdev
	start = clock();

	for (int i = 0; i < data_reuse_iters; i++) {
		check_neighbours<<<num_blks, block_size>>>(d_arr, arr_len, neighbour_delta, access_sparsity);
		TRY_DEVICE_SYNCHRONIZE();
	}

	end = clock();
	duration_ms = (end - start) * 1000 / CLOCKS_PER_SEC;
	printf("%s: GPU access: %.3f s\n", __FILE__, ((float) duration_ms) / 1000);


	start = clock();
	TOUCH_ARRAY(d_arr, arr_len);
	end = clock();
	duration_ms = (end - start) * 1000 / CLOCKS_PER_SEC;

	printf("%s: CPU access: %.3f s\n", __FILE__, ((float) duration_ms) / 1000);

	return 0;
}

