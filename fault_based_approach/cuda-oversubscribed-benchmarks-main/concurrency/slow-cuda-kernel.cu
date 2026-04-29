#include "../dev/cuda-macros-v1.h"
#include <time.h>
#include <sys/types.h>

/*
   This kernel runs for O(2^(argv[2])) time, but has negligible memory transfers
   or fault overhead. We use it to see the effect of multiple CUDA processes.

   Using unsigned long slows down execution about 4x

   Prints execution time (kernel call to cudaDeviceSynchronize().
   See the help message for the config options

   TLDR - we are unable to create a situation where two calls to this function
   take <= twice the time for single run
*/

#define ARRAY_DATATYPE unsigned long

__global__ void slow_cuda_kernel(ARRAY_DATATYPE *i1, int len, unsigned long n_iter) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= len)
		return;
	
	i1[idx] = 0;
	for (int k = 0; k < n_iter; k++) {
		i1[idx] += 1;
	}
}

// Arguments for child thread, to match the pthreads function prototype
struct execution_params {
	unsigned long array_len;
	unsigned long n_iter;
	int threads_per_block;
	int n_kernels;
	bool allocation_padding;
};

void *execute_kernel(void *params) {
	execution_params *p = (execution_params *) params;

	// allocate both arrays
	unsigned long alloc_size = p->array_len;
	if (p->allocation_padding) {
		long padding = 0x200000 * 256;
		alloc_size = (alloc_size + padding - 1)/padding;
		alloc_size *= padding;
	}

	TRY_ALLOC_UVM(ARRAY_DATATYPE, i1, alloc_size);
	TRY_ALLOC_UVM(ARRAY_DATATYPE, i2, alloc_size);

	if (p->n_kernels == 1)
		cudaFree(i2);

	clock_t start, end;

	int threads_per_block = p->threads_per_block;

	long num_blocks = (p->array_len + (threads_per_block - 1))/threads_per_block;

	start = clock();
	slow_cuda_kernel<<<num_blocks, threads_per_block>>>(i1, p->array_len, p->n_iter);

	if (p->n_kernels == 2)
		slow_cuda_kernel<<<num_blocks, threads_per_block>>>(i2, p->array_len, p->n_iter);

	CHECK_RETURN_VALUE(cudaDeviceSynchronize());

	end = clock();

	printf("After execution array[0] is 0x%lx Expected value 0x%lx | #kernels = %d\n", i1[0], p->n_iter, p->n_kernels);
	printf("Execution time from clock(): %f s\n", ((float)(end - start))/CLOCKS_PER_SEC);
	return NULL;
}

int main(int argc, char *argv[]) {
	CHECK_ARGC_SHOW_USAGE(3, "./a.out <array length> <log [n_iter]> [-fork] [-thread] "
			"[-two-kernels-async] [-padding] [-1024] [-256] [-1]\n"
			"-1, -256, -1024 are number of threads per block. Default 1\n"
			"-two-kernels-async makes two calls to the kernel without cudaDeviceSynchronize() between them\n"
			"-fork and -thread can be used together to run 4 times\n"
			"-padding rounds up allocation size to 2M block\n");

	unsigned long array_len = atol(argv[1]);
	unsigned long n_iter = atol(argv[2]);
	if (n_iter > 63)
		return printf("Can't do 1<<%ld iterations, out of range\n", n_iter);
	n_iter = 1UL << n_iter;

	bool two_threads = false;
	bool do_fork = false;

	struct execution_params ep = {
		.array_len = array_len,
		.n_iter = n_iter,
		.threads_per_block = 1,
		.n_kernels = 1,
		.allocation_padding = false,
	};

	FOR_EACH_ARGUMENT {
		// directly update struct ep
		CHECK_ARG_AND_SET_VAL("-fork", do_fork, true);
		CHECK_ARG_AND_SET_VAL("-thread", two_threads, true);
		CHECK_ARG_AND_SET_VAL("-padding", ep.allocation_padding, true);
		CHECK_ARG_AND_SET_VAL("-1024", ep.threads_per_block, 1024);
		CHECK_ARG_AND_SET_VAL("-256", ep.threads_per_block, 256);
		CHECK_ARG_AND_SET_VAL("-1", ep.threads_per_block, 1);
		CHECK_ARG_AND_SET_VAL("-two-kernels-async", ep.n_kernels, 2);
	}

	// Package the arguments into a struct for execute_kernel()

	int retval;
	if (do_fork) {
		retval = fork();
		if (retval < 0)
			return printf("Error in fork()\n");
	}

	// First launch the second thread, else pthread_join will have to wait 
	// for the first to complete execution.
	pthread_t tpid;
	if (two_threads) {
		CHECK_RETURN_VALUE_ZERO(pthread_create(&tpid, NULL, &execute_kernel, &ep));
	}

	// Now execute the kernel from the parent process
	execute_kernel(&ep);

	// Wait for child thread to exit before return
	if (two_threads) {
		CHECK_RETURN_VALUE_ZERO(pthread_join(tpid, NULL));
	}

	return 0;
}
