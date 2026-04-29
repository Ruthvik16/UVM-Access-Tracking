/*
   This program is a sibling to gpu_scheduling_noise.cu. This generates many 
   faults, but doesn't occupy a lot of memory.

   We'll allocate about 2MB memory and touch it from both CPU and GPU on loop.
*/

#if __has_include("../dev/cuda-macros-v1.h")
#include "../dev/cuda-macros-v1.h"
#else
#error "Find cuda-macros-v1.h. Maybe you're compiling in some other directory?"
#endif

#define KERNEL_LOOP_FOR(duration)               \
        long now = clock64();                   \
        long elapsed = 0;                       \
        while (elapsed < duration) {            \
                elapsed = clock64() - now;      \
        }

__global__ void gen_gpu_fault(long *mem, long len, long n_iter) {

	for (int offset = 0; offset < len; offset += 4096/8) {
		// From gpu_scheduling_noise.cu, it appears that 
		// clock64() doesn't have much overhead
		mem[offset] += n_iter * offset;
		if (offset > 3 * n_iter)
			mem[offset] += 3;
		KERNEL_LOOP_FOR(1000);
	}
}

struct gen_faults_struct {
	long *mem;
	long len;
};

void *gen_cpu_faults(void *args_struct) {
	struct gen_faults_struct *st = (struct gen_faults_struct *)args_struct;
	long *mem = st->mem;
	long len = st->len;
	int n_64K_pages = len * 8 / 16 / 4096;

	struct timespec tm_sleep = {
		.tv_sec = 0,
		.tv_nsec = 500,
	};

	srandom((unsigned) (getpid() + len));

	// An infinite while loop is safe because returning from the main 
	// thread kills the thread group. That happens when the specified 
	// number of GPU iterations is completed
	while (1) {
		// one fault per 64K page, but random
		for (int i = 0; i < n_64K_pages; i++) {
			int offset = random();
			offset = offset % len;
			assert(offset < len);

			mem[offset] = offset * i + (i > offset);

			int ret = nanosleep(&tm_sleep, NULL);
			assert(ret == 0 /* unless you did ^C */);
		}
	}
	assert(0);
	return NULL;
}


int main(int argc, char *argv[]) {
	int array_len = 1 << 21;
	long n_iter = 1;
	int ret;
	int print_status = 1;

	FOR_EACH_ARGUMENT {
		CHECK_ARG_AND_SET_PARAM("-iter", n_iter);
		CHECK_ARG_AND_SET_PARAM("-len", array_len);
		CHECK_ARG_AND_SET_VAL("-silent", print_status, 0);
	}

	TRY_ALLOC_UVM(long, contention_region, array_len);
	memset(contention_region, 0, array_len<<3);

	struct gen_faults_struct str = {
		.mem = contention_region,
		.len = array_len,
	};

	pthread_t cpu_fault_tid;
	ret = pthread_create(&cpu_fault_tid, NULL, &gen_cpu_faults, (void *) &str);
	assert(ret == 0);

	for (long i = 0; i < n_iter; i++) {
		gen_gpu_fault<<<1, 1>>>(contention_region, array_len, i);
		TRY_DEVICE_SYNCHRONIZE();
		if (print_status)
			if ((i == 5000) || ((n_iter >100) && (i % (n_iter / 100)) == 0) || ((i % 20000) == 0))
				printf(__FILE__ ": finished iteration %ld/%ld\n", i, n_iter);
	}

	assert(sizeof(long) == 8);
	return 0;
}
