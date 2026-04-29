/*
 * Simple RD is very slow with initializing data on the CPU.
 * This benchmark captures that difference.
 * To be a bit realistic, there's a -random flag to use random data like
 * Polybench does.
 */

#if __has_include("cuda-macros-v1.h")
#include "cuda-macros-v1.h"
#elif __has_include("../dev/cuda-macros-v1.h")
#include "../dev/cuda-macros-v1.h"
#else
#warning "Can't find cuda-macros-v1.h"
#endif

int main(int argc, char *argv[]) {
	unsigned long data_size = (1 << 12);
	unsigned long block_size = 32;
	unsigned access_stride = sizeof(unsigned);
	unsigned init_random = 0;
	unsigned read_access = 0;
	unsigned set_preferred_location_hint = 0;
	unsigned long data_mb = 0;

	if (argv[1] && atol(argv[1])) {
		fprintf(stderr, "Unrecognized argument %s, did you miss the \"-data\"?\n", argv[1]);
		return -1;
	}

	for(int i = 1; i < argc; i++) {
		GET_INT_FLAG(i, "-data", data_size);
		GET_BOOL_FLAG(i, "-random", init_random, 1);
		GET_BOOL_FLAG(i, "-read", read_access, 1);
		GET_INT_FLAG(i, "-blk-size", block_size);
		GET_BOOL_FLAG(i, "-stride-4k", access_stride, 4096);
		GET_BOOL_FLAG(i, "-stride-2m", access_stride, (1<<21));
		GET_BOOL_FLAG(i, "-preferred-loc-hint", set_preferred_location_hint, 1);
		GET_INT_FLAG(i, "-mb", data_mb);
		get_hints(i, argv);
		if (strncmp(argv[i], "-h", 2) == 0)
			return 1;
		UNRECOGNIZED_ARGUMENT(i); 
	}

	if (data_mb)
		data_size = data_mb << 20;

	assert(access_stride == 4 || access_stride == 4096 || access_stride == (1<<21));

	clock_t start, end;
	unsigned long duration_ms;

	start = clock();
	TRY_DEVICE_SYNCHRONIZE();
	end = clock();
	duration_ms = (end - start) * 1000 / CLOCKS_PER_SEC;
	printf("%s: driver init: %.3f s\n",
		__FILE__,
		((float) duration_ms) / 1000
		);

	start = clock();
	TRY_ALLOC_UVM_BY_DATA_SIZE(unsigned, cpu_arr, data_size);

	unsigned long arr_len = data_size / sizeof(unsigned);
	unsigned long pid = (unsigned long) getpid();

	if (set_preferred_location_hint) {
		CUDA_PIN_CPU_HINT(cpu_arr, data_size);
	}

	for (unsigned long i = 0;
			i < arr_len;
			i += access_stride/sizeof(unsigned)
			) {
		if (read_access) {
			if (cpu_arr[i] == (i ^ pid) && ((unsigned long)cpu_arr + i == pid))
				fprintf(stderr, "Fake compiler dependency\n");
		}
		else if (init_random) {
			cpu_arr[i] = (unsigned) random();
		} else {
			cpu_arr[i] = 42;
		}
	}
	end = clock();

	duration_ms = (end - start) * 1000 / CLOCKS_PER_SEC;

	if (cpu_arr[pid % arr_len] == 45)
		printf("gotcha\n");

	printf("%s: init %ld MB | CPU: %.3f s\n", __FILE__,
		data_size/1000000,
		((float) duration_ms)/1000
		);
	return 0;
}
