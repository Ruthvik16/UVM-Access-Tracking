/*
   Scant details are available about GPU microarchitecture. The mechanisms
   and overhead of scheduling, context switches and SM allocation, for 
   instance, are not known.

   This program creates some on-GPU noise by introducing stray faults and
   scheduling stuff. We plan to run this in parallel with compute-intensive
   and memory-intensive benchmarks to see what happens.
   By compute-intensive, I mean something like matrix multiplication which 
   takes super-linear time and is somewhat immune to fault servicing 
   latency.

   pseudocode:

   long timestamps[array_len];
   for (i < num_iter) {
       fill_timestamps(timestamps);
   }
   TOUCH_ARRAY(timestamps); // from CPU

*/

#if __has_include("cuda-macros-v1.h")
#include "cuda-macros-v1.h"
#elif __has_include("/data/pranjal/cuda-oversubscribed-benchmarks/dev/cuda-macros-v1.h")
#include "/data/pranjal/cuda-oversubscribed-benchmarks/dev/cuda-macros-v1.h"
#else
#error "can't access the headers and helpers file. Copy the full repo"
#endif

__global__ void fill_timestamps(long *array, long num_iter, long fill_ts_or_const) {
	for (long i = 0; i < num_iter; i++)
		array[i] = fill_ts_or_const ?  clock64() : 3;
}

int main(int argc, char *argv[]) {
	long ts_array_len = 1 << 30;
	int num_iter = 1;
	long fill_ts_or_const = 1;

	if (argc > 1 && argv[1][0] >= '0' && argv[1][0] <= '9')
		ts_array_len = atol(argv[1]);
	FOR_EACH_ARGUMENT {
		CHECK_ARG_AND_SET_PARAM("-len", ts_array_len);
		CHECK_ARG_AND_SET_PARAM("-iter", num_iter);
		CHECK_ARG_AND_SET_VAL("-no-timestamp", fill_ts_or_const, 0);
	}

	for (int i = 1; i < argc; i++) {
		if (strncmp(argv[i], "-help", 2) == 0) {
			printf("Usage: ./a.out [num steps] [options ..]\n");
			printf("Source file: %s\n", __FILE__);
			printf("Pages are not accessed from the CPU between iterations\n");
			return 1;
		}
	}

	clock_t t1, t2, t3, t4, t5, t6;
	t1 = clock();

	TRY_ALLOC_UVM(long, timestamps, ts_array_len);
	memset(timestamps, 0, sizeof(long) * ts_array_len);

	t2 = clock();

	for (int i = 0; i < num_iter; i++) {
		fill_timestamps<<<1, 1>>>(timestamps, ts_array_len, fill_ts_or_const);
		TRY_DEVICE_SYNCHRONIZE();
	}

	t3 = clock();

	TOUCH_ARRAY(timestamps, ts_array_len << 3);
	t4 = clock();

	for (int i = 0; i < num_iter; i++) {
		fill_timestamps<<<1, 1>>>(timestamps, ts_array_len, fill_ts_or_const);
		TRY_DEVICE_SYNCHRONIZE();
	}

	t5 = clock();

	TOUCH_ARRAY(timestamps, ts_array_len << 3);
	t6 = clock();

	printf("Array init time: %ld ms\nGPU access time: %ld ms\n"
		"CPU touch time: %ld ms\nSecond GPU Touch time: %ld ms\n"
		"Second CPU touch time: %ld ms\n",
			(t2 - t1)/1000,
			(t3 - t2)/1000,
			(t4 - t3)/1000,
			(t5 - t4)/1000,
			(t6 - t5)/1000
		);
	// find some way to process all the data	

	assert(CLOCKS_PER_SEC == 1000000);
	assert(sizeof(long) == 8);
	return 0;
}
