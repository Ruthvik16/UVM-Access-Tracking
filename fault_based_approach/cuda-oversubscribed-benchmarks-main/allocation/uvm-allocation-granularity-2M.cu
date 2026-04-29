/*
	What is the exact allocation granularity?
	We allocate 1 B, but there is no error in running with 
	nthreads = 2^19 or data size = 2^21 or VA block size
	At 2^20 or even (2^19)+1, cudaDeviceSynchronize() returns an error.
	
	CONCLUSION:
	UVM allocations are at a 2M granularity
*/

#include "../dev/cuda-macros-v1.h"

#define CUDA_MALLOC_MANAGED_SIZE 1

int main(int argc, char *argv[]) {
	CHECK_ARGC_SHOW_USAGE(3, "./a.out <2^A data size> <add B bytes (0 if you want)>\n1 Byte is allocated using cudaMallocManaged, and then (2^A) + B bytes are accessed.");
	TRY_ALLOC_UVM_BY_DATA_SIZE(float, farray, CUDA_MALLOC_MANAGED_SIZE);
	long data_size = 1 << atol(argv[1]);
	if (argc > 2)
		data_size += atol(argv[2]);

	long nthreads = data_size/sizeof(float);
	printf("Allocated %d bytes. Now we will access 0x%lx bytes at that pointer\n",
		   CUDA_MALLOC_MANAGED_SIZE,
			nthreads * sizeof(float));
	initialize_array<<<(nthreads + 255)/256, 256>>>(farray, data_size, 3.141592);
	CHECK_RETURN_VALUE(cudaDeviceSynchronize());

	long i;
	for (i = 0;
			i + (sizeof(float)-1) < data_size; // full float should fit inside array
			i+= sizeof(float)) {
		float diff = (farray[(i/sizeof(float))] - 3.141592);
		if (diff * diff > 0.0001)
			return printf("cpu_array: index %ld value is %f instead of 3.141592\n",
					(i/sizeof(float)), farray[(i/sizeof(float))]);
	}
	printf("Checked UVM array's contents: 0x%lx elements or %ld MiB was filled with 3.14 in GPU and was read in CPU\n",
			(i/sizeof(float)), (i >> 20));

	printf("Execution complete\n");
	return 0;
}
