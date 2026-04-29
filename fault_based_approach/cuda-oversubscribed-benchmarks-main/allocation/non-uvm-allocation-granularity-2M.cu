/*
	What is the exact allocation granularity for cudaMalloc?
	We allocate 1 B, but there is no error in running with 
	nthreads = 2^19 or data size = 2^21 = VA block size
	At 2^22 or even (2^21)+4, cudaDeviceSynchronize() returns an error.
	2^21 + 3 is rounded off to 2^21 after division by sizeof(float)
	
	CONCLUSION:
	This shows allocations are at a 2M granularity at least on device
	cudaMemcpy size neeeds to be <= cudaMalloc size. Else, either drivers or
	userspace throws an error.
*/

#include "../dev/cuda-macros-v1.h"

#define CUDA_MALLOC_SIZE 2
#define CUDA_MEMCPY_H2D_SIZE 1
#define CUDA_MEMCPY_D2H_SIZE 2

int main(int argc, char *argv[]) {
	CHECK_ARGC_SHOW_USAGE(3, "./a.out <2^A data size> <add B bytes (0 if you want)>\n1 Byte is allocated using cudaMallocManaged, and then (2^A) + B bytes are accessed.\nError upon crossing 2M limit");

	TRY_ALLOC_NON_UVM_BY_DATA_SIZE(float, farray, CUDA_MALLOC_SIZE);
	float h = 44.0;
	CHECK_RETURN_VALUE(cudaMemcpy(farray, &h, CUDA_MEMCPY_H2D_SIZE, cudaMemcpyHostToDevice));

	long nthreads = (1 << atol(argv[1]))/sizeof(float);
	if (argc > 2)
		nthreads += (atol(argv[2]) / sizeof(float));

	printf("Allocated %d bytes. Now we will access 0x%lx bytes at that address\n", CUDA_MALLOC_SIZE, nthreads * sizeof(float));

	// +255 for ceil-like behaviour
	initialize_array<<<(nthreads + 255)/256, 256>>>(farray, nthreads*sizeof(float), 3.141592);
	CHECK_RETURN_VALUE(cudaDeviceSynchronize());

	float *cpu_array = (float *) malloc(CUDA_MEMCPY_D2H_SIZE);
	if (cpu_array == NULL)
		return printf("Error in malloc() %d bytes\n", CUDA_MEMCPY_D2H_SIZE);

	
	CHECK_RETURN_VALUE(cudaMemcpy(cpu_array, farray, CUDA_MEMCPY_D2H_SIZE, cudaMemcpyDeviceToHost));
	for (int i = 0; i + (sizeof(float)-1) < CUDA_MEMCPY_D2H_SIZE; i+= 4) {
		float diff = (cpu_array[(i >> 2)] - 3.141592);
		if (diff * diff > 0.0001)
			return printf("cpu_array: index %d value is %f instead of 3.141592\n",
					i, cpu_array[(i>>2)]);
	}

	printf("Execution complete\n");
	return 0;
}

