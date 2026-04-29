#include <unistd.h>
#include <stdio.h>
#include "cuda.h"
#include <assert.h>

__global__ void store_square(unsigned *arr, unsigned long arr_len) {
	unsigned long tid =  threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < arr_len/2 && tid % 15 != 0)
		arr[tid*2 + 1] = arr[tid] * arr[tid];
	else if (tid < arr_len && tid % 13 == 7)
		arr[tid] = 55443322;
}

__global__ void blind_write(unsigned *arr, unsigned long arr_len) {
	unsigned long tid =  threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < arr_len && tid % 15 == 0)
		arr[tid] = 142857;
}

int main(int argc, char *argv[]) {
	unsigned long ar_len = 10UL << 30;
	unsigned *arr;
	cudaError_t err = cudaMallocManaged(&arr, ar_len * sizeof(arr[0]));
	assert(err == cudaSuccess);
	printf("allocated array arr[%lu] at %p\n", ar_len, arr);

	for (unsigned long i = 0; i < ar_len/2; i++) {
		arr[i*2] = i * i * i % 54321;
		if ((i & 0xfffff) == 0)
			printf("array init at index %lu\n", i);
	}

	store_square<<<(ar_len + 31)/32 , 32>>>(arr, ar_len);
	// blind_write<<<(ar_len + 31)/32, 32>>>(arr, ar_len);
	sleep(3);
	err = cudaDeviceSynchronize();
	assert(err == cudaSuccess);

	return 0;
}
