#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "dir_name.h"

__global__ void vector_set(float *ar, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < len) {
		ar[idx] = (float) idx;
	}
}

__global__ void vector_add(float *ar, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < len) {
		ar[idx] += (float) idx;
	}
}

int main(int argc, char *argv[]) {
	cudaError_t ret;
	float *gpu_array;
	
	if (argc < 2) {
		printf("Usage: ./fill-memory <array size in B>\n");
		return 1;
	}

	long len = atol(argv[1]);
	long numfl = len/4; // number of array elements

	unlink(STOP_FILE_ABS_PATH);

	ret = cudaMalloc(&gpu_array, len);

	if (ret != cudaSuccess) {
		printf("Error in cudaMalloc %ld MB\n", len >> 20);
		return 1;
	}

	int bsize = 256;
	int numBlocks = (numfl + bsize - 1)/bsize; // ceil

	vector_set<<<numBlocks, bsize>>>(gpu_array, numfl);

	ret = cudaDeviceSynchronize();
	if (ret != cudaSuccess)
		return printf("Error in first cudaDeviceSynchronize\n");

	int scan_val ;
	scan_val = getpid();
	struct stat statbuf;
	while (stat(STOP_FILE_ABS_PATH, &statbuf) != 0)
		sleep(5);
	
	vector_add<<<numBlocks, bsize>>>(gpu_array, numfl);

	ret = cudaDeviceSynchronize();
	if (ret != cudaSuccess)
		return printf("Error in second cudaDeviceSynchronize\n");


	printf("Scanned int value is %d\n", scan_val);
	if (scan_val + 1024 >= numfl) {
		printf("Expected a smaller value. Changing to 5\n");
		scan_val = 5;
	}


	float copy_here[1024];
	ret = cudaMemcpy(&copy_here, &gpu_array[scan_val], 4096, cudaMemcpyDeviceToHost);

	if (ret != cudaSuccess)
		return printf("Error in cudaMemcpy\n");

	
	ret = cudaDeviceSynchronize();
	if (ret != cudaSuccess)
		return printf("Error in third cudaDeviceSynchronize\n");

	printf("Expected value in array: %d | Actual value %f\n", 2 * scan_val, copy_here[0]);

	cudaFree(gpu_array);
	return 0;

}
