#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

using namespace std;

/*
   allocate a UVM memory chunk. Initialize in CPU, modify in GPU and then read again in
   CPU.
   Measure time taken in full operation to check if oversubscription is being used.
*/

__global__ void vector_add(float *a, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
      	a[idx] += idx;
    }
}


int main(int argc, char* argv[]) {
    long int size;
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <array_size> [num_loops=5]\n", argv[0]);
        return 1;
    }
    size = atol(argv[1]);
    int num_loops = argv[2] ? atoi(argv[2]) : 5;
    size /= sizeof(float);
    float *a;
    cudaError_t ret;
    
    bool force_sync = true;

    for (int k = 2; k < argc; k++) {
	    if ((strcmp(argv[k], "--dont-sync-kernels") == 0) || (strcmp(argv[k], "-d") == 0)) {
		    force_sync = false;
		    printf("Not synchronising GPU kernel calls with CPU\n");
	     }
    }

    ret = cudaMallocManaged(&a, size * sizeof(float));

    if (ret != cudaSuccess) {
        printf("cudaMallocManaged error\n");
	return 1;
    }
    for (int i = 0; i < size; i++) {
        a[i] = 0.0f;
    }
    int bSize = 256;
    // kind of like ceil
    int numBlocks = ((size) + bSize - 1) / bSize;

    // GPU faults - looping/thrashing access pattern
    clock_t start, end;

    start = clock();
    for (int k = 0; k < num_loops; k++) {
	    vector_add<<<numBlocks, bSize>>>(a, size);
	    ret = force_sync ? cudaDeviceSynchronize() : cudaSuccess;
	    if (ret != cudaSuccess) {
		    printf("Error in cudaDeviceSynchronize() iteration %d\n", k + 1);
	    }
    }
    if (!force_sync) {
	    ret = cudaDeviceSynchronize();
	    if (ret != cudaSuccess) {
		    printf("Error in cudaDeviceSynchronize() iteration %d\n", num_loops);
	    }
    }

    // CPU faults
    for (int i = 0; i < size; i++) {
	    if (a[i] < -1)
    		printf ("Error at index %d\n", i);
    }
    end = clock();

    printf ("Program exiting | Data size %ld MB | %d loops over data |time taken in operation %lf ms\n",
		sizeof(float)*size >> 20, num_loops, ((double) (end - start))/CLOCKS_PER_SEC * 1000);
    return 0;
}
