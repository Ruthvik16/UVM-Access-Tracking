#include "../dev/cuda-macros-v1.h"

/*
	Allocate multiple regions in memory. Check that they're virtually contiguous.
	Then, try running a CUDA kernel with a pointer such that it accesses its neighbour
	regions.
	Then, check if they're operated upon 'accidentally', correctly and if the data
	can be read from CPU to get the expected values

	Here, we see that adjacent VA blocks can be operated upon by a CUDA kernel.
	Either addresses are not virtualized inside the GPU, or the same addresses are used,
	or the virtualization layer has adjacent mappings.
*/

#define PAGES(n) ((n) * 0x1000)

// pointer arithmetic is such that &a + b returns (b * sizeof(a)) added to &a
// so, we use a different macro for pointer arithmetic
#define FLOAT_PAGES(n) ((n) * 0x1000 / sizeof(float))

int main() {
	TRY_ALLOC_NON_UVM_BY_DATA_SIZE(float, i1, PAGES(512));
	TRY_ALLOC_NON_UVM_BY_DATA_SIZE(float, i2, PAGES(512));
	TRY_ALLOC_NON_UVM_BY_DATA_SIZE(float, i3, PAGES(512));
	TRY_ALLOC_NON_UVM_BY_DATA_SIZE(float, i4, PAGES(512));
	TRY_ALLOC_NON_UVM_BY_DATA_SIZE(float, i5, PAGES(512));
	printf("got pointers\n%p\n%p\n%p\n%p\n%p\n", i1, i2, i3, i4, i5);
	if (((long int) i1) & 0x1fffff)
		printf("They're not 2M aligned\n");
	if ((i2 - i1 != FLOAT_PAGES(512)) ||(i3 - i2 != FLOAT_PAGES(512))
			||(i4 - i3 != FLOAT_PAGES(512)) || (i5 - i4 != FLOAT_PAGES(512)))
		printf("They're not virtually contiguous\n");
	else
		printf("They're virtually contiguous\n");

	// initialize 2, 3, 4, 5 to 5.0
	initialize_array_in_gpu<<<(FLOAT_PAGES(512)/1024), 1024>>>(i2, PAGES(512), 5.0);
	initialize_array_in_gpu<<<(FLOAT_PAGES(512)/1024), 1024>>>(i3, PAGES(512), 5.0);
	initialize_array_in_gpu<<<(FLOAT_PAGES(512)/1024), 1024>>>(i4, PAGES(512), 5.0);
	initialize_array_in_gpu<<<(FLOAT_PAGES(512)/1024), 1024>>>(i5, PAGES(512), 5.0);
	CHECK_RETURN_VALUE(cudaDeviceSynchronize());

	// usually, they're adjacent pages. now try setting them all to 6.0
	initialize_array_in_gpu<<<(FLOAT_PAGES(5 * 512)/1024), 1024>>>(i1, PAGES(5 * 512), 6.0);
	CHECK_RETURN_VALUE(cudaDeviceSynchronize());

	float f6[1024];
	CHECK_RETURN_VALUE(cudaMemcpy(f6, i2, 4096, cudaMemcpyDeviceToHost));
	printf("Got values %f %f %f\n", f6[0], f6[10], f6[1023]);

	CHECK_RETURN_VALUE(cudaMemcpy(f6, i3 + FLOAT_PAGES(11), 4096, cudaMemcpyDeviceToHost));
	printf("Got values %f %f %f\n", f6[0], f6[10], f6[1023]);

	CHECK_RETURN_VALUE(cudaMemcpy(f6, i4 + FLOAT_PAGES(511), 4096, cudaMemcpyDeviceToHost));
	printf("Got values %f %f %f\n", f6[0], f6[10], f6[1023]);

	CHECK_RETURN_VALUE(cudaMemcpy(f6, i5 + FLOAT_PAGES(311), 4096, cudaMemcpyDeviceToHost));
	printf("Got values %f %f %f\n", f6[0], f6[10], f6[1023]);
	return 0;
}

