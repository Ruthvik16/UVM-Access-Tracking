/*
 * mem_blocker.cu — hold a fixed amount of GPU memory via cudaMalloc (non-UVM)
 * to simulate reduced free GPU memory for oversubscription benchmarks.
 *
 * Usage: ./mem_blocker.out <bytes>
 *   Allocates <bytes> bytes, prints "READY\n", then holds until SIGTERM/SIGINT.
 *   The benchmark driver reads "READY" before launching the workload, then
 *   sends SIGTERM to release memory after the workload finishes.
 */
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <cuda_runtime.h>

static volatile int g_running = 1;
static void on_signal(int s) { (void)s; g_running = 0; }

/* Touch all pages so they are physically resident on the GPU */
__global__ void touch_pages(char *buf, long n) {
    long i      = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)blockDim.x * gridDim.x;
    for (; i < n; i += stride)
        buf[i] = (char)(i & 0xff);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: mem_blocker.out <bytes>\n");
        return 1;
    }

    long bytes = atol(argv[1]);
    char *buf  = NULL;

    cudaError_t err = cudaMalloc((void **)&buf, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "mem_blocker: cudaMalloc(%ld MB) failed: %s\n",
                bytes >> 20, cudaGetErrorString(err));
        return 1;
    }

    /* Touch pages in blocks of 256 threads; cap grid at 65535 */
    int  bsz     = 256;
    long nblocks  = bytes / bsz;
    if (nblocks > 65535L) nblocks = 65535L;
    if (nblocks < 1L)     nblocks = 1L;
    touch_pages<<<(int)nblocks, bsz>>>(buf, bytes);
    cudaDeviceSynchronize();

    signal(SIGTERM, on_signal);
    signal(SIGINT,  on_signal);

    /* Signal Python driver that memory is held */
    printf("READY\n");
    fflush(stdout);

    while (g_running)
        sleep(1);

    cudaFree(buf);
    return 0;
}
