# Candidates for UVM Benchmarks

This file lists codebases we considered adding to our benchmark suite.

## LoneStar and Galois
Lonestar is a benchmark suite, and a part of the [Galois](https://github.com/IntelligentSoftwareSystems/Galois)
concurrency library. It is maintained by Keshav Pingali's group, and appears to
be a good candidate for thread-safe STL, such as parallelizing GPGPU-Sim.

This uses mostly DVM (discrete virtual memory or `cudaMalloc()`) allocations.
UVM is used for a few flag variables.

## BlazingSQL

Blazing SQL is a GPU-based database engine, and it also uses UVM. However, it is
not actively maintained and has been difficult to set up. I think it can be set
up with some effort though. (The original authors abandoned this project for a
proprietary startup version.


## Neural Network Implementation

[This](https://github.com/BobMcDear/neural-network-cuda) is a popular repository
with a simple implementation of NN training and inference. It is "educational",
and not optimized.

## K-Dimensional Tree

k-D trees are used in computational geometry and databases. A UVM implementation
is available at [https://github.com/ingowald/cudaKDTree](https://github.com/ingowald/cudaKDTree).

## Phased Programs
- Synthetic program ideas: multiple passes over an array, with increasing r/w propensity
- 2DCONV extension: current is `A -> B`. Let us make it:
- Already done for 2DCONV, we can do it for the rest.

```C
void 2DCONV(float *A, float *B) {
    // read A
    // write B
    return;
}

driver() {
    P = float_array_init();
    Q = cudaMallocManaged();

    for (i < N) {
        2DCONV(P, Q);
        2DCONV(Q, P);
    }
}
```

## Database: HRocks
[https://www.csa.iisc.ac.in/~arkapravab/papers/SIGMOD25_HRocks.pdf](paper)  
Needs `libpmem` to be installed from apt.
