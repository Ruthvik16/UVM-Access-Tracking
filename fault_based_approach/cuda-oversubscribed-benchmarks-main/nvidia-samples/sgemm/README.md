# Matrix Multiplication

Uses `libcublas`.  
Downloaded from
[https://github.com/NVIDIA/cuda-samples/blob/master/Samples/4_CUDA_Libraries/matrixMulCUBLAS/matrixMulCUBLAS.cpp](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/4_CUDA_Libraries/matrixMulCUBLAS/matrixMulCUBLAS.cpp).

### Behaviour

The program logic is $C = A \cdot B$.  
Broadly speaking, we expect $A$ and $C$ to see row-major accesses, and B to see
column-major accesses.  

The per-fault batch locality is highest in $B$, but multi-batch locality is
lowest in $B$ by about an order.

Because $B$ sees the most "irregular" accesses, it is best to pin B to the GPU
(by pinning the other two to CPU) at data sizes close to thrice the HBM.

### Hints

There are separate flags for `AccessedBy` because the behaviour depends on
when the hint is passed.  
In any case, all flags pass the hint before any data is copied to the GPU
(except for the output matrix $C$), and we do not get the desired
map-on-eviction behaviour.

### Data Size
Permits a lot of oversubscription. Starts timing out at 3x Vindhya, probably because the left
and output matrices can be processed row-by-row.
```
./mm_cublas.out -data 17000000000      74.415538854 seconds time elapsed
./mm_cublas.out -data 17300000000     241.534831769 seconds time elapsed
./mm_cublas.out -data 17600000000    1103.809517780 seconds time elapsed
```
