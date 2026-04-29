# Polybench

This directory contains benchmarks from the Polybench suite. These have been ported 
to UVM by the UVMBench authors. UVMBench also has non-UVM versions, which 
we haven't included here.  


For internal reference: the older/original non-UVM version of polybench has been 
published. UVMBench has NOT been published. It is on ArXiv currently.

## Usage (TLDR)

- Do `./whatever.exe [optional data size]`
- For a summary of command-line options, do `./whatever.exe -h[elp]`
- Helper: `find_kexec_time.sh` runs all ported benchmarks thrice to evaluate a modification in one go.

## Our Changes to Polybench/UVMBench

The working set size can be configured for all benchmarks.  
All changes haven't been made to all benchmarks. Currently, these are up to date:

- 2MM
- 2DCONV
- 3MM
- 3DCONV
- ATAX
- BICG
- MVT
- CORR
- GRAMSCHM

We have made the following changes (defaults configurable in `./polybenchUtilFuncts.h`):

- Application footprints are hardcoded in the source. Now, it is the first command-line argument (in Bytes).
- Comparing results with CPU computation can take very long (ex 2MM has super-linear execution time). Use `-compare-with-cpu` to enable it.
- Results of GPU computation are not copied back to the CPU by default. To enable it, do `-copy-back`
- The last line of output has the GPU-only and GPU+CPU-touch run times.

### Benchmark-specific Changes

(Tidbits)
- 2DCONV has a `-phases 10` flag now, to alternate between read-only and write-only
  behaviour for the two arrays
- MVT has a `-read-mostly-hint` flag to test CUDA's memory hints.

### Plans/TODO

- The last line of output should have GPU-only and CPU-GPU run time, to make performance runs easier than the current `sed` gymnastics.
- Touching pages to effect D2H migrations has a baseline cost even when pages are resident on CPU. Find it and compare with the migration time/cost.
- Benchmarks can time out in long performance runs. Add a `SIGKILL` handler that
  prints `Timed out or killed` to the console.


## Helpers

- Testing changes to the suite: `./compileCodes.sh`
- Cleaning up: `rm */*exe`
- Run all files: `./testCodes.sh [arguments passed to binaries]`

## About the Benchmarks

Vague details are available at [this](https://www.cs.colostate.edu/~pouchet/software/polybench/) webpage.   
Original Polybench is cleaner than the UVMBench version, but that too has shabby
loose ends, like initializing an array which is _not_ an input parameter (in
2MM).

### ATAX
ATAX computes $y = A^T A x$, given a 2D matrix $A$ (of size $NX \times NY$) and a vector $x$.
In the CPU version, the intermediate result $Ax$ is stored in `tmp[]`. It
accesses each row of $A$ twice, in order.   

In the GPU version, `atax_kernel<<<NX/256, 256>>>()` is launched, followed by a
synchronization call, and then a call to `atax_kernel2<<<NY/256, 256>>>()`. The
first kernel computes `tmp[*]`, or $A \dot x$, with an access stride of one
`float`. (Over 1000 warps are launched simultaneously, so it is possible for the
actual access pattern to be random.)

The second kernel computes $y = A^T \dot tmp$. This has an access stride of
$NY$, which is more than 100 kB for a workload of a few GB. (The stride for a
warp of 32 can thus run into tens of megabytes.)
The nested loop can be inverted to make the accesses contiguous, with
extensive synchronization or use of atomic add primitives.


### MVT

Computes $y_1 = A \dot x$ and $y_2 = A^T \dot x$, for a 1D vector $x$.

### BICG

BICG sub-kernel of the BiCGSTAB Linear solver.

### 3DCONV


### 2MM - Matrix Multiplication
This benchmark computes $C = A \dot B$, and then $E = C \dot D$. This has 32 * 8
sized thread blocks by default. As there are 5 matrices, actual oversubscription
requires a data size of over 30 GB for our 6GB GPU.   
The GPU version calls `mm2_kernel1` with `NI * NJ` threads and `mm2_kernel2`
with `NI * NL` threads. Each thread computes one entry in `C[]` or `E[]`.   

In both CPU and GPU versions, the access stride for `B` and `D` is the size of a
row, and is 4 bytes for `A` and `C`.
