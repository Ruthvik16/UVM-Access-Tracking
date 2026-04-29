# CUDA UVM Benchmarks

This repository has multiple benchmark suites, from multiple sources and 
with some of our modifications.  

There is no satisfactory suite out there (established + UVM + 
memory-footprint-configurable + no-binaries-loaded), and that is why we 
maintain this. Please keep the code here "clean" and portable.


### Polybench

The official version of Polybench does not use UVM. However, UVMBench's 
authors ported it to UVM.  
We have made some modifications to it (see `./polybench/README.md`).

### RAPIDS

RAPIDS is a set of open source CUDA-accelerated ML and data science libraries.
Many of them use an in-house userspace memory allocator, RMM. By default, most
do not use UVM, but RMM can be configured to use UVM.  
We use dataframes (`pandas`, `cudf`) and graph algorithms (`cugraph`).

### Other Benchmarks

- `UVMBench/bfs` - BFS
- `rodinia/nw`: Needleman-Wunsch Protein Alignment
- `nvidia-samples/sgemm`: matrix multiplication
- Synthetic benchmarks - memset (`int_set_uvm.cu`), random access
  (`random_acc.cu`)
- `rapids/cugraph_sssp.py`: Single-source shortest paths
- `rapids/cugraph_pagerank.py`: the pagerank algorithm
- `rapids/cudf_stdev.py`: Finding the mean and standard deviation of a random
  array.
- `data_structures/bptree`: B-Trees on GPUs
- `data_structures/skiplist`: incomplete

### VLLM

VLLM is a submodule. I used this command (and this source).  
You might need to change the `--branch` and the URL.  

```sh
git submodule add -b uvm_0.16.0rc1  git@git.cse.iitk.ac.in:prospar/vllm.git
```

## Usage

- `make` works in each directory, as well as the root.
- RAPIDS benchmarks need a virtual environment. See `rapids/README.md`.
- Each binary/directory has a `README.md` and `-h` flag.
- For some benchmarks, `README.md` and `performance.md` discuss the behaviour
  and appropriate memory hints.

#### Helpers and Macros: Porting Workloads

- Using allocation macros from `dev/cuda-macros-v1.h` also enables memory hints
  from the CLI.
- Reference (to clean up and port) benchmarks: `polybench/2DCONV`,
  `polybench/MVT`. Do `git diff @~10 @ polybench/2DCONV`.
- Also see `dev/README.md`, `perf_logs/README.md`, all the `-h` flags.
