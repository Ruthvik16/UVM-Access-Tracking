# UVMBench

This directory has some benchmarks from [UVM Bench](http://arxiv.org/abs/2007.09822).

I ain't no ML or BFS expert, but I am fairly sure this is a blind port from DVM
to UVM, and an expert could do a more faithful task.   
We will probably compare these with `cugraph` and `cuML` implementations of the
same, later.

Benchmarks used:
- BFS - very difficult to chunk up a graph into 10 independent pieces and do
  separate BFSs
- Logistic regression

## Data Sizes
- For BFS, `./main 0  10000000 800000000` creates a lot of evictions and doesn't
  terminate even in 7 hours.
- `bfs/sizes.sh` has some ballpark numbers. In general, BFS with
  oversubscription is slower than CPU BFS.
- I haven't found a way to increase/decrease data size for
  `logistic-regression`. But the running time can be changed with `MAX_ITER` and
  `LEARNING_RATE`.
