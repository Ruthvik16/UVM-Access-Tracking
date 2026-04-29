# BFS CUDA

## Description: 
Different approaches for implementation of bfs on GPU with CUDA Driver API.   
Source: [https://github.com/OSU-STARLAB/UVM_benchmark][UVMBench].   

## Algorithms:
- O(V + E) sequential bfs
- O(V^2 + E) parallel simple bfs
- O(V + E) parallel queue with atomic operations bfs (slow)
- O(V + E) parallel queue with scan bfs

## Usage:
```
To build the project run:
make

To run algorithms on random generated graphs:
./bfs <start vertex> <number of vertices> <number of edges>

To run algorithm on graphs from standard input:
./bfs <start vertex> < start input in next line ... >
Input should be in the form:
<number of vertices> <number of edges>
<end of edge1> <end of edge1>
<end of edge2> <end of edge2>
...

```

Input can be generated using `graphgen` in UVMBench's `data/bfs/inputGen` directory.  
`../README.md` has some suggestions/details common to BFS and logistic
regression.

### Thread Block Size
By default, thread blocks have 1024 threads. This can be changed by:
- compiling with `-DTHREAD_BLOCK_SIZE=x`
- The CLI flag `-thread-block` does not yet work perfectly.

## Links:
Accelerating large graph algorithms on the GPU using CUDA, Pawan Harish and P. J. Narayanan
https://pdfs.semanticscholar.org/4c77/e5650e2328390995f3219ec44a4efd803b84.pdf

Scalable GPU Graph Traversal, Duane Merrill, Michael Garland, Andrew Grimshaw
http://research.nvidia.com/sites/default/files/pubs/2012-02_Scalable-GPU-Graph/ppo213s-merrill.pdf

There is also an O(V + E) algorithm described below that uses hierarchical queues and works efficiently 
with shared memory but it needs to convert graph into a near regular-graph before running the kernels.  

An Effective GPU Implementation of Breadth-First Search, Lijuan Luo, Martin Wong, Wen-mei Hwu  
http://impact.crhc.illinois.edu/shared/papers/effective2010.pdf

## Data Structures Used
Graph initialization is SLOW. Often far slower than BFS itself.   

For CPU BFS, the adjacency lists are initialized in `graph.cpp` as
`std::vector<std::vector<int>> adjacencyLists(n);`. The outer vector isn't
resized, but replacing it with an array-of-vectors probably won't save much.    

I haven't looked at the GPU part, but I know that the original paper from 2007/8
concatenated all adjacency lists into a single array and had a separate array of
length `n` telling where each list begins/ends. So, somewhere in between,
UVMBench probably converts the first to the second, and we can expect that to be
costly.

## Reasonable Data Sizes

On Vindhya (6 GiB), oversubscription starts at `n = 10M`, `m = 725M`.  
CPU initialization can take a few minutes because of the C++ magic.
- `n = 20M`, `m = 1400M` takes about 5 min.
- `m = 1460M` takes 30 min.
- `m = 1480M` takes 140 min.

## Behaviour Under Oversubscription

For 10M vertices and up to 2000M edges,
the time the driver spends copying data is well under 3% of the wall clock time.
On average, a block is copied back and forth five times in the program lifetime.

This benchmarks appears to be bottlenecked on computation, not thrashing or
communication, which might change at higher oversubscription levels.

This is difficult to check because (1) it is already slow and (2) it is even
slower with remote mappings.

## Code Flow
This benchmark is somewhat complicated.

- `main.cu` has `int main` and the driver for CPU and the three GPU BFSs.
- All files/algorithms are in `bfsCuda.cu[h]`, `bfsCPU.*`.
- The `Graph` class has `n`, `m`, etc. The actual graph is stored in global
  pointers at the top of `main.cu`. (`u_adj`, `u_edgesOffset`, `u_*`)
- Data structure: all adjacency lists are concatenated in `u_adjacencyList`. The offsets within
  the list are in `u_edgesOffset`.
- Number of edges is `u_edgesSize[i]`
- Notation: depth of graph is `d`.


#### Simple BFS
This is textbook BFS. `simpleBfs()` is invoked `d` times. Each invocation has `n` threads.
Thread `i`, for the `i`th vertex, runs only in the `d(i)`-th invocation, i.e. that vertex's
depth.
- Input: the usual
- Output/modified: `u_distance[i]`, `u_parent[i]`, `*changed`


#### Queue BFS

#### Scan BFS
- Instead of launching `n` threads for each level, find the number of vertices with `depth == i`
and launch `queueSize` threads.  
- `nextLayer()` processes the current set (periphery/cut) of the graph.
- `countDegrees()` finds the number of neighbours introduced by each vertex. Not
  clear why it isn't a part of `nextLayer()`.
- `scanDegrees()` sums the number of such vertices. It is hardcoded to use 1024
  threads, and gives the `queueSize` for the next iteration.

All implementations modify *only* `u_distance[]` and `u_parent[]`. Thus, the other arrays are
good candidates for the `readMostly` hint.

## Possible Optimizations
- Merge `countDegrees()` and `nextLayer()`
- Instead of initializing adjacency lists through C++ vectors, directly fill the
  UVM array with random numbers modulo `n`
