# UVM Hash Table and Skiplist

Copied from [hetero-ds](git.cse.iitk.ac.in:prospar/hetero-ds.git)'s `hashtable`
branch. Written by Vipin.    
Restructured to manage some `#include`s.   


Compilation requires `-std=c++17`. You can use `make` and `make clean`.  
Use the experiment script for performance runs.   
To run the program, you need to run `export TRACE_ROOT=/path/to/traces/`. It is
fine if the traces don't exist, but the environment variable needs to exist for
the current version of `functions.h`. If there are no traces, the program uses
random values.

## Using Traces
Console messages are not very clear for the case when traces haven't been
generated. As mentioned earlier, traces are **not** necessary.  
There is a `driver-tracegen.cu` in the original repo. We'll look at that stuff
later.  

## Flags

Uses two sets of flags: from the original hashtable implementation
(`-ops=10000`, `-add=30`) without spaces, and two-token flags from
`cuda-macros-v1.h` (`-readmostly 1`, `-gohan-static 3`).  

- TODO: Prefetch flags do not work at the moment.

## Behaviour

For `./driver_hashtable_UVM.out -ops=3000000000 -add=70 -rem=10 -rns=1`

Without traces, the program appears to be slow at generating operations, faults
and evictions, probably because of the cost of `random()` (for the find and
deletion parts).
For insertions, the first few seconds see ~1000 evictions per second which is
good. There is ~1 eviction for second in deletions and searches.

Without `-add=70 -rem=10`, almost all accesses are reads, and enabling read
duplication does not change behaviour.

### Execution Time
#### Simple Read Duplication
- `-ops={10M, ... 50M} -rns=10`: 20s to 480s. Allocations about 4 GB

#### RD With True Prefetches
- `-ops=550M -rns=1` - no evictions, 1500s

#### Vanilla (20% and 5%)
- From `perf_run_all.51934`, zero evictions for these data sizes
- Search kernel takes a LOT of time
- `-ops=700000000 -rns=1 -add=20 -rem=5` takes 3000 s
- `-ops=800000000 -rns=1 -add=20 -rem=5` takes 3750 s
- `-ops=900000000 -rns=1 -add=20 -rem=5` takes 4500 s
- `-ops=1000000000 -rns=1 -add=20 -rem=5` takes 4900 s. Breakdown is
```
Total time taken by insert kernel (ms): 541.814
Total time taken by delete kernel (ms): 225607
Total time taken by search kernel (ms): 4.69374e+06
Total time taken by HeteroHash kernel (ms): 4.91989e+06
```

## Skiplists

- From the `skiplist-uvm-memadvise` branch.
- In the `skiplist` directory.


## Errors
- The hashtable needs `libnvidia-ptxjitcompiler.so.1`.
- Pretty sure we had it earlier, but apparently it disappears at times.
- Extract it from the run file.
