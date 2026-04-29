# RAPIDS

RAPIDS is a set of GPU-accelerated data science and ML libraries. Unlike
benchmarks off the internet, these aren't a decade old, and are probably
optimized. I plan to use these:

## `cuDF`
A CUDA-based replacement to `pandas` dataframes. Does not use UVM by default. To
use UVM, add this to the `py` script, or reinitialize RMM (in the RMM
subsection).
```py
import cudf.pandas
cudf.pandas.install()
```

## `cugraph`
Graph algorithms - BFS, Page Rank, etc.


## Installing/Reproducing this Directory
Obviously, I am not committing the virtual environment.  
Use Python 3.10.12. The installation commands are pretty simple - `cugraph`
depends on `cudf`, `numpy`, `pandas`, and the other libraries I want to use.

```sh
# Get venv - DON'T follow the error message
apt install python3.10-venv # BAD!!
apt install python3-venv    # GOOD !! stays in sync with the rest of the world

# Make the venv
python3 -m venv rapids_venv

# The meat - avoids conda and other installations
pip install cugraph-cu12

# not relevant, mentioning for completeness
pip install matplotlib
```

To avoid the download and installation time, `scp` from Vindhya - if the
destination has CUDA 12.
```sh
# (Optional) As a good practice, clone from git.cse. Makes it easier to
# contribute.
git clone git@git.cse.iitk.ac.in:prospar/cuda-oversubscribed-benchmarks.git

# But this doesn't clone the monster libraries, so copy rapids_venv/lib/ with scp
# ask pranjal to chmod 755
scp -r \
        you@vindhya:/data/pranjal/cuda-benchmarks/rapids/rapids_venv/ \
        cuda-oversubscribed-benchmarks/rapids/
        # replace the destination with your download location
```

## RMM
RAPIDS libraries use the RAPIDS Memory Manager (?) which runs in userspace. It
can be configured to use UVM, but no one seems to know how. There are some
pointers in the [RMM README](https://github.com/rapidsai/rmm) -
`managed_memory_resource`.

To enable UVM,
```
import rmm
rmm.reinitialize(
    managed_memory = True,
    pool_allocator = True | False
    )
```
We need to check the performance of both configs. This replaces
`cudf.pandas.install()`.   
The blog below has some `rmm_cfg` stuff, but I haven't been able to make it
work.

## Simple Scripts

Normally, you need to do a `source venv/bin/activate` to run `something.py` that
uses these libraries, which is a bit difficult with scripts.  
Instead, you can add a shebang like this to the files:
```python
#!/data/pranjal/cuda-benchmarks/rapids/rapids_venv/bin/python3

# Now the default interpreter for this file is the venv's python version.
# These imports work out of the box now

import rmm
import cugraph
```

## References
No one seems to know who else uses UVM, or when to use UVM, whether in literature
or NVIDIA technical blogs. However, some good references are

- [RAPIDS](https://rapids.ai/) - GPU accelerated data science
- [Pandas with RAPIDS cuDF](https://developer.nvidia.com/blog/unified-virtual-memory-supercharges-pandas-with-rapids-cudf/)
- [cuGraph](https://developer.nvidia.com/blog/beginners-guide-to-gpu-accelerated-graph-analytics-in-python/) - a sequel to the above blog post
- [An unsuccessful blog on enabling UVM in RMM](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9726-unified-memory-for-data-analytics-and-deep-learning.pdf).
  This claims RMM reserves half of the available memory for its pools. This also
  has a comparison with the PyTorch Caching Allocator (PCA).
- Pool allocations make `cugraph_pagerank.py` slightly slower, but this blog post
  has a good use case: [RAPIDS Memory Manager Pool](https://medium.com/rapids-ai/rapids-memory-manager-pool-speed-up-your-memory-allocations-3bc53929066a)
- Original blog post on RMM's motivation - NVidia Technical Blog -
  [Fast, Flexible Allocation for NVIDIA CUDA with RAPIDS Memory Manager](https://developer.nvidia.com/blog/fast-flexible-allocation-for-cuda-with-rapids-memory-manager/)

## Benchmarks


### Page Rank - NVidia Version
Adapted from Alex Fender's GitHub:
[cugraph_uvm.ipynb](https://gist.github.com/afender/e1968c0f23f9da40edb9f843c46fe5de).
Saved in `cugraph_pagerank.py`.    
To run it multiple times (interactively, on the console), do
```py
from cugraph_pagerank import *
do_pagerank(10000, 50000)
do_pagerank(30000, 70000)

...
```
- At the moment, I've disabled checking the output because it took very very long
  in a run.
- Odd observation: `g1.number_of_vertices()` takes very long, and uses the GPU for
  large graphs. `watch`ing eviction stats proves it.
- Another observation - with the random RMAT graph, the largest page ranks are
  for $2^k$-like vertices

#### Typical Execution Times

Usually, initializing random graphs is _very_ costly (with the `rmat` generator).
These numbers are for `cugraph_pagerank.py`.

#### With Read Duplication
- `8M, 50M`: 5 s
- `33M, 130M`: 620 + 5 s
- `33M, 200M`: 2300 + 6 s
One of these configurations has some evictions.

#### In the Vanilla Driver
- `8M, 50M` takes 3 + 1 s
- `33M, 130M` takes 10 + 4 s
- `33M, 200M` takes 54 + 6 s

### Mean and Standard Deviation
- `cudf_stdev.py` finds the mean and standard deviation of an array.
- RD shows a 30% speedup in it, this is a good proof of concept.
- Invocation: `python3 cudf_stdev.py -rows 10000000 -cols 6`
- The shebang has the absolute path to the virtual environment's copy of python.
  So `./cudf_stdev.py` works.

### Single-Source Shortest Paths

- In `cugraph_sssp.py`. `-h` explains usage.
- Pitfall - generating the random graph is sometimes costlier than SSSP. Pay
  attention to the console.
- Reasonable data sizes - `n = 10M`, `m = 250M-350M` (with and without oversubscription.)

## Profiling
- Profiling these with `ncu` is impractical, even with 1 KB of data.
-  Use `-c 100` to profile the first few kernel
  invocations.
- `--graph-profiling <node|graph>` does not seem to have an effect.
