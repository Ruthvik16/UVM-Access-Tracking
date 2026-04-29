# Synthetic Benchmarks

NOTE: Most executables have a `-h` flag, and a handful of CLI options.

### Integer Write

- Sets array elements to 4. Stride can be one int or one page.
- To make it fast for GPGPU-Sim, we don't check the array size. Instead we under-allocate thread blocks, and for small inputs, it is possible that zero are allocated.
- It can be run with an access stride of 4 bytes (`int`), 4 KiB, or 2 MiB. This
  is to (quickly) check for integer casting/overflows in Accel-sim.

### Integer Square

- I overwrote this with integer write for speed.

### CPU Init

- Allocates a CPU array and fills it with 42 or random values.
- CPU init is slow with RD. We'll use this to study CPU initialization.
- Warning: some day `nvcc` might optimize out all of this.
- Begins with a `cudaDeviceSynchronize()` to initialize the driver, which has a
  nontrivial 2 s cost.
- Lot of interesting options, run a `-h`.

### GPU Read
- Read an array from GPU. Write to _some_ of the pages.
- You can tweak the fraction of pages written to using some knobs.

### Mean and Standard Deviation

- Implemented in `cupy` at the moment, in `../rapids/`

### Random Accesses

Copied from Tyler Allen's code artifact, `uvm-eval`. Changes are listed at the
beginning of the file.

----

## Working with GPGPU-Sim
Do `make gpgpusim`.

## TODO
Replace macros with `getopt(3)`. Make it portable


## Compilation

These benchmarks use a lot of macros from `../dev/cuda-macros-v1.h`. To make it
simple/more portable, I am symlink-ing the file here.

You might need to explicitly copy it, but I guess `scp -r` will now work if you
copy only this directory (out of the box). (`git` handles symlinks as symlinks.)
