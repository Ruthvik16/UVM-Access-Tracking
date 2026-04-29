# Needleman-Wunsch Protein Alignment

Note: This program generate two sequences randomly. Please specify your own sequences for different uses.  
      At the current stage, the program only supports two sequences with the same lengh, which can be divided by 16.  
Usage:
```sh
./needle <data_size_bytes>
./needle -mb <data_size_mb>
# or
./needle -h
```

### Adjustable work group size

- `RD_WG_SIZE_0` or `RD_WG_SIZE_0_0`
- This cannot be made a CLI argument because the kernels use shared static
  memory. Instead, `make` compiles both the default and optimal configurations.

USAGE:
```
make clean
make KERNEL_DIM="-DRD_WG_SIZE_0=16"
```

- The block size needs to be a compile-time constant, because of a `__shared__`
  memory allocation.
- `needle_kernel.cu` is `#include`d twice in `needle.cu` with two `BLOCK_SIZE`s,
  16 and 64. There are two pairs of kernels, whose names are suffixed with `_16`
  and `_64` by the preprocessor.
- The latter pair runs faster. But, the shared memory does not support larger thread blocks.

For some reason, the stats for the two VA ranges are nearly identical.

### Behaviour
- Under Mickey, there are _less_ GPU faults under zero prefetching, at 10 GB.
- Statistics for the two buffers/UVM ranges are eerily similar, but the fault
  locality is very different.

### Sources/Credits

- Ported to UVM by the UVMBench authors.
- Originally from the Rodinia Suite.
