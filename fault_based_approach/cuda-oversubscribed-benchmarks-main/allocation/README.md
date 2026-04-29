# UVM and non-UVM Allocation Semantics


### Allocation Granularity - UVM

- `uvm-allocation-granularity-2M.cu` examines the allocation granularity for UVM allocations. From x86 semantics, we know that allocating one byte allocates at least 4 KiB. The question is what it is for UVM allocations.
- `./a.out A B` accesses (2^A + B) bytes of memory beginning at a pointer returned by `cudaMallocManaged`.
- Run `./a.out 21 0`, `./a.out 21 3`, `./a.out 21 4` and `./a.out 22 0`. The third and fourth configurations access the next 2M huge page, and result in errors. (1,2,3 become 0 on division by sizeof(float)).
- Additionally, after filling the 2M VA block, the same value can be read from CPU as well even if 1 Byte was allocated initially (without segfault).
- This shows that 2 MiB is allocated in one go. We did not verify that the pointer returned is 2M-aligned, but that was observed in CS614.

### Allocation Granularity - non-UVM

- Errors occur in the same cases in `non-uvm-allocation-granularity-2M.cu`, at 2 MiB boundaries.
- Additionally, `cudaMemcpy(X)` fails if allocation size < `X`. So, we could not check if the values filled beyond the allocated region can be read from CPU.
- It appears that the userspace runtime keeps track of allocations (so that a 2M region can be shared by multiple allocations, probably). Use the macros to verify this.
- TLDR: non-UVM allocations also have a 2 MiB granularity.

### Addressing inside GPU - Kernels can 'accidentally' access an adjacent region

- `access-adjacent-cudamalloc-regions.cu` allocates 5 regions of 2M each. Their (device) pointers are virtually contiguous.
- After calling a kernel with the first region as pointer, which is asked to write to 10M at that address, data is read in CPU.
- We find that data in all pages is overwritten.
