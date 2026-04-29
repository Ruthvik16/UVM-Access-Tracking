This directory contains header file(s) with macros to make CUDA programming easier.

Do NOT modify the headers! We need all source files to build and run smoothly for eternity.
You can _add_ to them, however, but make sure to keep functions and variable names coherent.

We will decide later and add comments to either the `*.h` files or in markdown.

### Hints

There is a "generic" framework to insert hints in UVM buffers at the **time of
allocation**.  
Check `polybench/2DCONV/*.cu` for the usage of `get_hints`.  
Each ported binary has a `-h` flag, and _prints_ the hints inserted if any.  

This is one of the parts of the headers that uses carelessly declared global
variables. These hints work only with the allocation macros defined in this
file.

#### First Touch Behaviour:

- `./cpu_init.out -mb 10000 -initcpu 1`: 6.03 s
- `./cpu_init.out -mb 10000 -prefloc-cpu 1`: 6.26 s

#### Disabling Mickey

`export MICKEY_DISABLE=1`

#### Porting Benchmarks

- Use the allocation and hint macros in this file.
- Use 64 bit integers for array sizes and indices (we've had enough 50 MB
  simulations, welcome to the driver world).
- Use **signed** integers for the above to detect overflows and underflows.
- Between initialization, computation, and reading the results:

```C
    init_array_with_random(...);
#ifdef CUDA_CLI_HINTS
    HINTS_POST_INIT(...);
    HINTS_POST_INIT(...);
#endif

    compute<<<XX>>>(YY);

    // Remember to synchronize! the kernel returns asynchronously.
    CHECK_RETURN_VALUE(cudaDeviceSynchronize());

#ifdef CUDA_CLI_HINTS
    HINTS_POST_COMPUTE(...);
    // You can skip the non-output buffers here.
#endif
```

- Remember to time each step - those numbers are useful to _analyze_
  optimizations. Including parsing the output.
- GPU and systems researchers speak gigabytes, not array dimensions. Add an
  automatic `-data` and `-mb` flag to find the array dimensions from the
  specified WSS (working set size). See 2DCONV.
