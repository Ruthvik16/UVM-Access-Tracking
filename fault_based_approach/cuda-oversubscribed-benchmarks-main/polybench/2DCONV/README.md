# 2 Dimensional Convolution

#### Output Correctness

With non-random data, the GPU output is incorrect for some reason (`-compare`).  
This combination results in zero errors: `-compare -random-data`.  

#### Optimal Parameters:

With one phase (the default):

- `./2DConvolution.exe -mb 30000 -accessed-by-gpu`: 18.9549 s
- `./2DConvolution.exe -mb 30000`: 20.8786 s
- `./2DConvolution.exe -mb 30000 -pin-cpu`: 28.521 s

With three phases, there is a good deal of reuse:

- `./2DConvolution.exe -mb 30000 -phases 3 -accessed-by-gpu`: 22.08 s
- `./2DConvolution.exe -mb 30000 -phases 3`: 36.5091 s
- `./2DConvolution.exe -mb 30000 -phases 3 -pin-cpu`: 31.7327 s

With a smaller data size, there is less eviction-reuse, and less of a
difference.

- `./2DConvolution.exe -mb 10000 -accessed-by-gpu`: 7.78130 s
- `./2DConvolution.exe -mb 10000`: 7.845 s
- `./2DConvolution.exe -mb 10000 -pin-cpu`: 10.56806 s

TODO: test with the prefetch-remote-mappings driver configuration.
