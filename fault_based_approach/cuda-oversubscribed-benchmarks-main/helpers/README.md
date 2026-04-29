## `cudamalloc-sleep`
Can be used to
- initialize a token cuda process and keep the kernelspace framework alive (saves 2s, gives better results)
- fill the gpu for artificial oversubscription
- access using aliases beginbgcp, killbgcp for token background process

## `uvm-thrashing`
- prints cuda kernel execution time from clock()

## `gpu_fault_noise.cu`
- To be run in the background
- Will soon check if it matters

## `gpu_scheduling_noise.cu`
- To be run in the background
- Will soon check if matters
