# iGUARD

iGUARD is a concurrency bug detection tool from Arka's group. The tool uses a
bit of UVM. The benchmarks they tested were:

- Scor is DVM
- CG is about warps, not unified memory
- LonestarGPU is DVM
- SlabHash is DVM
- cuML haven't checked
- Kilo-TM: artifact for BARRACUDA (Race detection). Fully DVM.
- SHoC is DVM  
- CUB has been replaced by CCCL, CUDA Core Compute Library. It has scattered UVM
  usage. It has primitives that have more to do with compute than communicate.
  CCCL has many clients, including cuML and RAPIDS, but doesn't seem to be of
  interest to us.
