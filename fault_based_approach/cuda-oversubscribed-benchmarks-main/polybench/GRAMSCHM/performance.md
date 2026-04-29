pranjal@vindhya:/data/pranjal/cuda-benchmarks/polybench/GRAMSCHM$ ./gramschmidt.exe  -mb 5500; ./gramschmidt.exe  -mb 6000

GPU Runtime: 458.298799s
CPU+GPU: 458.298809 s | Mem 5500 MB

GPU Runtime: 577.766313s
CPU+GPU: 577.766327 s | Mem 6000 MB

-------------

pranjal@vindhya:/data/pranjal/cuda-benchmarks/polybench/GRAMSCHM$ ./gramschmidt.exe  -mb 5500 -prefloc-cpu all; ./gramschmidt.exe  -mb 6000 -prefloc-cpu all 

GPU Runtime: 4831.706996s
CPU+GPU: 4831.707010 s | Mem 5500 MB

GPU Runtime: 9397.519324s
CPU+GPU: 9397.519336 s | Mem 6000 MB

