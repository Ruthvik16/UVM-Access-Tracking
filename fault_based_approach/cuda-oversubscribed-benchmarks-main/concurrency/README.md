# Concurrency in Memory Transfers and GPU Computation

### GPU Computation

- `slow-cuda-kernel.cu` runs a compute-intensive process, incrementing an integer/long array 2^N times.
- Roughly, we observe that running two processes simultaneously takes twice the time.
- Need to decide on the right way to get more of this dat. beacuse `clock()` roughly matches `time`'s user time, and clock time matches real time reported. Difference is <2s in all cases.
- Time returned by `$ time` and `clock()` doesn't match clock time for two threads.

- Approximate execution time for
- `./a.out 8 34` - 83s
- `./a.out 8 34 & ./a.out 8 34` - 173s
- `./a.out 8 34 -thread` - 166s and 250
- `./a.out 8 34 -fork` - 173.33s and 173.34
- `./a.out 8 34 -two-kernels-async` - 167s

- There is some relevant documentation, but it doesn't answer whether two kernels from the same process/thread can run concurrently.
- Haven't checked if two streams can run concurrently.

