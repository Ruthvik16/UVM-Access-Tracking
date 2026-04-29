# MVT

MVT has two kernels: $y_1 = M \cdot v$ and $y_2 = M^T \cdot v$.  
Comments in the source describe the performance of both.

### Performance

- For some reason, 10001 MB and 9999 MB are over twice as fast as 10000 MB.
- Not surprisingly, the optimal configurations for both kernels are different.
  The `-optimal` flag captures it for 10 GB.

#### "Optimal" Performance

Runs best with the driver at full prefetching.  
The `-optimal` flag is a bit more nuanced.  

#### Performance for 10 GB

- Vanilla: 250 or so seconds (check)
- FPF - far faster, about 40 seconds
- No PF: moderately faster, about 1.5 times
- Vanilla driver, `-optimal`: 10.8 s
- Full prefetching (1%, not full), `-optimal`: 9.78 s
- KProf's full prefetching and `-optimal`: 10.12 s
- `./mvt.exe -mb 10001 -thread-block 32`: 20.7 s
- `./mvt.exe -mb 10001 -thread-block 16`: 28.3 s
- On the Vanilla driver (not KProf):

```
./mvt.exe -mb 10000 -per-kernel-array       973.306538483 seconds time elapsed
./mvt.exe -mb 9999 -per-kernel-array       281.048785256 seconds time elapsed
./mvt.exe -mb 10001 -per-kernel-array       255.707443887 seconds time elapsed
./mvt.exe -mb 10001 -per-kernel-array -thread-block 16        31.454759930 seconds time elapsed
```

- The small (< 1 MB) 1-D vectors are accessed in each step, but pinning either
  to the CPU or GPU does not speed up execution.
- `./mvt.exe -mb 10001 -optimal -pin-vectors-cpu`: 10.82 s (slower)

