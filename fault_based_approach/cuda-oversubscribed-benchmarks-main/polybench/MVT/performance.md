# Performance

- Aggressive prefetching fills holes for _some_ threads in each warp.
- 16/32 threads is far better than the default of 256.
- 10000 MB is for some reason 2.5 times slower than 10001 and 9999. Not so with
  `uvm_perf_prefetch_enable=0`.

```
./mvt.exe  -mb 10001 -thread-block 32: 22 s
./mvt.exe  -mb 10001 -thread-block 32 -kernel2-accby: 12 s
./mvt.exe  -mb 10001 -thread-block 32 -accessed-by-gpu: 100 s
```

### Driver Configurations

```
[ 1 RUNS, kprof_fpf_fpf2274 ]	  ./mvt.exe -mb 10000 -per-kernel-array        67.788367633 seconds time elapsed
[ 1 RUNS, kprof_fpf_fpf2274 ]	  ./mvt.exe -mb 9999 -per-kernel-array        53.338783588 seconds time elapsed
[ 1 RUNS, kprof_fpf_fpf2274 ]	  ./mvt.exe -mb 10001 -per-kernel-array        63.623710556 seconds time elapsed
[ 1 RUNS, kprof_fpf_fpf2274 ]	  ./mvt.exe -mb 10001 -per-kernel-array -thread-block 16        25.257619636 seconds time elapsed

[ 1 RUNS, kprof_npf2274 ]	  ./mvt.exe -mb 10000 -per-kernel-array       692.199803700 seconds time elapsed
[ 1 RUNS, kprof_npf2274 ]	  ./mvt.exe -mb 9999 -per-kernel-array       664.752333099 seconds time elapsed
[ 1 RUNS, kprof_npf2274 ]	  ./mvt.exe -mb 10001 -per-kernel-array       689.635032438 seconds time elapsed
[ 1 RUNS, kprof_npf2274 ]	  ./mvt.exe -mb 10001 -per-kernel-array -thread-block 16        91.999530677 seconds time elapsed

[ 1 RUNS, kprof_remote_tbn2274 ]	  ./mvt.exe -mb 10000 -per-kernel-array       111.571284542 seconds time elapsed
[ 1 RUNS, kprof_remote_tbn2274 ]	  ./mvt.exe -mb 9999 -per-kernel-array       112.053296824 seconds time elapsed
[ 1 RUNS, kprof_remote_tbn2274 ]	  ./mvt.exe -mb 10001 -per-kernel-array       111.763509541 seconds time elapsed
[ 1 RUNS, kprof_remote_tbn2274 ]	  ./mvt.exe -mb 10001 -per-kernel-array -thread-block 16        40.179290874 seconds time elapsed

[ 1 RUNS, kprof_tbn5587 ]	  ./mvt.exe -mb 10000 -per-kernel-array      1176.544366920 seconds time elapsed
[ 1 RUNS, kprof_tbn5587 ]	  ./mvt.exe -mb 9999 -per-kernel-array       448.140876714 seconds time elapsed
[ 1 RUNS, kprof_tbn5587 ]	  ./mvt.exe -mb 10001 -per-kernel-array       412.360612579 seconds time elapsed
[ 1 RUNS, kprof_tbn5587 ]	  ./mvt.exe -mb 10001 -per-kernel-array -thread-block 16        35.068870702 seconds time elapsed

[ 1 RUNS, vanilla5587 ]	  ./mvt.exe -mb 10000 -per-kernel-array      1007.386055776 seconds time elapsed
[ 1 RUNS, vanilla5587 ]	  ./mvt.exe -mb 9999 -per-kernel-array       283.014426991 seconds time elapsed
[ 1 RUNS, vanilla5587 ]	  ./mvt.exe -mb 10001 -per-kernel-array       274.377231649 seconds time elapsed
[ 1 RUNS, vanilla5587 ]	  ./mvt.exe -mb 10001 -per-kernel-array -thread-block 16        31.608909901 seconds time elapsed
```

### Gohan

```
./mvt.exe  -mb 10001 -gohan4 all                20.74
./mvt.exe  -mb 10001 -gohan2 all                16.98
./mvt.exe  -mb 10001 -gohan-fpf-rdup all        21.34
```
