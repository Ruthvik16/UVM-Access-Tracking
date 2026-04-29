### Data Size(s)

There are three arrays: (1) left multiplicand, (2) right multiplicand
(irregular), and (3) output.  
Thrashes at about 3x HBM size:

```
./mm_cublas.out -data 17000000000      74.415538854 seconds time elapsed
./mm_cublas.out -data 17300000000     241.534831769 seconds time elapsed
./mm_cublas.out -data 17600000000    1103.809517780 seconds time elapsed
```

The left and output arrays do not seem to "matter" much to performance - the
eviction count is about half of the number of blocks, meaning they are accessed
in one go.   
So, pinning to the host does not produce a dramatic effect, beyond perhaps
causing less thrashing in the right multiplicand and the white elephant, `u_B`.  

### Hints

- `./mm_cublas.out -mb 7500 -accessed-by`: 140 s

- AccessedBy needs to be inserted **after** array initialization. In any case,
  that does not change it to an remote-map-on-evict semantic.
- Pinning A and C to the host is best for 15 GB and 17.5 GB.
- Read duplication does not seem to help.
- `-accby-gpu 3` seems to degenerate to `-prefloc-cpu 3`.

```
[ 1 RUNS, kprof ]	  ./mm_cublas.out -mb 4000 -acc-by-b        47.153838885 seconds time elapsed
[ 1 RUNS, kprof ]	  ./mm_cublas.out -mb 7000 -acc-by-b       102.668575199 seconds time elapsed
[ 1 RUNS, kprof ]	  ./mm_cublas.out -mb 12000 -acc-by-b       234.067272571 seconds time elapsed
[ 1 RUNS, kprof ]	  ./mm_cublas.out -mb 17300 -acc-by-b       411.994911964 seconds time elapsed
[ 1 RUNS, kprof ]	  ./mm_cublas.out -mb 17500 -acc-by-b       413.446681240 seconds time elapsed
```

At the below data sizes, three, two, one, and zero (respectively) of the
matrices fit on the HBM.

With the profiler, (`*.561161`):  

```
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 5800         8.889935169 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 5800 -accby-gpu 1        47.697439819 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 5800 -accby-gpu 1,3        47.744675040 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 5800 -accby-gpu all        93.603301102 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 5800 -prefloc-cpu 1,3        48.816419248 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 5800 -accby-gpu 2        39.790655998 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 5800 -accby-gpu 1,2        47.689619256 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 5800 -accby-gpu 2,3        39.755903469 seconds time elapsed


[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 8500        15.492744493 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 8500 -accby-gpu 1       127.784843516 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 8500 -accby-gpu 1,3       127.710195138 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 8500 -accby-gpu all       260.793202220 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 8500 -prefloc-cpu 1,3       128.561744148 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 8500 -accby-gpu 2       144.525075941 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 8500 -accby-gpu 1,2       127.710552071 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 8500 -accby-gpu 2,3       144.541846107 seconds time elapsed
./sgemm.out  -mb 8500  -accby-gpu 3  -prefloc-cpu 3                 14.11 s
./sgemm.out  -mb 8500  -accby-gpu 3  -prefloc-cpu 3 -pfcpu 1,2 -rm-late 1,2 -prefetch 1,2       13.90 s
./sgemm.out  -mb 8500  -accby-gpu 3  -prefloc-cpu 3 -pfcpu 1,2,3 -rm-late 1,2 -prefetch 1,2     13.62 s

./sgemm.out -mb 12000 -accby-gpu all -prefetch all                      332.6 s

[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 15000        40.576023534 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 15000 -accby-gpu 1       314.623754969 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 15000 -accby-gpu 1,3       314.651240903 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 15000 -accby-gpu all       637.863338978 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 15000 -prefloc-cpu 1,3       315.191117374 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 15000 -accby-gpu 2       332.703329328 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 15000 -accby-gpu 1,2       314.616335256 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 15000 -accby-gpu 2,3       332.838294954 seconds time elapsed


[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 17500       890.787360886 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 17500 -accby-gpu 1       383.951883747 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 17500 -accby-gpu 1,3       383.837902586 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 17500 -accby-gpu all       783.978870144 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 17500 -prefloc-cpu 1,3       437.388142817 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 17500 -accby-gpu 2       418.745655757 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 17500 -accby-gpu 1,2       383.495104629 seconds time elapsed
[ 1 RUNS, kprof_run ]	  ./sgemm.out -mb 17500 -accby-gpu 2,3       418.748671901 seconds time elapsed

./sgemm.out  -mb 17500 -pfcpu 1,2,3 -accby-gpu 3 -rm-late 1,2       923.40 s
./sgemm.out  -mb 17500 -accby-gpu 3 -pfcpu 3                        119 s
./sgemm.out  -mb 17500 -gohan2  3                                   200 s
./sgemm.out  -mb 17500 -accby-gpu 3 -pfcpu 3 -gohan-fpf-rdup 1,2    110 s
./sgemm.out  -mb 17500 -accby-gpu 3 -pfcpu 3 -gohan-fpf-rdup 2 -gohan2 1    85 s
./sgemm.out  -mb 17500 -accby-gpu 3 -pfcpu 3 -gohan-fpf-rdup 1 -gohan2 2    205 s
./sgemm.out  -mb 17500 -accby-gpu 3 -pfcpu 3 -gohan-fpf-rdup 2 -gohan4  1   111 s
./sgemm.out -mb 17500 -accby-gpu 3 -gohan2 1 -pfcpu 1,2 -gohan-fpf-rdup 2   391 s

# Static, global hints
./sgemm.out -mb 17500 -accby-gpu all       783.969738325 s
./sgemm.out -mb 17500 -prefloc-cpu all       775.600458240 s
./sgemm.out -mb 17500 -prefloc-cpu all -accby-gpu all       781.773482921 s
./sgemm.out -mb 17500 -prefloc-cpu all -accby-gpu all -prefetch all       775.152361060 s
./sgemm.out -mb 18000 -accby-gpu all       868.657556966 s
./sgemm.out -mb 18000 -prefloc-cpu all       860.915629924 s
./sgemm.out -mb 18000 -prefloc-cpu all -accby-gpu all       867.526271766 s
./sgemm.out -mb 18000 -prefloc-cpu all -accby-gpu all -prefetch all       872.720001180 s

./sgemm.out -mb 18000                                               1706 s
./sgemm.out -mb 18000 -prefloc-cpu 1                                1722 s
./sgemm.out -mb 18000 -prefloc-cpu 3                                1535 s
./sgemm.out -mb 18000 -accby-gpu 1,2,3 -prefetch 2 -initgpu 3        643 s (with spongeBob lazy mappings)
./sgemm.out -mb 18000 -accby-gpu 1,2,3 -prefetch 2 -initgpu 3        643 s
./sgemm.out -mb 18000 -accby-gpu 1,2,3 -prefetch 2 -prefloc-cpu 3 -rm-late 1,2 1016 s
./sgemm.out -mb 18000 -accby-gpu 1,2,3 -prefetch 2                -rm-late 2   1188 s
./sgemm.out -mb 18000 -accby-gpu 1,3 -prefetch 2 -prefloc-cpu 3         1290 s
./sgemm.out -mb 18000 -accby-gpu 1,3 -prefetch 2                        1664 s
./sgemm.out -mb 18000 -accby-gpu 1,2,3 -prefetch 2 -pfcpu 1,2,3         412.7 s
./sgemm.out -mb 18000 -accby-gpu 2                                      435 s

# Is ARIADNE really that intelligent?
./sgemm.out  -mb 18000 -pfcpu 3 -accby-gpu 3 -pin-mb-b 1000             90 s
./sgemm.out  -mb 18000 -pfcpu 3 -accby-gpu 3 -pin-mb-b 800              77 s
./sgemm.out  -mb 18000 -pfcpu 3 -accby-gpu 3 -pin-mb-b 600              67.55 s
./sgemm.out  -mb 18000 -pfcpu 3 -accby-gpu 3 -pin-mb-b 500              62.6 s
./sgemm.out  -mb 18000 -pfcpu 3 -accby-gpu 3 -pin-mb-b 400              59.6 s
./sgemm.out  -mb 18000 -pfcpu 3 -accby-gpu 3 -pin-mb-b 300              60 or more
./sgemm.out -mb 18000 -rm-late 1,2 -pfcpu 1,2,3   -pin-mb-b 400 -accby-gpu 3        52.17 s

# In these runs, -initgpu 3 somehow makes the kernel (13 s) faster (11 s).
./sgemm.out -mb 9000 -pfcpu 1,2 -initgpu 3 -rm-late 2 -accby-gpu 1      140 s
./sgemm.out -mb 9000 -pfcpu 1,2 -initgpu 3                              19.67 s
./sgemm.out -mb 9000 -pfcpu 1,2                                             17.2447
./sgemm.out -mb 9000 -pfcpu 1,2 -initgpu 3 -prefetch 2                      16.6061
./sgemm.out -mb 9000 -pfcpu 1,2 -initgpu 3 -prefetch 1,2                    16.83417
./sgemm.out -mb 9000 -pfcpu 1,2 -initgpu 3 -prefetch 2 -accby-gpu 1         139.0540
./sgemm.out -mb 9000 -pfcpu 1,2 -initgpu 3 -prefetch 1,2 -accby-gpu 1       139.36220
./sgemm.out -mb 9000 -pfcpu 1,2 -initgpu 3 -prefetch 1 -accby-gpu 2         157.0124
./sgemm.out -mb 9000 -pfcpu 1,2 -initgpu 3 -prefetch 1,2 -accby-gpu 2       51.7587
./sgemm.out -mb 9000 -pfcpu 1,2 -prefetch 3 -prefetch 2                     17.9126
./sgemm.out -mb 9000 -pfcpu 1,2 -prefetch 3 -prefetch 1,2                   18.2074
./sgemm.out -mb 9000 -pfcpu 1,2 -prefetch 3 -prefetch 2 -accby-gpu 1        140.2768
./sgemm.out -mb 9000 -pfcpu 1,2 -prefetch 3 -prefetch 1,2 -accby-gpu 1      140.6522
./sgemm.out -mb 9000 -pfcpu 1,2 -prefetch 3 -prefetch 1 -accby-gpu 2        156.3763
./sgemm.out -mb 9000 -pfcpu 1,2 -prefetch 3 -prefetch 1,2 -accby-gpu 2      86.8864
./sgemm.out -mb 9000 -pfcpu 1,2 -prefetch 3                                 17.46
./sgemm.out -mb 9000 -pfcpu 1,2 -prefetch 3,2                               17.84
./sgemm.out -mb 9000 -pfcpu 1,2 -prefetch 3,2 -output-pf 3                  17.66
./sgemm.out -mb 9000 -pfcpu 1,2 -initgpu 3 -prefetch 2 -output-pf 3         16.35
./sgemm.out -mb 9000 -pfcpu 1,2 -prefetch 2,3 -output-pf 3                  17.65


# 10 GB now
./sgemm.out -mb 10000                                           19.51 s
./sgemm.out -mb 10000 -prefloc-cpu 3 -accby-gpu 3               18.06 s
```

- See `ovh_summary.try_hints_sgemm.207953`. 

### ARIADNE
```
time ./sgemm.out -mb 18000                                      72 s
time ./sgemm.out -mb 18000                                      77 s # without pipelining
time ./sgemm.out -mb 18000 -prefloc-cpu 3 -accby-gpu 3          58 s
time ./sgemm.out -mb 18000 -prefloc-cpu 3 -accby-gpu 3          62 s # without pipelining
```
