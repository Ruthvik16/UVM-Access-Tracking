### Optimizations

Run `./needle -h` to see the available optimizations/hints.  
On Vindhya, for 10240 MB, the behaviour is:

- Vanilla driver: 1400-1450 s
- `-accessed-by`: 17 s
- `-accessed-by-input`: 13 s
- `-accessed-by-ref`: 8.15 s
- `-accessed-by-ref` with `BLOCK_SIZE=32`: 6.87 s
- `-pin-cpu`: 25 s
- `-optimal` is `-accessed-by-ref`
- A bit more optimal: `-DRD_WG_SIZE_0=64`, or `./needle64`: 6.36 s 

Complete log:
`/data/pranjal/cuda-benchmarks/perf_logs/perf_run_all.81261/ovh_summary.try_hints_per_buffer.81261`
#### Thread Block Size

- Originally, the block size was 16.
- 64 threads per block is 10-30% faster for varying data sizes.
- TODO: find optimal size per kernel (1 and 2).

#### Hints

#### 10 GB

```
```

#### 100% Oversubscription

- `./needle -mb 6000 -accby-gpu all -prefetch all -prefloc-cpu all`: the PrefLoc
  hint saves an additional 0.03 s, average 4.09 s

- `./needle -mb 9000 -accby-gpu 1 -prefetch all`: 6.38 s
- `./needle -mb 9000 -accby-gpu 1 -prefetch 1`: 6.55 s
- `./needle -mb 9000 -accby-gpu 2 -prefetch all`: 8.88 s
- `./needle -mb 9000 -accby-gpu all -prefetch all`: 6.41 s
- `./needle -mb 9000  -accby-gpu all  -prefetch 2 -prefloc-cpu 1`: 6.950s
- `./needle -mb 9000 -accby-gpu 1 -prefetch all  -prefloc-cpu all`: 6.31 s

12 GB:

- `./needle -mb 12000                                               1883.55 s
- `./needle -mb 12000 -accby-gpu all  -prefetch 2 -prefloc-cpu 1`: 10.00 s
- `./needle -mb 12000 -prefloc-cpu all -prefetch 2 -accby-gpu all`: 9.81 s
- `./needle -mb 12000 -prefloc-cpu all -prefetch 2 -accby-gpu 1`: 9.87 s
- `./needle -mb 12000 -prefloc-cpu all -prefetch 2 -accby-gpu 1 -prefloc-gpu 2`:
  12.53 s
- `./needle -mb 12000 -prefloc-cpu 1 -accby-gpu all`: 19.73 s
- `./needle -mb 12000 -accby-gpu all  -prefetch 2 -prefloc-cpu 1`: 9.74 s
- `./needle -mb 12000 -prefloc-cpu all -prefetch 2 -accby-gpu all`: 9.54 s

```
15 GB now

./needle -mb 15000 -gohan-disable all -accby-gpu all -prefetch 2         14.97 s
```

```
18 GB

./needle -mb 18000                                          3539.99
./needle -mb 18000 -accby-gpu all -prefetch 2               19.80 s
./needle -mb 18000 -accby-gpu all -prefetch 2 -pfcpu 1,2    19.19 s
```


### ARIADNE

```
./needle -mb 9000                           8.39 s
./needle -mb 9000                           8.71 s # uvm_perf_SDaware=0
./needle -mb 9000                           8.57 s # uvm_perf_fhp=0
./needle -mb 9000                           9.33 s # Baseline

./needle -mb 12000                          11.85 s
./needle -mb 12000                          12.5 s # Without pipelined prefetching
./needle -mb 12000  -accby-gpu 1            8.79 s
./needle -mb 12000  -accby-gpu 1            8.75 s # speeds up without pipelining
```

### From Performance Runs

```
# 2 or 5 runs per workload

# 6 GB - Very mild oversubscription, over about 5.7 GB available.
./needle -mb 6000             5.61271 +- 0.00545 seconds time elapsed  ( +-  0.10% )
./needle -mb 6000 -pfcpu all -prefetch 1 -accby-gpu 2              7.4418 +- 0.0144 seconds time elapsed  ( +-  0.19% )
./needle -mb 6000 -pfcpu all -prefetch 1 -accby-gpu all              7.4917 +- 0.0131 seconds time elapsed  ( +-  0.17% )
./needle -mb 6000 -pfcpu all -prefetch 1 -accby-gpu all             7.51061 +- 0.00942 seconds time elapsed  ( +-  0.13% )
./needle -mb 6000 -pfcpu all -prefetch 2 -accby-gpu 1              4.9299 +- 0.0174 seconds time elapsed  ( +-  0.35% )
./needle -mb 6000 -pfcpu all -prefetch 2 -accby-gpu all             4.96303 +- 0.00925 seconds time elapsed  ( +-  0.19% )
./needle -mb 6000 -pfcpu all -prefetch 2 -accby-gpu all             4.96954 +- 0.00521 seconds time elapsed  ( +-  0.10% )
./needle -mb 6000 -pfcpu all -prefetch 2 -accby-gpu all             4.97852 +- 0.00955 seconds time elapsed  ( +-  0.19% )
./needle -mb 6000 -pfcpu all -prefetch all              6.2139 +- 0.0117 seconds time elapsed  ( +-  0.19% )
./needle -mb 6000 -pfcpu all -prefetch all -accby-gpu all              3.9374 +- 0.0116 seconds time elapsed  ( +-  0.30% )
./needle -mb 6000 -pfcpu all -prefetch all -accby-gpu all              3.9827 +- 0.0171 seconds time elapsed  ( +-  0.43% )
./needle -mb 6000 -pfcpu all -prefetch all -accby-gpu all -prefloc-cpu all              3.9459 +- 0.0130 seconds time elapsed  ( +-  0.33% )
./needle -mb 6000 -pfcpu all -prefloc-cpu all -prefetch 1              8.9203 +- 0.0282 seconds time elapsed  ( +-  0.32% )
./needle -mb 6000 -pfcpu all -prefloc-cpu all -prefetch 2              7.8427 +- 0.0395 seconds time elapsed  ( +-  0.50% )
./needle -mb 6000 -pfcpu all -prefloc-cpu all -prefetch all -accby-gpu all             3.95547 +- 0.00702 seconds time elapsed  ( +-  0.18% )
./needle -mb 6000 -pfcpu all -rm-late 1 -accby-gpu all              7.5389 +- 0.0113 seconds time elapsed  ( +-  0.15% )

# 7.5 GB
./needle -mb 7500              361.58 +- 19.07 seconds time elapsed  ( +-  5.27% )
./needle -mb 7500 -pfcpu all -prefetch 1 -accby-gpu 2             9.62731 +- 0.00771 seconds time elapsed  ( +-  0.08% )
./needle -mb 7500 -pfcpu all -prefetch 1 -accby-gpu all              9.6475 +- 0.0112 seconds time elapsed  ( +-  0.12% )
./needle -mb 7500 -pfcpu all -prefetch 1 -accby-gpu all             9.64922 +- 0.00867 seconds time elapsed  ( +-  0.09% )
./needle -mb 7500 -pfcpu all -prefetch 2 -accby-gpu 1             5.85781 +- 0.00860 seconds time elapsed  ( +-  0.15% )
./needle -mb 7500 -pfcpu all -prefetch 2 -accby-gpu all              5.8742 +- 0.0120 seconds time elapsed  ( +-  0.20% )
./needle -mb 7500 -pfcpu all -prefetch 2 -accby-gpu all              5.8910 +- 0.0178 seconds time elapsed  ( +-  0.30% )
./needle -mb 7500 -pfcpu all -prefetch 2 -accby-gpu all             5.85198 +- 0.00999 seconds time elapsed  ( +-  0.17% )
./needle -mb 7500 -pfcpu all -prefetch all              308.99 +- 21.95 seconds time elapsed  ( +-  7.10% )
./needle -mb 7500 -pfcpu all -prefetch all -accby-gpu all              4.9632 +- 0.0101 seconds time elapsed  ( +-  0.20% )
./needle -mb 7500 -pfcpu all -prefetch all -accby-gpu all              5.0000 +- 0.0142 seconds time elapsed  ( +-  0.28% )
./needle -mb 7500 -pfcpu all -prefetch all -accby-gpu all -prefloc-cpu all             5.00393 +- 0.00328 seconds time elapsed  ( +-  0.07% )
./needle -mb 7500 -pfcpu all -prefloc-cpu all -prefetch 1             10.8506 +- 0.0318 seconds time elapsed  ( +-  0.29% )
./needle -mb 7500 -pfcpu all -prefloc-cpu all -prefetch 2              9.4380 +- 0.0310 seconds time elapsed  ( +-  0.33% )
./needle -mb 7500 -pfcpu all -prefloc-cpu all -prefetch all -accby-gpu all              5.0135 +- 0.0109 seconds time elapsed  ( +-  0.22% )
./needle -mb 7500 -pfcpu all -rm-late 1 -accby-gpu all              9.7115 +- 0.0130 seconds time elapsed  ( +-  0.13% )

# 9 GB - not a neat multiple of the WSS, so some gymnastics are neeed.
./needle -mb 9000            1013.847 +- 0.977 seconds time elapsed  ( +-  0.10% )
./needle -mb 9000 -pfcpu all -prefetch 1 -accby-gpu 2            11.61464 +- 0.00988 seconds time elapsed  ( +-  0.09% )
./needle -mb 9000 -pfcpu all -prefetch 1 -accby-gpu all             11.4969 +- 0.0216 seconds time elapsed  ( +-  0.19% )
./needle -mb 9000 -pfcpu all -prefetch 1 -accby-gpu all             11.4990 +- 0.0148 seconds time elapsed  ( +-  0.13% )
./needle -mb 9000 -pfcpu all -prefetch 2 -accby-gpu 1              6.5179 +- 0.0172 seconds time elapsed  ( +-  0.26% )
./needle -mb 9000 -pfcpu all -prefetch 2 -accby-gpu all              6.5303 +- 0.0165 seconds time elapsed  ( +-  0.25% )
./needle -mb 9000 -pfcpu all -prefetch 2 -accby-gpu all              6.5546 +- 0.0231 seconds time elapsed  ( +-  0.35% )
./needle -mb 9000 -pfcpu all -prefetch 2 -accby-gpu all              6.5604 +- 0.0192 seconds time elapsed  ( +-  0.29% )
./needle -mb 9000 -pfcpu all -prefetch all             1012.98 +- 2.64 seconds time elapsed  ( +-  0.26% )
./needle -mb 9000 -pfcpu all -prefetch all -accby-gpu all              6.0275 +- 0.0129 seconds time elapsed  ( +-  0.21% )
./needle -mb 9000 -pfcpu all -prefetch all -accby-gpu all             6.00578 +- 0.00651 seconds time elapsed  ( +-  0.11% )
./needle -mb 9000 -pfcpu all -prefetch all -accby-gpu all -prefloc-cpu all              6.0896 +- 0.0128 seconds time elapsed  ( +-  0.21% )
./needle -mb 9000 -pfcpu all -prefloc-cpu all -prefetch 1             13.2372 +- 0.0536 seconds time elapsed  ( +-  0.41% )
./needle -mb 9000 -pfcpu all -prefloc-cpu all -prefetch 2             11.6614 +- 0.0641 seconds time elapsed  ( +-  0.55% )
./needle -mb 9000 -pfcpu all -prefloc-cpu all -prefetch all -accby-gpu all              6.1173 +- 0.0256 seconds time elapsed  ( +-  0.42% )
./needle -mb 9000 -pfcpu all -rm-late 1 -accby-gpu all             11.9199 +- 0.0244 seconds time elapsed  ( +-  0.20% )

# 11 GB
./needle -mb 11000             1423.37 +- 2.79 seconds time elapsed  ( +-  0.20% )
./needle -mb 11000 -pfcpu all -prefetch 1 -accby-gpu 2             15.1906 +- 0.0113 seconds time elapsed  ( +-  0.07% )
./needle -mb 11000 -pfcpu all -prefetch 1 -accby-gpu all             15.1825 +- 0.0112 seconds time elapsed  ( +-  0.07% )
./needle -mb 11000 -pfcpu all -prefetch 1 -accby-gpu all            15.18327 +- 0.00587 seconds time elapsed  ( +-  0.04% )
./needle -mb 11000 -pfcpu all -prefetch 2 -accby-gpu 1              7.8228 +- 0.0154 seconds time elapsed  ( +-  0.20% )
./needle -mb 11000 -pfcpu all -prefetch 2 -accby-gpu all              7.8549 +- 0.0158 seconds time elapsed  ( +-  0.20% )
./needle -mb 11000 -pfcpu all -prefetch 2 -accby-gpu all              7.8848 +- 0.0205 seconds time elapsed  ( +-  0.26% )
./needle -mb 11000 -pfcpu all -prefetch 2 -accby-gpu all             7.93030 +- 0.00922 seconds time elapsed  ( +-  0.12% )
./needle -mb 11000 -pfcpu all -prefetch all             1419.99 +- 1.09 seconds time elapsed  ( +-  0.08% )
./needle -mb 11000 -pfcpu all -prefetch all -accby-gpu all              7.8753 +- 0.0276 seconds time elapsed  ( +-  0.35% )
./needle -mb 11000 -pfcpu all -prefetch all -accby-gpu all              7.8986 +- 0.0156 seconds time elapsed  ( +-  0.20% )
./needle -mb 11000 -pfcpu all -prefetch all -accby-gpu all -prefloc-cpu all              7.9614 +- 0.0128 seconds time elapsed  ( +-  0.16% )
./needle -mb 11000 -pfcpu all -prefloc-cpu all -prefetch 1             15.4452 +- 0.0571 seconds time elapsed  ( +-  0.37% )
./needle -mb 11000 -pfcpu all -prefloc-cpu all -prefetch 2             13.0080 +- 0.0228 seconds time elapsed  ( +-  0.18% )
./needle -mb 11000 -pfcpu all -prefloc-cpu all -prefetch all -accby-gpu all             7.94158 +- 0.00976 seconds time elapsed  ( +-  0.12% )
./needle -mb 11000 -pfcpu all -rm-late 1 -accby-gpu all             14.2920 +- 0.0147 seconds time elapsed  ( +-  0.10% )

# 12 GB
./needle -mb 12000             1874.62 +- 6.75 seconds time elapsed  ( +-  0.36% )
./needle -mb 12000 -pfcpu all -prefetch 1 -accby-gpu 2             17.4041 +- 0.0247 seconds time elapsed  ( +-  0.14% )
./needle -mb 12000 -pfcpu all -prefetch 1 -accby-gpu all             16.2782 +- 0.0180 seconds time elapsed  ( +-  0.11% )
./needle -mb 12000 -pfcpu all -prefetch 1 -accby-gpu all             16.3260 +- 0.0125 seconds time elapsed  ( +-  0.08% )
./needle -mb 12000 -pfcpu all -prefetch 2 -accby-gpu 1             12.1106 +- 0.0171 seconds time elapsed  ( +-  0.14% )
./needle -mb 12000 -pfcpu all -prefetch 2 -accby-gpu all              9.4383 +- 0.0198 seconds time elapsed  ( +-  0.21% )
./needle -mb 12000 -pfcpu all -prefetch 2 -accby-gpu all             9.44152 +- 0.00813 seconds time elapsed  ( +-  0.09% )
./needle -mb 12000 -pfcpu all -prefetch 2 -accby-gpu all             9.46017 +- 0.00699 seconds time elapsed  ( +-  0.07% )
./needle -mb 12000 -pfcpu all -prefetch all             1872.88 +- 4.73 seconds time elapsed  ( +-  0.25% )
./needle -mb 12000 -pfcpu all -prefetch all -accby-gpu all             10.8917 +- 0.0162 seconds time elapsed  ( +-  0.15% )
./needle -mb 12000 -pfcpu all -prefetch all -accby-gpu all             10.8987 +- 0.0209 seconds time elapsed  ( +-  0.19% )
./needle -mb 12000 -pfcpu all -prefetch all -accby-gpu all -prefloc-cpu all             10.9852 +- 0.0208 seconds time elapsed  ( +-  0.19% )
./needle -mb 12000 -pfcpu all -prefloc-cpu all -prefetch 1              18.566 +- 0.192 seconds time elapsed  ( +-  1.03% )
./needle -mb 12000 -pfcpu all -prefloc-cpu all -prefetch 2             16.1796 +- 0.0881 seconds time elapsed  ( +-  0.54% )
./needle -mb 12000 -pfcpu all -prefloc-cpu all -prefetch all -accby-gpu all             10.9970 +- 0.0162 seconds time elapsed  ( +-  0.15% )
./needle -mb 12000 -pfcpu all -rm-late 1 -accby-gpu all             15.5820 +- 0.0147 seconds time elapsed  ( +-  0.09% )

# 18 GB
./needle -mb 18000             3539.99 +- 2.91 seconds time elapsed  ( +-  0.08% )
./needle -mb 18000 -pfcpu all -prefetch 1 -accby-gpu 2              947.69 +- 55.51 seconds time elapsed  ( +-  5.86% )
./needle -mb 18000 -pfcpu all -prefetch 1 -accby-gpu all             29.6837 +- 0.0446 seconds time elapsed  ( +-  0.15% )
./needle -mb 18000 -pfcpu all -prefetch 1 -accby-gpu all            29.75724 +- 0.00358 seconds time elapsed  ( +-  0.01% )
./needle -mb 18000 -pfcpu all -prefetch 2 -accby-gpu 1             889.946 +- 0.176 seconds time elapsed  ( +-  0.02% )
./needle -mb 18000 -pfcpu all -prefetch 2 -accby-gpu all             19.1614 +- 0.0589 seconds time elapsed  ( +-  0.31% )
./needle -mb 18000 -pfcpu all -prefetch 2 -accby-gpu all             19.2068 +- 0.0219 seconds time elapsed  ( +-  0.11% )
./needle -mb 18000 -pfcpu all -prefetch 2 -accby-gpu all             19.3563 +- 0.0330 seconds time elapsed  ( +-  0.17% )
./needle -mb 18000 -pfcpu all -prefetch all             3557.84 +- 2.25 seconds time elapsed  ( +-  0.06% )
./needle -mb 18000 -pfcpu all -prefetch all -accby-gpu all             21.5226 +- 0.0135 seconds time elapsed  ( +-  0.06% )
./needle -mb 18000 -pfcpu all -prefetch all -accby-gpu all            21.35186 +- 0.00524 seconds time elapsed  ( +-  0.02% )
./needle -mb 18000 -pfcpu all -prefetch all -accby-gpu all -prefloc-cpu all             21.7112 +- 0.0731 seconds time elapsed  ( +-  0.34% )
./needle -mb 18000 -pfcpu all -prefloc-cpu all -prefetch 1              32.231 +- 0.197 seconds time elapsed  ( +-  0.61% )
./needle -mb 18000 -pfcpu all -prefloc-cpu all -prefetch 2              27.488 +- 0.241 seconds time elapsed  ( +-  0.88% )
./needle -mb 18000 -pfcpu all -prefloc-cpu all -prefetch all -accby-gpu all             21.5242 +- 0.0309 seconds time elapsed  ( +-  0.14% )
./needle -mb 18000 -pfcpu all -rm-late 1 -accby-gpu all              571.94 +- 1.45 seconds time elapsed  ( +-  0.25% )
```
