# Memory Hints

- Buffer 1 is input (A). Buffer 2 is output (B), initialized on the GPU.
- `-prefloc-cpu 1` makes the initialization faster. `-prefloc-gpu 2` does
  not help for some reason.

#### 6 GB

- Default: 5.46 s
- `./2DConvolution.exe -mb 6000 -prefloc-cpu 1 -prefetch all`: 4.93 s
- `./2DConvolution.exe -mb 6000 -prefloc-cpu 1 -prefetch 1 -prefloc-gpu 2`: 5.05
  s
- `./2DConvolution.exe -mb 6000 -prefloc-cpu 1 -prefloc-gpu 2`: 7.66 s

#### 9 GB

- `./2DConvolution.exe -mb 9000 -prefloc-cpu 1 -prefetch 2 -accby-gpu 1`: 6.65 s

#### 12 GB

- `./2DConvolution.exe -mb 12000 -accby-gpu 1`: 9.3027 s
- `./2DConvolution.exe -mb 12000 -accby-gpu 1 -prefloc-cpu 1`: 9.207 s
- `./2DConvolution.exe -mb 12000 -abg-late  1 -prefloc-cpu 1`: 9.303 s
- `./2DConvolution.exe -mb 12000 -accby-gpu 1 -prefloc-cpu 1 -prefloc-gpu 2`: 8.92
- `-abg-late 2`: slightly worse.
- `./2DConvolution.exe -mb 12000 -accby-gpu 1 -prefloc-cpu 1 -prefetch 2`:
  10.41 s. PF is not concurrent.
 
#### 30 GB

Entails a lot of evictions.

- `./2DConvolution.exe -mb 30000`: 22 s.
- `./2DConvolution.exe -mb 30000 -prefloc-cpu 1`: slower. 32 s.
- `./2DConvolution.exe -mb 30000  -abg-late 1`: 19.5 s
- `./2DConvolution.exe -mb 30000  -abg-late 1`: 19.5 s
- `./2DConvolution.exe -mb 30000  -accby-gpu 1`: 19.5 s
- `-prefloc-cpu 1` usually has no effect, although the first-touch
  initialization is faster.
- `-prefloc-gpu 2` helps a _little_, with the first touch on GPU, without
  greedily prefetching 15 GB to the device.
- `./2DConvolution.exe -mb 30000  -accby-gpu 1 -prefloc-gpu 2`: 19.1 s
- `./2DConvolution.exe -mb 30000  -accby-gpu 1 -prefloc-gpu 2 -prefloc-cpu 1`:
  18.9 s
- `-prefetch` usually backfires.
- `./2DConvolution.exe  -mb 30000 -rm-late 1 -prefloc-gpu 2 -prefloc-cpu 1`:
  20.7 s

#### The Full Log

Average of 3 runs:  

```
./2DConvolution.exe -mb 12000                                           9.3372 +- 0.0205 seconds time elapsed  ( +-  0.22% )
./2DConvolution.exe -mb 12000 -accby-gpu 1                              9.0899 +- 0.0158 seconds time elapsed  ( +-  0.17% )
./2DConvolution.exe -mb 12000 -accby-gpu 1 -prefloc-cpu 1               9.0334 +- 0.0196 seconds time elapsed  ( +-  0.22% )
./2DConvolution.exe -mb 12000 -accby-gpu 1 -prefloc-cpu 1 -prefetch 2   10.2836 +- 0.0256 seconds time elapsed  ( +-  0.25% )
./2DConvolution.exe -mb 12000 -accby-gpu 1 -prefloc-cpu 1 -prefloc-gpu 2    8.6618 +- 0.0146 seconds time elapsed  ( +-  0.17% )
./2DConvolution.exe -mb 12000 -accby-gpu 1 -prefloc-gpu 1 -prefetch 2       10.9835 +- 0.0265 seconds time elapsed  ( +-  0.24% )
./2DConvolution.exe -mb 12000 -accby-gpu 1 -prefloc-gpu 1 -prefloc-gpu 2    9.1684 +- 0.0495 seconds time elapsed  ( +-  0.54% )
./2DConvolution.exe -mb 12000 -accby-gpu 1 -prefloc-gpu 2 -prefloc-cpu 1    8.7213 +- 0.0404 seconds time elapsed  ( +-  0.46% )
./2DConvolution.exe -mb 12000 -accby-gpu 2                                  9.3738 +- 0.0207 seconds time elapsed  ( +-  0.22% )
./2DConvolution.exe -mb 12000 -accby-gpu all                                9.1368 +- 0.0123 seconds time elapsed  ( +-  0.13% )
./2DConvolution.exe -mb 12000 -accby-gpu all -prefetch 2                    8.89751 +- 0.00824 seconds time elapsed  ( +-  0.09% )
./2DConvolution.exe -mb 12000 -accby-gpu all -prefetch 2 -prefloc-cpu 1     8.83947 +- 0.00840 seconds time elapsed  ( +-  0.10% )
./2DConvolution.exe -mb 12000 -accby-gpu all -prefetch all                  10.71028 +- 0.00309 seconds time elapsed  ( +-  0.03% )
./2DConvolution.exe -mb 12000 -prefetch 1                                   11.1930 +- 0.0114 seconds time elapsed  ( +-  0.10% )
./2DConvolution.exe -mb 12000 -prefetch all                                 12.5494 +- 0.0118 seconds time elapsed  ( +-  0.09% )
./2DConvolution.exe -mb 12000 -prefloc-cpu 1 -prefetch all                  12.1874 +- 0.0415 seconds time elapsed  ( +-  0.34% )
./2DConvolution.exe -mb 12000 -prefloc-cpu all                              12.4732 +- 0.0537 seconds time elapsed  ( +-  0.43% )
./2DConvolution.exe -mb 12000 -prefloc-cpu all -accby-gpu all               8.20618 +- 0.00804 seconds time elapsed  ( +-  0.10% )
./2DConvolution.exe -mb 12000 -prefloc-cpu all -prefetch 2 -accby-gpu all   8.9188 +- 0.0103 seconds time elapsed  ( +-  0.12% )
./2DConvolution.exe -mb 12000 -prefloc-cpu all -prefetch all -accby-gpu 1   10.6609 +- 0.0336 seconds time elapsed  ( +-  0.31% )
./2DConvolution.exe -mb 12000 -prefloc-cpu all -prefetch all -accby-gpu all 10.7078 +- 0.0134 seconds time elapsed  ( +-  0.13% )
./2DConvolution.exe -mb 12000 -readmostly all                               60.3283 +- 0.0785 seconds time elapsed  ( +-  0.13% )


./2DConvolution.exe -mb 30000                                               21.36273 +- 0.00965 seconds time elapsed  ( +-  0.05% )
./2DConvolution.exe -mb 30000 -accby-gpu 1                                  19.3295 +- 0.0662 seconds time elapsed  ( +-  0.34% )
./2DConvolution.exe -mb 30000 -accby-gpu 1 -prefloc-cpu 1                   19.0802 +- 0.0489 seconds time elapsed  ( +-  0.26% )
./2DConvolution.exe -mb 30000 -accby-gpu 1 -prefloc-cpu 1 -prefetch 2       22.4805 +- 0.0543 seconds time elapsed  ( +-  0.24% )
./2DConvolution.exe -mb 30000 -accby-gpu 1 -prefloc-cpu 1 -prefloc-gpu 2    18.3876 +- 0.0137 seconds time elapsed  ( +-  0.07% )
./2DConvolution.exe -mb 30000 -accby-gpu 1 -prefloc-gpu 1 -prefetch 2       25.1274 +- 0.0240 seconds time elapsed  ( +-  0.10% )
./2DConvolution.exe -mb 30000 -accby-gpu 1 -prefloc-gpu 1 -prefloc-gpu 2    20.6202 +- 0.0376 seconds time elapsed  ( +-  0.18% )
./2DConvolution.exe -mb 30000 -accby-gpu 1 -prefloc-gpu 2 -prefloc-cpu 1    18.3833 +- 0.0243 seconds time elapsed  ( +-  0.13% )
./2DConvolution.exe -mb 30000 -accby-gpu 2                                  21.4689 +- 0.0146 seconds time elapsed  ( +-  0.07% )
./2DConvolution.exe -mb 30000 -accby-gpu all                                19.2651 +- 0.0705 seconds time elapsed  ( +-  0.37% )
./2DConvolution.exe -mb 30000 -accby-gpu all -prefetch 2                    18.9463 +- 0.0178 seconds time elapsed  ( +-  0.09% )
./2DConvolution.exe -mb 30000 -accby-gpu all -prefetch 2 -prefloc-cpu 1     18.7698 +- 0.0114 seconds time elapsed  ( +-  0.06% )
./2DConvolution.exe -mb 30000 -accby-gpu all -prefetch all                  23.0079 +- 0.0342 seconds time elapsed  ( +-  0.15% )
./2DConvolution.exe -mb 30000 -prefetch 1                                   24.80800 +- 0.00543 seconds time elapsed  ( +-  0.02% )
./2DConvolution.exe -mb 30000 -prefetch all                                 28.2480 +- 0.0538 seconds time elapsed  ( +-  0.19% )
./2DConvolution.exe -mb 30000 -prefloc-cpu 1 -prefetch all                  26.6220 +- 0.0242 seconds time elapsed  ( +-  0.09% )
./2DConvolution.exe -mb 30000 -prefloc-cpu all                              28.65080 +- 0.00918 seconds time elapsed  ( +-  0.03% )
./2DConvolution.exe -mb 30000 -prefloc-cpu all -accby-gpu all               17.78528 +- 0.00248 seconds time elapsed  ( +-  0.01% )
./2DConvolution.exe -mb 30000 -prefloc-cpu all -prefetch 2 -accby-gpu all   18.9919 +- 0.0117 seconds time elapsed  ( +-  0.06% )
./2DConvolution.exe -mb 30000 -prefloc-cpu all -prefetch all -accby-gpu 1   23.5752 +- 0.0198 seconds time elapsed  ( +-  0.08% )
./2DConvolution.exe -mb 30000 -prefloc-cpu all -prefetch all -accby-gpu all 23.0114 +- 0.0365 seconds time elapsed  ( +-  0.16% )
./2DConvolution.exe -mb 30000 -readmostly all                               156.546 +- 0.108 seconds time elapsed  ( +-  0.07% )

./2DConvolution.exe -mb 6000                                                5.42335 +- 0.00377 seconds time elapsed  ( +-  0.07% )
./2DConvolution.exe -mb 6000 -accby-gpu 1                                   5.46029 +- 0.00932 seconds time elapsed  ( +-  0.17% )
./2DConvolution.exe -mb 6000 -accby-gpu 1 -prefloc-cpu 1                    5.4555 +- 0.0155 seconds time elapsed  ( +-  0.28% )
./2DConvolution.exe -mb 6000 -accby-gpu 1 -prefloc-cpu 1 -prefetch 2        4.9401 +- 0.0182 seconds time elapsed  ( +-  0.37% )
./2DConvolution.exe -mb 6000 -accby-gpu 1 -prefloc-cpu 1 -prefloc-gpu 2     5.2356 +- 0.0101 seconds time elapsed  ( +-  0.19% )
./2DConvolution.exe -mb 6000 -accby-gpu 1 -prefloc-gpu 1 -prefetch 2        5.15345 +- 0.00502 seconds time elapsed  ( +-  0.10% )
./2DConvolution.exe -mb 6000 -accby-gpu 1 -prefloc-gpu 1 -prefloc-gpu 2     5.2640 +- 0.0219 seconds time elapsed  ( +-  0.42% )
./2DConvolution.exe -mb 6000 -accby-gpu 1 -prefloc-gpu 2 -prefloc-cpu 1     5.2545 +- 0.0123 seconds time elapsed  ( +-  0.23% )
./2DConvolution.exe -mb 6000 -accby-gpu 2                                   5.4340 +- 0.0216 seconds time elapsed  ( +-  0.40% )
./2DConvolution.exe -mb 6000 -accby-gpu all                                 5.5162 +- 0.0250 seconds time elapsed  ( +-  0.45% )
./2DConvolution.exe -mb 6000 -accby-gpu all -prefetch 2                     5.03079 +- 0.00798 seconds time elapsed  ( +-  0.16% )
./2DConvolution.exe -mb 6000 -accby-gpu all -prefetch 2 -prefloc-cpu 1      4.9902 +- 0.0128 seconds time elapsed  ( +-  0.26% )
./2DConvolution.exe -mb 6000 -accby-gpu all -prefetch all                   4.93079 +- 0.00442 seconds time elapsed  ( +-  0.09% )
./2DConvolution.exe -mb 6000 -prefetch 1                                    5.12577 +- 0.00681 seconds time elapsed  ( +-  0.13% )
./2DConvolution.exe -mb 6000 -prefetch all                                  5.5794 +- 0.0271 seconds time elapsed  ( +-  0.49% )
./2DConvolution.exe -mb 6000 -prefloc-cpu 1 -prefetch all                   4.82228 +- 0.00434 seconds time elapsed  ( +-  0.09% )
./2DConvolution.exe -mb 6000 -prefloc-cpu all                               7.11878 +- 0.00561 seconds time elapsed  ( +-  0.08% )
./2DConvolution.exe -mb 6000 -prefloc-cpu all -accby-gpu all                4.9888 +- 0.0111 seconds time elapsed  ( +-  0.22% )
./2DConvolution.exe -mb 6000 -prefloc-cpu all -prefetch 2 -accby-gpu all    5.0042 +- 0.0191 seconds time elapsed  ( +-  0.38% )
./2DConvolution.exe -mb 6000 -prefloc-cpu all -prefetch all -accby-gpu 1    4.82372 +- 0.00752 seconds time elapsed  ( +-  0.16% )
./2DConvolution.exe -mb 6000 -prefloc-cpu all -prefetch all -accby-gpu all  4.88290 +- 0.00356 seconds time elapsed  ( +-  0.07% )
./2DConvolution.exe -mb 6000 -readmostly all                                29.910 +- 0.148 seconds time elapsed  ( +-  0.49% )

./2DConvolution.exe -mb 7500                                                6.38120 +- 0.00532 seconds time elapsed  ( +-  0.08% )
./2DConvolution.exe -mb 7500 -accby-gpu 1                                   6.3704 +- 0.0319 seconds time elapsed  ( +-  0.50% )
./2DConvolution.exe -mb 7500 -accby-gpu 1 -prefloc-cpu 1                    6.3196 +- 0.0131 seconds time elapsed  ( +-  0.21% )
./2DConvolution.exe -mb 7500 -accby-gpu 1 -prefloc-cpu 1 -prefetch 2        5.7444 +- 0.0152 seconds time elapsed  ( +-  0.26% )
./2DConvolution.exe -mb 7500 -accby-gpu 1 -prefloc-cpu 1 -prefloc-gpu 2     6.1202 +- 0.0123 seconds time elapsed  ( +-  0.20% )
./2DConvolution.exe -mb 7500 -accby-gpu 1 -prefloc-gpu 1 -prefetch 2        6.01581 +- 0.00576 seconds time elapsed  ( +-  0.10% )
./2DConvolution.exe -mb 7500 -accby-gpu 1 -prefloc-gpu 1 -prefloc-gpu 2     6.2446 +- 0.0135 seconds time elapsed  ( +-  0.22% )
./2DConvolution.exe -mb 7500 -accby-gpu 1 -prefloc-gpu 2 -prefloc-cpu 1     6.1222 +- 0.0150 seconds time elapsed  ( +-  0.25% )
./2DConvolution.exe -mb 7500 -accby-gpu 2                                   6.4341 +- 0.0221 seconds time elapsed  ( +-  0.34% )
./2DConvolution.exe -mb 7500 -accby-gpu all                                 6.4445 +- 0.0265 seconds time elapsed  ( +-  0.41% )
./2DConvolution.exe -mb 7500 -accby-gpu all -prefetch 2                     5.8358 +- 0.0107 seconds time elapsed  ( +-  0.18% )
./2DConvolution.exe -mb 7500 -accby-gpu all -prefetch 2 -prefloc-cpu 1      5.7963 +- 0.0225 seconds time elapsed  ( +-  0.39% )
./2DConvolution.exe -mb 7500 -accby-gpu all -prefetch all                   5.97710 +- 0.00279 seconds time elapsed  ( +-  0.05% )
./2DConvolution.exe -mb 7500 -prefetch 1                                    6.1925 +- 0.0151 seconds time elapsed  ( +-  0.24% )
./2DConvolution.exe -mb 7500 -prefetch all                                  6.50006 +- 0.00828 seconds time elapsed  ( +-  0.13% )
./2DConvolution.exe -mb 7500 -prefloc-cpu 1 -prefetch all                   5.97386 +- 0.00829 seconds time elapsed  ( +-  0.14% )
./2DConvolution.exe -mb 7500 -prefloc-cpu all                               8.4408 +- 0.0104 seconds time elapsed  ( +-  0.12% )
./2DConvolution.exe -mb 7500 -prefloc-cpu all -accby-gpu all                5.7726 +- 0.0135 seconds time elapsed  ( +-  0.23% )
./2DConvolution.exe -mb 7500 -prefloc-cpu all -prefetch 2 -accby-gpu all    5.8271 +- 0.0148 seconds time elapsed  ( +-  0.25% )
./2DConvolution.exe -mb 7500 -prefloc-cpu all -prefetch all -accby-gpu 1    5.9213 +- 0.0363 seconds time elapsed  ( +-  0.61% )
./2DConvolution.exe -mb 7500 -prefloc-cpu all -prefetch all -accby-gpu all  5.9355 +- 0.0197 seconds time elapsed  ( +-  0.33% )
./2DConvolution.exe -mb 7500 -readmostly all                                36.9446 +- 0.0273 seconds time elapsed  ( +-  0.07% )

./2DConvolution.exe -mb 9000                                                7.3745 +- 0.0168 seconds time elapsed  ( +-  0.23% )
./2DConvolution.exe -mb 9000 -accby-gpu 1                                   7.3384 +- 0.0304 seconds time elapsed  ( +-  0.41% )
./2DConvolution.exe -mb 9000 -accby-gpu 1 -prefloc-cpu 1                    7.25521 +- 0.00794 seconds time elapsed  ( +-  0.11% )
./2DConvolution.exe -mb 9000 -accby-gpu 1 -prefloc-cpu 1 -prefetch 2        6.5299 +- 0.0108 seconds time elapsed  ( +-  0.17% )
./2DConvolution.exe -mb 9000 -accby-gpu 1 -prefloc-cpu 1 -prefloc-gpu 2     6.9803 +- 0.0219 seconds time elapsed  ( +-  0.31% )
./2DConvolution.exe -mb 9000 -accby-gpu 1 -prefloc-gpu 1 -prefetch 2        6.86752 +- 0.00981 seconds time elapsed  ( +-  0.14% )
./2DConvolution.exe -mb 9000 -accby-gpu 1 -prefloc-gpu 1 -prefloc-gpu 2     7.20210 +- 0.00592 seconds time elapsed  ( +-  0.08% )
./2DConvolution.exe -mb 9000 -accby-gpu 1 -prefloc-gpu 2 -prefloc-cpu 1     6.9265 +- 0.0136 seconds time elapsed  ( +-  0.20% )
./2DConvolution.exe -mb 9000 -accby-gpu 2                                   7.4410 +- 0.0195 seconds time elapsed  ( +-  0.26% )
./2DConvolution.exe -mb 9000 -accby-gpu all                                 7.3779 +- 0.0105 seconds time elapsed  ( +-  0.14% )
./2DConvolution.exe -mb 9000 -accby-gpu all -prefetch 2                     6.6584 +- 0.0138 seconds time elapsed  ( +-  0.21% )
./2DConvolution.exe -mb 9000 -accby-gpu all -prefetch 2 -prefloc-cpu 1      6.56496 +- 0.00898 seconds time elapsed  ( +-  0.14% )
./2DConvolution.exe -mb 9000 -accby-gpu all -prefetch all                   7.0206 +- 0.0229 seconds time elapsed  ( +-  0.33% )
./2DConvolution.exe -mb 9000 -prefetch 1                                    7.13288 +- 0.00277 seconds time elapsed  ( +-  0.04% )
./2DConvolution.exe -mb 9000 -prefetch all                                  7.3319 +- 0.0154 seconds time elapsed  ( +-  0.21% )
./2DConvolution.exe -mb 9000 -prefloc-cpu 1 -prefetch all                   7.1192 +- 0.0105 seconds time elapsed  ( +-  0.15% )
./2DConvolution.exe -mb 9000 -prefloc-cpu all                               9.7312 +- 0.0127 seconds time elapsed  ( +-  0.13% )
./2DConvolution.exe -mb 9000 -prefloc-cpu all -accby-gpu all                6.58896 +- 0.00633 seconds time elapsed  ( +-  0.10% )
./2DConvolution.exe -mb 9000 -prefloc-cpu all -prefetch 2 -accby-gpu all    6.6556 +- 0.0126 seconds time elapsed  ( +-  0.19% )
./2DConvolution.exe -mb 9000 -prefloc-cpu all -prefetch all -accby-gpu 1    6.9792 +- 0.0436 seconds time elapsed  ( +-  0.63% )
./2DConvolution.exe -mb 9000 -prefloc-cpu all -prefetch all -accby-gpu all  7.0075 +- 0.0356 seconds time elapsed  ( +-  0.51% )
./2DConvolution.exe -mb 9000 -readmostly all                                45.0511 +- 0.0710 seconds time elapsed  ( +-  0.16% )
```

### More Nuanced Hints

- Evictions start at about 5.8 GB.   
- We should be using separate streams for `cudaMemPrefetchAsync` for better performance.

```
./2DConvolution.exe -mb 6000 -pfcpu 1 -accby-gpu 1 -initgpu 2 -output-pf 2:   4.29 s
./2DConvolution.exe -mb 6000 -pfcpu 1 -abg-late  1  -initgpu 2  -output-pf 2: 4.30 s
./2DConvolution.exe -mb 6000 -pfcpu 1 -prefetch 1  -initgpu 2  -output-pf 2:  4.37 s
./2DConvolution.exe -mb 6000 -pfcpu 1 -prefetch 1  -prefetch 2 -output-pf 2:  4.43 s
./2DConvolution.exe -mb 6000 -pfcpu 1 -accby-gpu 1 -prefetch 2 -output-pf 2:  4.26 s
```


```
./2DConvolution.exe -mb 30000 -pfcpu 1 -initgpu 2 -output-pf 2 -accby-gpu 1:    17.19 s
./2DConvolution.exe -mb 30000 -pfcpu 1 -initgpu 2              -accby-gpu 1     18.16 s
./2DConvolution.exe -mb 30000 -pfcpu 1 -initgpu 2 -output-pf 2:                 19.49 s
./2DConvolution.exe -mb 30000 -pfcpu 1 -initgpu 2 -output-pf 2 -abg-late 1:     19.49 s
./2DConvolution.exe -mb 30000 -pfcpu 1 -initgpu 2 -output-pf 2 -abg-late 1      17.61 s

./2DConvolution.exe -mb 30000 -pfcpu 1 -initgpu 2 -rm-late 1:                   19.64 s
./2DConvolution.exe -mb 30000 -pfcpu 1 -initgpu 2 -rm-late 1:                   19.13 s
./2DConvolution.exe -mb 30000 -pfcpu 1 -prefloc-cpu 2 -accby-gpu all:           17.38 s
./2DConvolution.exe -mb 30000 -pfcpu 1,2  -accby-gpu all                        15.07 s
./2DConvolution.exe -mb 30000 -pfcpu 1,2 -accby-gpu 2 -rm-late 1                18.53 s
./2DConvolution.exe -mb 30000 -pfcpu 1 -accby-gpu 1,2 -prefetch 2 -output-pf 2  17.80 s
```
