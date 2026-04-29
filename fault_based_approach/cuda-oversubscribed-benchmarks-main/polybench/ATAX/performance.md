### Thread Block Size

- Default: 256
- 32 is great
- 64 is worse than the baseline, somehow.

### Prefetch Hint

- Does not help with larger data sizes. Kernel is much faster.
```
./atax.exe -mb 5500:		4.13 s
CPU+GPU: 0.997 s | Mem 5500 MB

./atax.exe -mb 5500 -prefetch all:	3.93 s
CPU+GPU: 0.684 s | Mem 5500 MB
```
