## Pagerank

On an RTX 2060 (6 GB),

- `./cugraph_pagerank.py -n 10000000 -m 100000000` takes 15 s.
- `./cugraph_pagerank.py -n 10000000 -m 200000000` has some evictions, takes 65 s.
- `./cugraph_pagerank.py -n 10000000 -m 300000000` incurs thrashing. Takes 160 s.

## SSSP

```
[ 1 RUNS, vanilla43462 ]	  ./cugraph_sssp.py -n 10000000 -m 300000000 -iters 1      2008.881762919 seconds time elapsed
[ 1 RUNS, vanilla43462 ]	  ./cugraph_sssp.py -n 10000000 -m 303000000 -iters 1      2746.654465049 seconds time elapsed
[ 1 RUNS, vanilla43462 ]	  ./cugraph_sssp.py -n 10000000 -m 306000000 -iters 1      3648.007928059 seconds time elapsed

[ 1 RUNS ]   ./cugraph_sssp.py -n 10000000 -m 225000000 -iters 2  _npf     234.556458237 seconds time elapsed
[ 1 RUNS ]   ./cugraph_sssp.py -n 10000000 -m 250000000 -iters 2  _npf     477.606883762 seconds time elapsed
[ 1 RUNS ]   ./cugraph_sssp.py -n 10000000 -m 275000000 -iters 2  _npf     970.283184589 seconds time elapsed
[ 1 RUNS ]   ./cugraph_sssp.py -n 10000000 -m 300000000 -iters 2  _npf    4288.877798119 seconds time elapsed
[ 1 RUNS ]   ./cugraph_sssp.py -n 10000000 -m 225000000 -iters 2        65.716612433 seconds time elapsed
[ 1 RUNS ]   ./cugraph_sssp.py -n 10000000 -m 250000000 -iters 2       120.085522526 seconds time elapsed
[ 1 RUNS ]   ./cugraph_sssp.py -n 10000000 -m 275000000 -iters 2       234.595570695 seconds time elapsed
[ 1 RUNS ]   ./cugraph_sssp.py -n 10000000 -m 300000000 -iters 2      4322.047583340 seconds time elapsed
[ 1 RUNS ]  ./cugraph_sssp.py -n 10000000 -m 325000000 -iters 2 : Timed out or killed [6000]
[ 1 RUNS ]   ./cugraph_sssp.py -n 10000000 -m 225000000 -iters 2  _fpf      52.597628598 seconds time elapsed
[ 1 RUNS ]   ./cugraph_sssp.py -n 10000000 -m 250000000 -iters 2  _fpf      96.110007942 seconds time elapsed
[ 1 RUNS ]   ./cugraph_sssp.py -n 10000000 -m 275000000 -iters 2  _fpf    2257.226516513 seconds time elapsed
[ 1 RUNS ]   ./cugraph_sssp.py -n 10000000 -m 300000000 -iters 2  _fpf    2932.448436546 seconds time elapsed
[ 1 RUNS ]  ./cugraph_sssp.py -n 10000000 -m 325000000 -iters 2 _fpf: Timed out or killed [8000]
[ 1 RUNS ]   ./cugraph_sssp.py -n 10000000 -m 250000000 -iters 1  _npf     376.403672223 seconds time elapsed
[ 1 RUNS ]   ./cugraph_sssp.py -n 10000000 -m 300000000 -iters 1  _npf    2305.372499733 seconds time elapsed
[ 1 RUNS ]   ./cugraph_sssp.py -n 10000000 -m 325000000 -iters 1  _npf    7367.370999980 seconds time elapsed
[ 1 RUNS ]  ./cugraph_sssp.py -n 10000000 -m 350000000 -iters 1  _npf: Timed out or killed [8000] <elapsed>
[ 1 RUNS ]   ./cugraph_sssp.py -n 10000000 -m 250000000 -iters 1  _tbn      97.316554222 seconds time elapsed
[ 1 RUNS ]   ./cugraph_sssp.py -n 10000000 -m 300000000 -iters 1  _tbn    2282.027474386 seconds time elapsed
[ 1 RUNS ]  ./cugraph_sssp.py -n 10000000 -m 325000000 -iters 1  _tbn: Timed out or killed [8000] <elapsed>
[ 1 RUNS ]   ./cugraph_sssp.py -n 10000000 -m 250000000 -iters 1  _fpf      79.408429339 seconds time elapsed
[ 1 RUNS ]   ./cugraph_sssp.py -n 10000000 -m 300000000 -iters 1  _fpf    1629.682958715 seconds time elapsed
[ 1 RUNS ]  ./cugraph_sssp.py -n 10000000 -m 325000000 -iters 1  _fpf: Timed out or killed [8000] <elapsed>
```
