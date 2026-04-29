#!/data/pranjal/cuda-benchmarks/rapids/rapids_venv/bin/python3
# Change the above shebang if you clone this script
"""
Copy of https://gist.github.com/afender/e1967c0f23f9da40edb9f843c46fe5deimport
Originally written by Alex Fender
Modified by Pranjal

Changes made: this uses a random graph, not the Twitter graph.
And this is a bit modular

This can be imported in py scripts as `from cugraph_uvm import *` to use these
imports and helpers readily.
"""

import time
import rmm
import cudf
import cugraph
import cupy # for randint
import argparse
import numpy as np
from cugraph.generators import rmat

from cugraph_helpers import *

def do_pagerank(n = 10000, m = 100000, use_rmat = True):
    """
    use_rmat uses cugraph.generators.rmat()
    Reduces N to the previous power-of-two
    """
    n = int(n)
    m = int(m)
    t1 = time.time()

    print("\n*** ***")
    if use_rmat:
        log_n = int(np.log2(n))
        g1 = make_random_graph_rmat(log_n, m)
        n = 2**log_n
    else:
        g1 = make_random_graph(n, m)
    t2 = time.time()

    print("generated random graph:", (t2 - t1), "s")
    pgrank_output = cugraph.pagerank(g1)

    t3 = time.time()
    print("executed pagerank algorithm:", (t3 - t2), "s")

    if (verbose):
        print("completed page rank execution")
        print(pgrank_output.head());

    # Fake dependency on the output - compiler shouldn't remove the function
    # call, and the output should be copied back to the CPU at the end

    """
    for i in range(n):
        if pgrank_output['vertex'][i] == n*n - i*i:
            print("magic!")
    """
    """
    pr = (pgrank_output['vertex']).to_numpy()
    if (pr.sum() == n*(n-1)/2):
        print("magic!")
    """
    t4 = time.time()

    print("checking output:", (t4 - t3), "s")
    print("graph init:", (t2 - t1), "s")
    print(f"pagerank [n = {n}, m = {m}]: {(t3 - t2):.3f} s\n*** ***\n")

    return pgrank_output


def main():
    parser = argparse.ArgumentParser(description = "Run the page rank algorithm \
    on randomly generated graphs using cugraph and UVM")
    parser.add_argument("-n", required=False, type=int, nargs='?', const=10, help="Number of vertices")
    parser.add_argument("-m", required=False, type=int, nargs='?', const=30, help="Number of edges")
    parser.add_argument("-pool", required=False, action = "store_true", help = "Use RMM pool allocator")
    
    args = parser.parse_args()
    n = args.n if args.n else 20
    m = args.m if args.m else 100

    do_pagerank(n, m)

    if (n == 20 and m == 100):
        do_pagerank(10e6, 50e6) # 1.0 seconds with the vanilla driver
        do_pagerank(50e6, 130e6) # 4.5
        do_pagerank(50e6, 200e6)


if __name__ == "__main__":
    main()
