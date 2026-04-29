import time
import rmm
import cudf
import cugraph
import cupy # for randint
import argparse
import numpy as np
from cugraph.generators import rmat
import sys
import signal

# perplexity-generated code to speed up ctrl-c.
def quick_exit(signum, frame):
    print(f"quick_exit() invoked by signal {signum}")
    sys.exit(1)

signal.signal(signal.SIGINT, quick_exit)

if "-pool" in sys.argv:
    print("Using RMM pool allocator")
    rmm.reinitialize(
        managed_memory = True,
        pool_allocator = True,
        initial_pool_size = (6 << 30),
        )
else:
    rmm.reinitialize(managed_memory = True)

"""
sometimes execution is slow, this prints what's going on.
Common culprits: make_random_graph(), g1.number_of_vertices(), and checking the
output (to copy it back to CPU.
"""
verbose = False

assert(rmm.is_initialized())

def make_random_graph(n, m):
    """
    This function is very slow.
    """
    print(f"init graph {n} vertices, {m} edges")
    if (n // 1000000 and m // 1e6):
        print(f"init graph {n//1000000}M vertices, {m//1000000}M edges")

    g1 = cugraph.Graph()

    df = cudf.DataFrame()
    df['source']      = cupy.random.randint(n, size = m, dtype = np.int64)
    df['destination'] = cupy.random.randint(n, size = m, dtype = np.int64)
    df['weight']      = cupy.random.randint(1024, size = m, dtype = np.int64)

    if (verbose):
        print("initialized cudf.DataFrame for graph")
    g1.from_cudf_edgelist(
        df,
        store_transposed = True,
        )
    del df

    if (verbose):
        print("Initialized graph")
        print(g1.edges().head())

    if (m != g1.edges().shape[0]):
        print(f"lost some edges: expected {m}, got {g1.edges().shape[0]}")
    return g1

def make_random_graph_rmat(log_n, m):
    """
    This is fast. This throws a performance warning for store_transposed
    which doesn't seem to matter.
    """
    n = 1<<log_n
    print(f"init graph {n} vertices, {m} edges")
    if (n // 1000000 and m // 1e6):
        print(f"init graph {n//1000000}M vertices, {m//1000000}M edges")

    return rmat(log_n, m)

def weighted_directed_graph(
        n = 1000,
        m = 10000,
        renumber = False,
        ):
    """Generate a weighted, directed random graph.
    This does not use the rmat generator. It is slow, but truly random.
    """
    print(f"Generating graph: n = {n} {n//1000000} M, m = {m} {m // 1000000} M")
    t1 = time.time()
    source  = cupy.random.randint(0, n, size=m)
    dest    = cupy.random.randint(0, n, size=m)
    weights = cupy.random.uniform(1.0, 10001.0, size=m)
    t2 = time.time()

    graph_dict = {'source': source, 'destination': dest, 'weight': weights}

    df = cudf.DataFrame(graph_dict)

    G = cugraph.Graph(directed=True)
    try:
        G.from_cudf_edgelist(
            df,
            edge_attr = 'weight',
            renumber = renumber,
            )
    except RuntimeError as e:
        print(f"graph creation error\n{e}")
        exit(3)

    t3 = time.time()
    print(f"random numbers: {t2 - t1} s; generate graph: {t3 - t2} s")
    print(f"Memory usage for dataFrame (NOT cugraph) in MB:")
    print(f"{df.memory_usage(deep = True)//1000000}")
    del source, dest, weights

    return G
