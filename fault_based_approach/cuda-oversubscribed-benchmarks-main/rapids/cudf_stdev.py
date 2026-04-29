#!/data/pranjal/cuda-benchmarks/rapids/rapids_venv/bin/python3
# Change the above shebang if you clone this script

import time
import cudf
import cudf.pandas
import cupy
import numpy as np
import pandas as pd
import argparse
import rmm
import sys

cudf.pandas.install()
rmm.reinitialize(managed_memory = True)
if "-pool" in sys.argv:
    rmm.reinitialize(managed_memory = True, pool_allocator = True)
    print("mm.reinitialize(managed_memory = True, pool_allocator = True)")


def random_df_numpy(nrows, ncols, max_int):
    """ Option 1: Uses numpy.random.randint. Slow.
    """
    t1 = time.time()
    random_array = np.random.randint(max_int, size = [nrows, ncols])
    t2 = time.time()
    print(f"Initialized random np array, size {size}: {(t2 - t1):.3f}")

    t1 = time.time()
    df = cudf.DataFrame(random_array)
    t2 = time.time()
    print(f"build dataframe: {(t2 - t1):.3f} s")

    return df

def random_df_cupy(nrows, ncols, max_int):
    """ Option 2. Fast. Not truly random.
    Uses on-GPU arithmetic operations instead of numpy.random
    Disregards max_int.
    """
    t1 = time.time()
    df = cudf.DataFrame(columns = range(ncols),
                        index = range(nrows),
                        dtype = cupy.int64,
                        )

    for col in df.columns:
        df[col] = df.index * 3 - col * col
    t2 = time.time()
    print(f"initialize arithmetic df: {(t2 - t1):.3f} s")
    return df

def find_stdev(
        max_int = 10**9,
        size    = [10**7, 5],
        runs    = 1,
        ):
    """ make a random np.array, make a DF from it and find some aggregate statistics
    on the GPU.
    """

    df = random_df_cupy(size[0], size[1], max_int)
    t2 = time.time()

    for i in range(runs):
        print(df.mean())
        t3 = time.time()
        print(f"find mean: {(t3 - t2):.3f} s")

        print(df.std())
        t4 = time.time()
        print(f"find stdev: {(t4 - t3):.3f} s")

        # a small change to stop interpreter optimizations
        r = np.random.randint(size[0])
        c = np.random.randint(size[1])
        df.loc[r, c] = np.random.randint(max_int)

        t2 = time.time()


def main():
    parser = argparse.ArgumentParser(description = "Find mean and stdev of a random DF")

    parser.add_argument("-rows", required=False, type=int, nargs=1, help="Number of rows")
    parser.add_argument("-cols", required=False, type=int, nargs=1, help="Number of columns")
    parser.add_argument("-runs", required=False, type=int, nargs=1, help="Number of iterations")
    parser.add_argument("-pool", required=False, action = "store_true", help = "Use RMM pool allocator")

    args = parser.parse_args()
    rows = args.rows[0] if args.rows else 20
    cols = args.cols[0] if args.cols else 100
    runs = args.runs[0] if args.runs else 5

    find_stdev(size = [rows, cols], runs = runs);


if __name__ == "__main__":
    main()
