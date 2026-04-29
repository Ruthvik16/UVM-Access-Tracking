#!/data/pranjal/cuda-benchmarks/rapids/rapids_venv/bin/python3

from cugraph_helpers import *

def do_sssp(n = 10000, m = 100000, iters = 3):
    """Generating random graphs is usually slower than SSSP.
    So, do multiple iterations.
    """

    sssp_args = f"[n = {n//1000000}M, m = {m//1000000}M]"
    print(f"Doing SSSP: {iters} iterations")
    gr = weighted_directed_graph(n, m)

    for src_node in range(iters):
        t2 = time.time()
        distances = cugraph.sssp(gr, src_node)
        t3 = time.time()
        print(f"Completed SSSP {sssp_args} from src {src_node}: ", (t3 - t2), "s")

        # Fake dependency on the output - avoid compiler/JIT optmizations
        if distances['predecessor'].max() == 3010:
            print("magic! (Ignore)")
            print(f"obtained distances\n{distances}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run SSSP")
    parser.add_argument("-n", required=False, type=int, nargs=1, help="Number of vertices")
    parser.add_argument("-m", required=False, type=int, nargs=1, help="Number of edges")
    parser.add_argument("-iters", required=False, type=int, nargs=1, help="SSSP iterations")
    parser.add_argument("-pool", required=False, action = "store_true", help = "Use RMM pool allocator")
    
    args = parser.parse_args()
    n = args.n[0] if args.n else 10000
    m = args.m[0] if args.m else 100000
    iters = args.iters[0] if args.iters else 3

    do_sssp(n = n, m = m, iters = iters)
