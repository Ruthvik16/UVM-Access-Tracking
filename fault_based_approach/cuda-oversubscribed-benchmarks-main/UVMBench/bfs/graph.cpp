#include <ctime>
#include "graph.h"
#include <assert.h>

extern int *u_adjacencyList, *u_edgesOffset, *u_edgesSize;

extern void initCuda(Graph&);

void get_random_bytes(void *buffer, long nbytes) {
    time_t random_begin, random_end;
    time(&random_begin);
    int *random_arr = (int *)malloc(nbytes);
    assert(random_arr);
    FILE *fp_random = fopen("/dev/urandom", "r");
    assert(fp_random);
    size_t retval = fread(random_arr, 1, nbytes, fp_random);
    assert(retval == nbytes);
    time(&random_end);
    fprintf(stderr, "random array creation: %ld MB: %ld s\n", nbytes >> 20,
        random_end - random_begin);
}

/*
 * For CPU BFS, fill up adjacencyLists[][].
 * For GPU BFS, fill up G.adjacencyLists, G.edgesOffset, G.edgesSize,
 */
void readGraph(Graph &G, int argc, char **argv) {
    int n;
    int m;

    time_t begin, end, random_begin, random_end;
    time(&begin);

    //If no arguments then read graph from stdin
    bool fromStdin = argc <= 2;
    if (fromStdin) {
        scanf("%d %d", &n, &m);
    } else {
        if (!argv[2] || !argv[3]) {
            fprintf(stderr, "Error: expected three arguments. Check usage from README\n");
            exit(1);
        }
        srand(12345);
        n = atoi(argv[2]);
        m = atoi(argv[3]);
    }

    std::vector<std::vector<int> > adjecancyLists(n);

    #define RANDOM_READ_ONCE

    #ifdef RANDOM_READ_ONCE
    time(&random_begin);
    long nbytes = sizeof(int) * m * 2;
    unsigned int *random_arr = (unsigned int *)malloc(nbytes);
    assert(random_arr);
    FILE *fp_random = fopen("/dev/urandom", "r");
    assert(fp_random);
    size_t retval = fread(random_arr, 1, nbytes, fp_random);
    assert(retval == nbytes);
    time(&random_end);
    fprintf(stderr, "random array creation: %ld MB: %ld s\n", nbytes >> 20,
        random_end - random_begin);
    #endif

    for (int i = 0; i < m; i++) {
        int u, v;
        if (fromStdin) {
            scanf("%d %d", &u, &v);
        } else {
            #ifdef RANDOM_READ_ONCE
            u = random_arr[i*2] % n;
            v = random_arr[i*2 + 1] % n;
            #else

            u = rand() % n;
            v = rand() % n;
            #endif

        }
        adjecancyLists[u].push_back(v);
    }

    #ifdef RANDOM_READ_ONCE
    free(random_arr);
    #endif

    for (int i = 0; i < n; i++) {
        G.edgesOffset.push_back(G.adjacencyList.size());
        G.edgesSize.push_back(adjecancyLists[i].size());
        for (auto &edge: adjecancyLists[i]) {
            G.adjacencyList.push_back(edge);
        }
    }

    G.numVertices = n;
    G.numEdges = G.adjacencyList.size();
    time(&end);

    printf("Graph creation time [n = %ldM; m = %ldM]: %ld s\n",
        (long) n/1000000, (long) m/1000000, end - begin);
}

// IDEA: loop through src, fill each node's G.edgesSize.
// Then edgesOffset, and finally adjacencyLists.
void genGraph_GPU(Graph &G, int argc, char **argv) {
    long m, n;
    unsigned *src_arr, *dst_arr;
    time_t t1, t2;

    time(&t1);
    assert(argc >= 4);
    n = atol(argv[2]);
    m = atol(argv[3]);
    assert(n);
    assert(m);
    assert(m < ((unsigned) -1)); // overflow?
    G.numVertices = n;
    G.numEdges = m;

    initCuda(G);
    src_arr = (unsigned *) malloc(sizeof(unsigned) * m);
    // dst_arr = (unsigned *) malloc(sizeof(unsigned) * m);
    dst_arr = (unsigned *) u_adjacencyList;
    assert(src_arr);
    assert(dst_arr);

    get_random_bytes(src_arr, sizeof(unsigned) * m);
    get_random_bytes(dst_arr, sizeof(unsigned) * m);

    for (long i = 0; i < m; i++) {
        unsigned src = src_arr[i] % n;
        // G.edgesSize[src]++;
        u_edgesSize[src]++;
    }

    long cur_offset = 0;
    for (long i = 0; i < n; i++) {
        // G.edgesOffset[i] = cur_offset;
        u_edgesOffset[i] = cur_offset;
        cur_offset += u_edgesSize[i];
    }

    // Now, the adjacency list for vertex I in G.edgesOffset[I] to
    // G.edgesOffset[I+1]. We'll loop through dst_array and fill up this
    // sub-array. Within the sub-array, the offset is stored/maintained in:
    unsigned *offset_per_vertex = (unsigned *) calloc(n, sizeof(unsigned));
    assert(offset_per_vertex);

    for (long i = 0; i < m; i++) {
        unsigned src = src_arr[i] % n;

        // unsigned dst = dst_arr[i] % n;
        // G.adjacencyList[G.edgesOffset[src] + offset_per_vertex[src]] = dst;
        dst_arr[i] = dst_arr[i] % n;

        offset_per_vertex[src]++;
        assert(offset_per_vertex[src] <= u_edgesSize[src]);
    }
    time(&t2);
    printf("Generated random graph: %ld MB: %ld s\n", (4*m) >> 20, (t2 - t1));
}
