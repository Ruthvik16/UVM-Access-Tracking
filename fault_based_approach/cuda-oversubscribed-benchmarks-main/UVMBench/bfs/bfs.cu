#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <cstring>
#include "graph.h"
#include "bfsCPU.h"
#include "bfsCUDA.cuh"
#include "cuda-macros-v1.h"

#ifndef THREAD_BLOCK_SIZE
#define THREAD_BLOCK_SIZE 1024
#endif

unsigned long thread_block_size;
unsigned compare_with_cpu = 0;

void runCpu(int startVertex, Graph &G, std::vector<int> &distance,
            std::vector<int> &parent, std::vector<bool> &visited) {
    printf("Starting sequential bfs.\n");
    auto start = std::chrono::steady_clock::now();
    bfsCPU(startVertex, G, distance, parent, visited);
    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n\n", duration);
}


#define checkError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }

}

int *u_adjacencyList;
int *u_edgesOffset;
int *u_edgesSize;
int *u_distance;
int *u_parent;
int *u_currentQueue;
int *u_nextQueue;
int *u_degrees;

int *incrDegrees;


void initCuda(Graph &G) {
    UVM_ALLOC_ARR(int, u_adjacencyList, G.numEdges);
    UVM_ALLOC_ARR(int, u_edgesOffset,   G.numVertices);
    UVM_ALLOC_ARR(int, u_edgesSize,     G.numVertices);
    UVM_ALLOC_ARR(int, u_distance,      G.numVertices);
    UVM_ALLOC_ARR(int, u_parent,        G.numVertices);
    UVM_ALLOC_ARR(int, u_currentQueue,  G.numVertices);
    UVM_ALLOC_ARR(int, u_nextQueue,     G.numVertices);
    UVM_ALLOC_ARR(int, u_degrees,       G.numVertices);

    fprintf(stderr, "%s: cudaMallocManaged: %ld MB for edges | %ld MB vertices, misc\n",
        __FILE__,
        G.numEdges * sizeof(int) / 1000000,
        (G.numVertices * 7 * sizeof(int) / 1000000));

    mickey_clear();
    mickey_register_va(u_adjacencyList);
    mickey_register_va((char*)u_adjacencyList + (1 << 28));


    checkError(cudaMallocHost((void **) &incrDegrees, sizeof(int) * G.numVertices));


    if (!compare_with_cpu)
        return;
    memcpy(u_adjacencyList, G.adjacencyList.data(), G.numEdges * sizeof(int));
    memcpy(u_edgesOffset, G.edgesOffset.data(), G.numVertices * sizeof(int));
    memcpy(u_edgesSize, G.edgesSize.data(), G.numVertices * sizeof(int));
}

void finalizeCuda() {

    checkError(cudaFree(u_adjacencyList));
    checkError(cudaFree(u_edgesOffset));
    checkError(cudaFree(u_edgesSize));
    checkError(cudaFree(u_distance));
    checkError(cudaFree(u_parent));
    checkError(cudaFree(u_currentQueue));
    checkError(cudaFree(u_nextQueue));
    checkError(cudaFree(u_degrees));
    checkError(cudaFreeHost(incrDegrees));
}



void checkOutput(std::vector<int> &distance, std::vector<int> &expectedDistance, Graph &G) {
    if (!compare_with_cpu) {
        printf("\n");
        return;
    }
    for (int i = 0; i < G.numVertices; i++) {
        if (*(u_distance+i) != expectedDistance[i]) {
            printf("%d %d %d\n", i, distance[i], expectedDistance[i]);
            printf("Wrong output!\n");
            exit(1);
        }
    }

    printf("Output OK!\n\n");
}


void initializeCudaBfs(int startVertex, std::vector<int> &distance, std::vector<int> &parent, Graph &G) {
    //initialize values
    std::fill(distance.begin(), distance.end(), std::numeric_limits<int>::max());
    std::fill(parent.begin(), parent.end(), std::numeric_limits<int>::max());
    distance[startVertex] = 0;
    parent[startVertex] = 0;

    memcpy(u_distance, distance.data(), G.numVertices * sizeof(int));
    memcpy(u_parent, parent.data(), G.numVertices * sizeof(int));

    int firstElementQueue = startVertex;
    *u_currentQueue = firstElementQueue;
}

void runCudaSimpleBfs(int startVertex, Graph &G, std::vector<int> &distance,
                      std::vector<int> &parent) {
    initializeCudaBfs(startVertex, distance, parent, G);

    int *changed;
    checkError(cudaMallocHost((void **) &changed, sizeof(int)));

    //launch kernel
    printf("Starting simple parallel bfs.\n");
    auto start = std::chrono::steady_clock::now();

    *changed = 1;
    int level = 0;
    while (*changed) {
        *changed = 0;
        // void *args[] = {&G.numVertices, &level, &d_adjacencyList, &d_edgesOffset, &d_edgesSize, &d_distance, &d_parent,
        //                 &changed};
        // checkError(cuLaunchKernel(cuSimpleBfs, G.numVertices / 1024 + 1, 1, 1,
        //                           1024, 1, 1, 0, 0, args, 0));

        simpleBfs<<<G.numVertices / thread_block_size + 1, thread_block_size>>>(
            G.numVertices,
            level,
            u_adjacencyList,
            u_edgesOffset,
            u_edgesSize,
            u_distance,
            u_parent,
            changed);
        cudaDeviceSynchronize();
        level++;
    }

    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n", duration);
    CHECK_CUDA_ERROR();
}


void runCudaQueueBfs(int startVertex, Graph &G, std::vector<int> &distance,
    std::vector<int> &parent) {
    initializeCudaBfs(startVertex, distance, parent, G);

    int *nextQueueSize;
    checkError(cudaMallocHost((void **)&nextQueueSize, sizeof(int)));
    //launch kernel
    printf("Starting queue parallel bfs.\n");
    auto start = std::chrono::steady_clock::now();

    int queueSize = 1;
    *nextQueueSize = 0;
    int level = 0;
    while (queueSize) {

        queueBfs<<<queueSize / thread_block_size + 1, thread_block_size>>>(
            level, u_adjacencyList, u_edgesOffset, u_edgesSize,
            u_distance, u_parent, queueSize, nextQueueSize,
            u_currentQueue, u_nextQueue);
        cudaDeviceSynchronize();
        level++;
        queueSize = *nextQueueSize;
        *nextQueueSize = 0;
        std::swap(u_currentQueue, u_nextQueue);
    }

    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n", duration);
    CHECK_CUDA_ERROR();
}

void nextLayer(int level, int queueSize) {

    nextLayer<<<queueSize / thread_block_size + 1, thread_block_size>>>(
        level, u_adjacencyList, u_edgesOffset, u_edgesSize,
        u_distance, u_parent, queueSize, u_currentQueue);
    cudaDeviceSynchronize();

}

void countDegrees(int level, int queueSize) {

    countDegrees<<<queueSize / thread_block_size + 1, thread_block_size>>>(
        u_adjacencyList, u_edgesOffset, u_edgesSize, u_parent, queueSize,
        u_currentQueue, u_degrees);
    cudaDeviceSynchronize();

}

void scanDegrees(int queueSize) {
//run kernel so every block in d_currentQueue has prefix sums calculated

    long num_tbs = queueSize / thread_block_size + 1;
    scanDegrees<<<queueSize / 1024 + 1, 1024>>>(queueSize, u_degrees, incrDegrees);
    cudaDeviceSynchronize();
    //count prefix sums on CPU for ends of blocks exclusive
    //already written previous block sum
    incrDegrees[0] = 0;
    for (int i = 1024; i < queueSize + 1024; i += 1024) {
        incrDegrees[i / 1024] += incrDegrees[i / 1024 - 1];
    }
}

void assignVerticesNextQueue(int queueSize, int nextQueueSize) {

    long num_tbs = queueSize / thread_block_size + 1;
    assignVerticesNextQueue<<<num_tbs, thread_block_size>>>(u_adjacencyList, u_edgesOffset, u_edgesSize, u_parent, queueSize, u_currentQueue,
        u_nextQueue, u_degrees, incrDegrees, nextQueueSize);
    cudaDeviceSynchronize();

}

void runCudaScanBfs(int startVertex, Graph &G, std::vector<int> &distance,
   std::vector<int> &parent) {
    initializeCudaBfs(startVertex, distance, parent, G);

    //launch kernel
    printf("Starting scan parallel bfs.\n");
    auto start = std::chrono::steady_clock::now();

    int queueSize = 1;
    int nextQueueSize = 0;
    int level = 0;
    while (queueSize) {
        assert(queueSize > 0);
        // next layer phase
        nextLayer(level, queueSize);
        // counting degrees phase
        countDegrees(level, queueSize);
        // doing scan on degrees
        scanDegrees(queueSize);
        nextQueueSize = incrDegrees[(queueSize - 1) / 1024 + 1];
        assert(nextQueueSize >= 0);
        // assigning vertices to nextQueue
        assignVerticesNextQueue(queueSize, nextQueueSize);

        level++;
        queueSize = nextQueueSize;
        std::swap(u_currentQueue, u_nextQueue);
    }

    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n", duration);
    CHECK_CUDA_ERROR();
}


int main(int argc, char **argv) {

    // read graph from standard input
    Graph G;
    int startVertex = atoi(argv[1]);
    bool cuda_madvise_read_mostly = false;
    bool adj_lists_pin_host = false;
    bool accessed_by_gpu_hint = false;

    thread_block_size = THREAD_BLOCK_SIZE;
    for (int i = 1; i < argc; i++) {
        GET_INT_FLAG(i, "-thread-block", thread_block_size);
        GET_BOOL_FLAG(i, "-read-mostly-hint", cuda_madvise_read_mostly, true);
        GET_BOOL_FLAG(i, "-pin-adj-cpu", adj_lists_pin_host, 1);
        GET_BOOL_FLAG(i, "-accessed-by-gpu", accessed_by_gpu_hint, true);
        GET_BOOL_FLAG(i, "-compare", compare_with_cpu, 1);
        get_hints(i, argv);
        if (strncmp(argv[i], "-h", 2) == 0)
            return -1;
        UNRECOGNIZED_ARGUMENT(i);
    }

    if (!compare_with_cpu) {
        genGraph_GPU(G, argc, argv);
    } else {
        readGraph(G, argc, argv);
        initCuda(G);
    }


    printf("Number of vertices %d\n", G.numVertices);
    printf("Number of edges %d\n\n", G.numEdges);

    //vectors for results
    std::vector<int> distance(G.numVertices, std::numeric_limits<int>::max());
    std::vector<int> parent(G.numVertices, std::numeric_limits<int>::max());
    std::vector<bool> visited(G.numVertices, false);

    //run CPU sequential bfs
    if (compare_with_cpu)
        runCpu(startVertex, G, distance, parent, visited);

    //save results from sequential bfs
    std::vector<int> expectedDistance(distance);
    std::vector<int> expectedParent(parent);
    auto start = std::chrono::steady_clock::now();

    if (cuda_madvise_read_mostly) {
        printf("Passing hint cudaMemAdviseSetReadMostly\n");
        CUDA_READ_MOSTLY_HINT(u_adjacencyList, G.numEdges * sizeof(int));
        CUDA_READ_MOSTLY_HINT(u_edgesOffset, G.numVertices * sizeof(int));
        CUDA_READ_MOSTLY_HINT(u_edgesSize, G.numVertices * sizeof(int));
    }

    if (adj_lists_pin_host) {
        CUDA_PIN_CPU_HINT(u_adjacencyList, G.numEdges*sizeof(int));
    }

    if (accessed_by_gpu_hint) {
        CUDA_ACCESSED_BY_GPU_HINT(u_adjacencyList, G.numEdges * sizeof(int));
    }

    //run CUDA simple parallel bfs
    runCudaSimpleBfs(startVertex, G, distance, parent);
    checkOutput(distance, expectedDistance, G);

    //run CUDA queue parallel bfs
    runCudaQueueBfs(startVertex, G, distance, parent);
    checkOutput(distance, expectedDistance, G);

    // run CUDA scan parallel bfs
    runCudaScanBfs(startVertex, G, distance, parent);
    checkOutput(distance, expectedDistance, G);
    CHECK_CUDA_ERROR();

    finalizeCuda();
    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Overall Elapsed time in milliseconds : %li ms.\n", duration);
    fprintf(stdout, "BFS [n = %ldM; m = %ldM]: %ld ms\n",
        (long) G.numVertices/1000000, (long) G.numEdges/1000000, duration);
    return 0;
}
