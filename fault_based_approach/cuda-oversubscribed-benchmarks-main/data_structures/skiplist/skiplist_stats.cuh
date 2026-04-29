#ifndef SKIPLIST_STATS_CUH
#define SKIPLIST_STATS_CUH

#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>

#define DEPTH 32
#define LATERAL_NODES 32

struct perNodeLockStat {
  unsigned long long int lockAttempts;
  unsigned long long int lockSuccesses;
  int level;
};

class SkiplistStats {
private:
  unsigned long long int *splitCount;
  unsigned long long int *mergeCount;
  unsigned long long int *maxHeight;
  unsigned long long int *avgHeight;
  unsigned long long int *observedHeight;
  unsigned long long int *lateralMovement;
  unsigned long long int *downwardMovement;
  unsigned long long int *downMv;
  unsigned long long int *lateralMv;
  unsigned long long int *lockRetry;
  unsigned long long int *retryCounterOverflow;
  unsigned long long int *lockSuccess;
  unsigned int totalNodes;
  perNodeLockStat *perNodeArray;

  // TODO: extend for each level for different memory pool impl

public:
  SkiplistStats(uint32_t totalNodes) {
    cudaMallocManaged(&splitCount, sizeof(unsigned long long int));
    cudaMallocManaged(&mergeCount, sizeof(unsigned long long int));
    cudaMallocManaged(&maxHeight, sizeof(unsigned long long int));
    cudaMallocManaged(&avgHeight, sizeof(unsigned long long int));
    cudaMallocManaged(&observedHeight, sizeof(unsigned long long int));
    cudaMallocManaged(&lateralMovement, sizeof(unsigned long long int));
    cudaMallocManaged(&downwardMovement, sizeof(unsigned long long int));
    cudaMallocManaged((void **)&downMv, sizeof(unsigned long long int) * DEPTH);
    cudaMallocManaged((void **)&lateralMv,
                      sizeof(unsigned long long int) * LATERAL_NODES);
    cudaMallocManaged(&lockRetry, sizeof(unsigned long long int));
    cudaMallocManaged(&retryCounterOverflow, sizeof(unsigned long long int));
    cudaMallocManaged(&lockSuccess, sizeof(unsigned long long int));
    cudaMallocManaged((void **)&perNodeArray,
                      sizeof(perNodeLockStat) * totalNodes);
    reset();
  }

  ~SkiplistStats() {
    cudaFree(splitCount);
    cudaFree(mergeCount);
    cudaFree(maxHeight);
    cudaFree(avgHeight);
    cudaFree(observedHeight);
    cudaFree(lateralMovement);
    cudaFree(downwardMovement);
    cudaFree(lockRetry);
    cudaFree(retryCounterOverflow);
    cudaFree(lockSuccess);
    cudaFree(perNodeArray);
  }

  // Device functions
  __device__ void recordSplit() { atomicAdd(splitCount, 1); }
  __device__ void recordSplit(int n) { atomicAdd(splitCount, n); }
  __device__ void recordMerge() { atomicAdd(mergeCount, 1); }
  __device__ void recordMaxHeight() { atomicAdd(maxHeight, 1); }
  __device__ void recordAvgHeight() { atomicAdd(avgHeight, 1); }
  __device__ void recordObservedHeight() { atomicAdd(observedHeight, 1); }
  __device__ void recordLateralMovement() { atomicAdd(lateralMovement, 1); }
  __device__ void recordDownwardMovement() { atomicAdd(downwardMovement, 1); }
  __device__ void recordLateralMv(int level) {
    atomicAdd(&lateralMv[level], 1);
  }
  __device__ void recordDownMv(int level) { atomicAdd(&downMv[level], 1); }
  __device__ void recordLockAttempt() {
    if (*lockRetry == UINT64_MAX) {
      printf("Lock retry overflow detected!\n");
      atomicAdd(retryCounterOverflow, 1);
    }
    atomicAdd(lockRetry, 1);
  }
  __device__ void recordLockSuccess() { atomicAdd(lockSuccess, 1); }
  // Host functions
  void reset() {
    *splitCount = 0;
    *mergeCount = 0;
    *avgHeight = 0;
    *maxHeight = 0;
    *observedHeight = 0;
    *lateralMovement = 0;
    *downwardMovement = 0;
    *lockRetry = 0;
    *lockSuccess = 0;
    *retryCounterOverflow = 0;
    int i = 0;
    for (; i < DEPTH; i++)
      downMv[i] = 0;

    i = 0;
    for (; i < LATERAL_NODES; i++)
      lateralMv[i] = 0;
  }

  unsigned long long int getSplitCount() const { return *splitCount; }

  unsigned long long int getMergeCount() const { return *mergeCount; }
  unsigned long long int getAvgHeight() const { return *avgHeight; }
  unsigned long long int getMaxHeight() const { return *maxHeight; }
  unsigned long long int getObservedHeight() const { return *observedHeight; }
  unsigned long long int getLateralMovement() const { return *lateralMovement; }
  unsigned long long int getDownwardMovement() const {
    return *downwardMovement;
  }
  unsigned long long int getLockSuccess() const { return *lockSuccess; }
  unsigned long long int getLockRetry() const { return *lockRetry; }
  unsigned long long int getRetryCounterOverflow() const {
    return *retryCounterOverflow;
  }

  perNodeLockStat getPerNodeLockStats(int node) const {
    return perNodeArray[node];
  }

  __device__ void recordPerNodeLockStat(uint32_t node, bool success,
                                        int level) {
    if (node == (unsigned int)-1) {
      printf("Invalid node index: %d\n", node);
      return;
    }
    atomicAdd(&perNodeArray[node].lockAttempts, 1);
    if (success) {
      atomicAdd(&perNodeArray[node].lockSuccesses, 1);
    }
    perNodeArray[node].level = level + 1; // 0 denotes untouched node
  }

  void printStats(uint32_t totalNodes) const {
    std::cout << "Total splits: " << getSplitCount() << "\n"
              << "Total merges: " << getMergeCount() << "\n"
              << "Avg Height: " << getAvgHeight() << "\n"
              << "Max height: " << getMaxHeight() << "\n"
              << "Observed Height: " << getObservedHeight() << "\n"
              << "Total lateral movement: " << getLateralMovement() << "\n"
              << "Total downward movement: " << getDownwardMovement() << "\n"
              << "Total successful lock acquisitions: " << getLockSuccess()
              << "\n"
              << "Total lock retries: " << getLockRetry() << "\n"
              << "Retry counter overflow: " << getRetryCounterOverflow()
              << std::endl;
    int i = 0;
    std::cout << "Lateral movement:\n";
    for (i = 0; i < LATERAL_NODES; i++) {
      std::cout << "Lvl" << i << ":" << lateralMv[i] << " ";
    }
    std::cout << "\nDownward movement:\n";
    for (i = 0; i < DEPTH; i++) {
      std::cout << "Lvl" << i << ":" << downMv[i] << " ";
    }

    std::cout << "\nLogging per node stats:\n";
    for (uint32_t idx = 0; idx < totalNodes; idx++) {
      if (perNodeArray[idx].level) {
        std::cout << "Node " << idx << " ";
        std::cout << "on lvl: " << perNodeArray[idx].level << " ";
        std::cout << "Retries: " << perNodeArray[idx].lockAttempts << " ";
        std::cout << "Success: " << perNodeArray[idx].lockSuccesses << "\n";
      }
    }
  }

  void logStatsToFile(const std::string &filename) const {
    std::ofstream logfile(filename, std::ios::app);
    if (!logfile) {
      std::cerr << "Error opening log file: " << filename << std::endl;
      return;
    }

    logfile << "Splits: " << getSplitCount() << ", Merges: " << getMergeCount()
            << std::endl;
    logfile.close();
  }
};

#endif // SKIPLIST_STATS_CUH
