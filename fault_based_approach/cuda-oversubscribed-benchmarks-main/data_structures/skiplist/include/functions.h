#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "constants.h"
#include "datatypes.h"
#include "global-vars.h"
#include "primes.h"

using std::atomic;
using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;
using std::filesystem::path;

// using std::chrono::duration_cast;
// using std::chrono::microseconds;

// using Time = std::chrono::steady_clock;
// using ms = std::chrono::milliseconds;
// using float_sec = std::chrono::duration<float>;
// using float_time_point = std::chrono::time_point<Time, float_sec>;

/* __device__ static int bucket_id = 0; */

/** Helper for CUDA errors */
#define cudaCheckErrorMacro(ans, msg)                                          \
  { gpuAssert((ans), msg, __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, string msg, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA ERROR: %s, Message: %s, FILE: %s, LINE: %d\n",
            cudaGetErrorString(code), msg.c_str(), file, line);
    if (abort)
      exit(code);
  }
}

#define SDIV(x, y) ((((x) + (y)) - 1) / y)

__device__ __forceinline__ uint32_t hashFuncIdentity(uint32_t key) {
  return key;
}

/** Hash function used by WarpCore */
__device__ __forceinline__ uint32_t hashFuncMurmur(uint32_t key) {
  key ^= key >> 16;
  key *= 0x85ebca6b;
  key ^= key >> 13;
  key *= 0xc2b2ae35;
  key ^= key >> 16;
  return key;
}

/** Hash function used by SlabHash */
__device__ __forceinline__ uint32_t hashFuncSH(uint32_t key,
                                               uint32_t rand_int) {
  uint32_t a = rand_int % 4294967291u;
  if (a == 0) {
    a = 1;
  }
  uint32_t b = rand_int % 4294967291u;
  key = ((a ^ key) + b) % 4294967291u;
  return key;
}

/** Should only be used by the CPU */
inline uint32_t cpuHashFuncHT(uint32_t key) {
  key = key ^ (key >> 16);
  key *= 0x85ebca6b;
  key ^= key >> 13;
  key *= 0xc2b2ae35;
  key ^= key >> 16;
  return key;
}

/** Should only be used by the CPU */
inline uint32_t cpuHashFuncModulo(uint32_t key) { return key; }

/** Should only be used by the CPU */
inline uint32_t cpuHashFuncSHModulo(uint32_t key, uint32_t rand_int) {
  uint32_t a = rand_int % 4294967291u;
  if (a == 0) {
    a = 1;
  }
  uint32_t b = rand_int % 4294967291u;
  key = ((a ^ key) + b) % 4294967291u;
  return key;
}

// TODO: Function name conflicts with cuCollections helper function
/** Elements are uint32_t */
void create_file(path pth, uint32_t *data, uint64_t size) {
  FILE *fptr = fopen(pth.string().c_str(), "wb+");
  // return total object written to file
  uint64_t totalEle = fwrite(data, sizeof(uint32_t), size, fptr);
  assert(totalEle == size);
  fclose(fptr);
}

/** Read n integer elements from file given by pth and fill in the output
   variable data */
void read_data(path pth, uint64_t n, uint32_t *data) {
  FILE *fptr = fopen(pth.string().c_str(), "rb");
  string fname = pth.string().c_str();
  if (!fptr) {
    string error_msg = "Unable to open file: " + fname;
    perror(error_msg.c_str());
  }
  int freadStatus = fread(data, sizeof(uint32_t), n, fptr);
  if (freadStatus == 0) {
    string error_string = "Unable to read the file " + fname;
    perror(error_string.c_str());
  }
  fclose(fptr);
}

// getCapacityPrime() performs better with double hashing.

/** Get GPU memory capacity. Use a prime value slightly larger than capacity. */
inline uint64_t getCapacity(uint64_t gpu_size) {
#if defined(GET_CAP)
#if defined(CG)
  const auto x = SDIV(gpu_size, GROUP_SIZE);
  const auto y = std::lower_bound(primes.begin(), primes.end(), x);
  return (y == primes.end()) ? 0 : ((*y) * GROUP_SIZE);
#else
  const auto x = SDIV(gpu_size, 32);
  const auto y = std::lower_bound(primes.begin(), primes.end(), x);
  return (y == primes.end()) ? 0 : ((*y) * 32);
#endif
#else
  const auto y = std::lower_bound(primes.begin(), primes.end(), gpu_size);
  return (y == primes.end()) ? 0 : (*y);
#endif
}

inline uint32_t linearProbing(uint32_t index) { return (index + 1); }

inline uint32_t quadraticProbing(uint32_t index, uint32_t probingAttempt) {
  return (index + (probingAttempt * probingAttempt));
}

inline uint32_t doubleHashing(uint32_t index, uint32_t key,
                              uint32_t probing_attempt,
                              uint64_t smaller_prime) {
  uint64_t newHashIndex = smaller_prime;
  //   uint32_t newHashIndex = 1 + key % (smaller_prime -2);

  return (uint32_t)(index + (probing_attempt * newHashIndex));
}

// Comparator for sorting by key
bool compareByKey(const KeyValue &a, const KeyValue &b) {
  return a.key < b.key;
}

// Comparator for uniqueness by key only
bool equalByKey(const KeyValue &a, const KeyValue &b) { return a.key == b.key; }

bool compareByRange(const KeyValue &a, const KeyValue &b) {
  return (a.key >> power_of_two) < (b.key >> power_of_two);
}

bool compareByRangeDS(const uint32_t a, const uint32_t b) { return a < b; }

//Comparator functor for thrust
struct CompareByRangeShift {
  int shift;

  // Constructor
  __host__ __device__ CompareByRangeShift(int power_of_two)
      : shift(power_of_two) {}

  // Comparison operator
  __host__ __device__ bool operator()(const KeyValue &a,
                                      const KeyValue &b) const {
    return (a.key >> shift) < (b.key >> shift);
  }
};

//Comparator functor for thrust in case of delete and search
struct CompareKeysByRangeShift {
  int shift;

  __host__ __device__ CompareKeysByRangeShift(int power_of_two)
      : shift(power_of_two) {}

  __host__ __device__ bool operator()(const unsigned int &a,
                                      const unsigned int &b) const {
    return (a >> shift) < (b >> shift);
  }
};

//comparator functor for thrust
struct CompareByKey {
  __host__ __device__ bool operator()(const KeyValue &a,
                                      const KeyValue &b) const {
    return a.key < b.key;
  }
};

struct CompareByKeyDS {
  __host__ __device__ bool operator()(const uint32_t &a,
                                      const uint32_t &b) const {
    return a < b;
  }
};

//Custom allocator for Thrust
template <typename T> struct simple_cached_allocator {
  using value_type = T;

  T *cached_ptr = nullptr;
  std::size_t cached_size = 0;

  simple_cached_allocator() {}

  T *allocate(std::size_t n) {
    if (cached_ptr && n <= cached_size) {
      // std::cout << "Reusing cached memory for " << n << " elements\n";
      return cached_ptr;
    } else {
      if (cached_ptr) {
        cudaFree(cached_ptr);
      }
      cudaMalloc(&cached_ptr, n * sizeof(T));
      cached_size = n;
      // std::cout << "Allocated new memory for " << n << " elements\n";
      return cached_ptr;
    }
  }

  void deallocate(T *ptr, std::size_t n) {
    // Don't free immediately; only free if it's not the cached block
    if (ptr != cached_ptr) {
      cudaFree(ptr);
    }
  }

  // Reset function to free cached memory and clear state
  void reset() {
    if (cached_ptr) {
      cudaFree(cached_ptr);
      cached_ptr = nullptr;
      cached_size = 0;
      // std::cout << "Cached memory has been reset\n";
    }
  }

  ~simple_cached_allocator() { reset(); }
};



class CachingAllocator {
public:
  using value_type = char;

  char *allocate(std::ptrdiff_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = cache.find(size);
    if (it != cache.end() && !it->second.empty()) {
      char *ptr = it->second.back();
      it->second.pop_back();
      return ptr;
    }

    char *ptr = nullptr;
    cudaMalloc((void **)&ptr, size);
    return ptr;
  }

  void deallocate(char *ptr, std::ptrdiff_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    cache[size].push_back(ptr);
  }

  ~CachingAllocator() {
    for (auto &pair : cache) {
      for (char *ptr : pair.second) {
        cudaFree(ptr);
      }
    }
  }

  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &pair : cache) {
      for (char *ptr : pair.second) {
        cudaFree(ptr);
      }
    }
    cache.clear();
  }

private:
  std::mutex mutex_;
  std::map<std::ptrdiff_t, std::vector<char *>> cache;
};

/** Describe flags */
void validFlagsDescription() {
  cout << "ops: total number of operations\n"
       << "add: percentage of insert operations\n"
       << "rem: percentage of delete operations\n"
       << "rns: the number of iterations\n"
       << "off: control the amount of work (percentage) to offload to GPU\n"
       << "ppp: specify whether to prepopulate the hash table\n"
       << "str: access stride for UVM based hashtable\n"
       << "mTh: specify the total threads required for strided accesses\n"
       << "fil: control how input random numbers are generated\n"
       << "hsh: specify the hash function to use\n"
       << "mod: select the mode of offload to CPU\n"
       << "rng: specify the size of the coarse-grained range\n"
       << "gbs: specify the GPU batch size for processing\n"
       << "tra: insertion trace file name\n"
       << "trr: deletion trace file name\n"
       << "trf: search trace file name\n";
}

/** Parse command line flags and initialize the variables */
int parse_args(char *arg) {
  string s = string(arg);
  string s1;
  uint64_t val;
  string fileName;
  try {
    s1 = s.substr(0, 4);
    string s2 = s.substr(5);
    if ((s1 == "-tra") || (s1 == "-trr") || (s1 == "-trf"))
      fileName = s2;
    else
      val = stol(s2);
  } catch (...) {
    cout << "Supported: " << endl;
    cout << "-*=[], where * is:" << endl;
    validFlagsDescription();
    return 1;
  }

  if (s1 == "-ops") {
    NUM_OPS = val;
  } else if (s1 == "-off") {
    PERCENT_OFFLOAD = val;
  } else if (s1 == "-rns") {
    runs = val;
  } else if (s1 == "-add") {
    PERCENT_INSERT = val;
  } else if (s1 == "-rem") {
    PERCENT_DELETE = val;
  } else if (s1 == "-fil") {
    USE_TRACE_FILE = val;
  } else if (s1 == "-ppp") {
    PRE_POPULATE_HT_PERCENT = val;
  } else if (s1 == "-str") {
    stride = val;
  } else if (s1 == "-mTh") {
    maximumThread = val;
  } else if (s1 == "-hsh") {
    hashflag = val;
  } else if (s1 == "-mod") {
    mode = static_cast<PartitionMode>(val);
  } else if (s1 == "-tra") { // insertion trace file name
    addTrace = fileName;
  } else if (s1 == "-trr") { // deletion trace file name
    delTrace = fileName;
  } else if (s1 == "-trf") { // search trace file name
    findTrace = fileName;
  } else if (s1 == "-rng") { // range size
    rangeSize = val;
  } else if (s1 == "-gbs") { // GPU batch size
    gpuBatchSize = val;
  } else if (s1 == "-blk") { // total blocks in kernel
    NUM_BLOCKS = val;
  } else if (s1 == "-siz") { // total threads per block
    BLOCK_SIZE = val;
  } else if (s1 == "-ovr") { // oversubscription ratio
    OVERSUB_RATIO = (float)val;
  } else if (s1 == "-kpw") { // keys per warp
    KEYS_PER_WARP = val;
  } else if (s1 == "-wtw") { // number of warp participate in waiting
    WAITING_WARPS = val;
  } else {
    cout << "Unsupported flag:" << s1 << "\n";
    cout << "Use the below list flags:\n";
    validFlagsDescription();
    return 1;
  }
  return 0;
}

/** Pack key-value into a 64-bit integer */
inline uint64_t packKeyValue(uint32_t key, uint32_t val) {
  return (static_cast<uint64_t>(key) << 32) |
         (static_cast<uint32_t>(val) & 0xFFFFFFFF);
}

/** Unpack a 64-bit integer into two 32-bit integers */
inline void unpackKeyValue(uint64_t value, uint32_t &key, uint32_t &val) {
  key = static_cast<uint32_t>(value >> 32);
  val = static_cast<uint32_t>(value & 0xFFFFFFFF);
}

/** Extract the key from KVPair */
inline uint32_t extractKey(uint64_t KVPair) {
  return static_cast<uint32_t>(KVPair >> 32);
}

inline int fnv_hash(int key) {
  bool prob = (key % 100) > PERCENT_OFFLOAD;
  return prob;
}

__device__ bool lookupduplimpl(uint32_t *searchArr, uint32_t key, size_t tid,
                               uint64_t size) {
  uint64_t start = 0;
  uint64_t end = size;
  while (start < end) {
    uint64_t mid = ((end - start) / 2) + start;
    uint32_t key_p = searchArr[mid];
    if (key_p == key) {
      return true;
    } else if (key_p < key) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return false;
}

__global__ void lookup_dupl(uint32_t *searchArr, uint32_t *search_queries,
                            bool *search_status, uint64_t num_queries,
                            uint64_t size) {
  size_t tid = (size_t)threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < num_queries) {
    bool present;
    present = lookupduplimpl(searchArr, search_queries[tid], tid, size);
    if (present == true) {
      search_status[tid] = present;
    }
  }
}

float lookup_duplicate(uint32_t *h_search_arr, uint32_t *keyList,
                       bool *searchStatus, uint64_t num_queries,
                       uint64_t arrSize) {
  uint64_t num_blocks = SDIV((num_queries), BlockSize);
  uint32_t *d_search_arr;
  uint32_t *d_keyList;
  bool *d_search_status;
  cudaMalloc(&d_search_arr, sizeof(uint32_t) * arrSize);

  cudaMalloc(&d_keyList, sizeof(uint32_t) * num_queries);
  cudaMalloc(&d_search_status, sizeof(uint32_t) * num_queries);
  cudaMemset(&d_search_status, 0x00, sizeof(uint32_t) * num_queries);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  cudaMemcpy(d_search_arr, h_search_arr, sizeof(uint32_t) * arrSize,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_keyList, keyList, sizeof(uint32_t) * num_queries,
             cudaMemcpyHostToDevice);
  lookup_dupl<<<num_blocks, BlockSize>>>(d_search_arr, d_keyList,
                                         d_search_status, num_queries, arrSize);
  cudaDeviceSynchronize();
  cudaMemcpy(searchStatus, d_search_status, sizeof(bool) * num_queries,
             cudaMemcpyDeviceToHost);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaFree(d_search_arr);
  cudaFree(d_keyList);
  cudaFree(d_search_status);
  return elapsedTime;
}

// Enable stats with CPU_STATS
#if CPU_STATS
static atomic<uint64_t> noCollisionKeys(0);
static atomic<uint64_t> numCollisions(0);
static atomic<uint64_t> numRetries(0);
static atomic<uint64_t> minRetriesLength(1 << 30);
static atomic<uint64_t> maxRetriesLength(0);
static atomic<uint64_t> meanRetriesLength(0);
static atomic<uint64_t> retriesHistogram8(0);
static atomic<uint64_t> retriesHistogram16(0);
static atomic<uint64_t> retriesHistogram32(0);
static atomic<uint64_t> retriesHistogram64(0);
static atomic<uint64_t> retriesHistogram128(0);
static atomic<uint64_t> retriesHistogram256(0);
static atomic<uint64_t> retriesHistogram512(0);
static atomic<uint64_t> retriesHistogram1K(0);
static atomic<uint64_t> retriesHistogram2K(0);
static atomic<uint64_t> retriesHistogram4K(0);
static atomic<uint64_t> retriesHistogram8K(0);
static atomic<uint64_t> retriesHistogram16K(0);
static atomic<uint64_t> retriesHistogram32K(0);
static atomic<uint64_t> retriesHistogram64K(0);
static atomic<uint64_t> retriesHistogram1M(0);
static atomic<uint64_t> retriesHistogram32M(0);
static atomic<uint64_t> retriesHistogram1G(0);
static atomic<uint64_t> duplicateKeys(0);
static atomic<uint64_t> insertedKeys(0);
static atomic<uint64_t> deletedKeys(0);
static atomic<uint64_t> searchedKeys(0);

void initializeStats() {
  noCollisionKeys.store(0);
  numCollisions.store(0);
  numRetries.store(0);
  minRetriesLength.store(1 << 30);
  maxRetriesLength.store(0);
  meanRetriesLength.store(0);
  retriesHistogram8.store(0);
  retriesHistogram16.store(0);
  retriesHistogram32.store(0);
  retriesHistogram64.store(0);
  retriesHistogram128.store(0);
  retriesHistogram256.store(0);
  retriesHistogram512.store(0);
  retriesHistogram1K.store(0);
  retriesHistogram2K.store(0);
  retriesHistogram4K.store(0);
  retriesHistogram8K.store(0);
  retriesHistogram16K.store(0);
  retriesHistogram32K.store(0);
  retriesHistogram64K.store(0);
  retriesHistogram1M.store(0);
  retriesHistogram32M.store(0);
  retriesHistogram1G.store(0);
  duplicateKeys.store(0);
  insertedKeys.store(0);
  deletedKeys.store(0);
  searchedKeys.store(0);
}

void printStats() {
  cout << "************** START STATS*****************"
       << "\nTotal keys without any collisions: " << noCollisionKeys.load()
       << "\nTotal Collisions: " << numCollisions.load()
       << "\nTotal numRetries: " << numRetries.load()
       << "\nMin Retries for a key: " << minRetriesLength
       << "\nMax Retries for a key: " << maxRetriesLength
       << "\nNumber of Retries(mean): " << meanRetriesLength
       << "\nRetries Histogram (key count):\n"
       << "\t< 8 retries: " << retriesHistogram8.load() << " keys\n"
       << "\t< 16 retries: " << retriesHistogram16.load() << " keys\n"
       << "\t< 32 retries: " << retriesHistogram32.load() << " keys\n"
       << "\t< 64 retries: " << retriesHistogram64.load() << " keys\n"
       << "\t< 128 retries: " << retriesHistogram128.load() << " keys\n"
       << "\t< 256 retries: " << retriesHistogram256.load() << " keys\n"
       << "\t< 512 retries: " << retriesHistogram512.load() << " keys\n"
       << "\t< 1K retries: " << retriesHistogram1K.load() << " keys\n"
       << "\t< 2K retries: " << retriesHistogram2K.load() << " keys\n"
       << "\t< 4K retries: " << retriesHistogram4K.load() << " keys\n"
       << "\t< 8K retries: " << retriesHistogram8K.load() << " keys\n"
       << "\t< 16K retries: " << retriesHistogram16K.load() << " keys\n"
       << "\t< 32K retries: " << retriesHistogram32K.load() << " keys\n"
       << "\t< 64K retries: " << retriesHistogram64K.load() << " keys\n"
       << "\t< 1M retries: " << retriesHistogram1M.load() << " keys\n"
       << "\t< 32M retries: " << retriesHistogram32M.load() << " keys\n"
       << "\t< 1G retries: " << retriesHistogram1G.load() << " keys\n"
       << "Duplicate KEYS: " << duplicateKeys.load()
       << "\nInserted Keys: " << insertedKeys.load()
       << "\nDeleted Keys: " << deletedKeys.load()
       << "\nSearched Keys: " << searchedKeys.load() << "\n";
  cout << "************** END STATS*****************\n";
}
#endif

// total key range ~ 4*10^9

// root directory of the project
path getProjectRoot() {
  if (getenv(PROJECT_ROOT_DIR.c_str()) == NULL) {
    return std::filesystem::current_path();
  }
  string projectRootStr = getenv(PROJECT_ROOT_DIR.c_str());
  path projectRootPath = projectRootStr;
  return projectRootPath;
}

// TODO: extend for the different percent of duplicate in each add,
//  delete, and search
string constructTraceFilename(string traceFileName) {
  path currPath = getProjectRoot();
  path filePathStr = currPath / traceFileName;
  return filePathStr;
}

/** check if the trace files exists and populate the insert, delete and search
    query vectors */
bool checkTraceFiles(const string &addTraceFile, const string &delTraceFile,
                     const string &findTraceFile, KeyValue *kvs_insert,
                     uint32_t *keys_del, uint32_t *keys_lookup) {
  uint64_t addOp = PERCENT_INSERT; // * (NUM_OPS / 100.0);
  uint64_t remOp = PERCENT_DELETE; //  * (NUM_OPS / 100.0);
  uint64_t searchOp = NUM_OPS - (addOp + remOp);

  // filepath for different operations
  string path_insert_keys = constructTraceFilename(addTraceFile);
  cout << "[info] Trace file for insert operations: " << path_insert_keys
       << "\n";

  if (std::filesystem::is_directory(path_insert_keys)) {
    cerr << "[error] Wrong insert trace\n";
    std::exit(EXIT_FAILURE);
  }

  bool traceStatus = std::filesystem::exists(path_insert_keys);
  if (traceStatus) {
    uint32_t *h_keys_insert = (uint32_t *)malloc(sizeof(uint32_t) * addOp);
    read_data(path_insert_keys, addOp, h_keys_insert);
    std::mt19937 mt_value(RANDOM_SEED);
    std::uniform_int_distribution<uint32_t> valueDistribution(1,
                                                              UINT32_MAX - 1);
    // Storing values in trace will increase the storage overhead and file IO
    // overhead. We store only the keys because that is what matters.
    for (uint64_t i = 0; i < addOp; i++) {
      kvs_insert[i].key = h_keys_insert[i];
      kvs_insert[i].value = valueDistribution(mt_value);
    }
    // read all values from trace, free intermediate array
    free(h_keys_insert);
  } else {
    cout << "[error] Insert trace does not exist, run trace generation "
            "script\n";
    assert(traceStatus);
  }

  // if no delete queries, path is empty
  if (remOp) {
    string path_delete_keys = constructTraceFilename(delTraceFile);
    cout << "Path for delete operation:\n";
    cout << path_delete_keys << "\n";
    traceStatus = std::filesystem::exists(path_delete_keys);
    if (traceStatus) {
      uint32_t *h_keys_delete = (uint32_t *)malloc(sizeof(uint32_t) * remOp);
      read_data(path_delete_keys, remOp, h_keys_delete);
      for (uint64_t i = 0; i < remOp; i++) {
        keys_del[i] = h_keys_delete[i];
      }
      free(h_keys_delete);
    } else {
      cout << "Delete trace does not exists, Run trace generation scripts\n";
      assert(traceStatus);
    }
  }
  // if no search queries, path is empty
  if (searchOp) {
    string path_search_keys = constructTraceFilename(findTraceFile);
    cout << "Path for search operation:\n";
    cout << path_search_keys << std::endl;
    traceStatus = std::filesystem::exists(path_search_keys);
    if (traceStatus) {
      uint32_t *h_keys_search = (uint32_t *)malloc(sizeof(uint32_t) * searchOp);
      read_data(path_search_keys, searchOp, h_keys_search);
      for (uint64_t i = 0; i < searchOp; i++) {
        keys_lookup[i] = h_keys_search[i];
      }
      free(h_keys_search);
    } else {
      cout << "Search trace does not exists, run trace generation script\n";
      assert(traceStatus);
    }
  }

  cout << "[info] Done processing trace files\n";
  return traceStatus;
}

// /** Assumes a single GPU, device 0.
//     FIXME: SB: Is this method doing what it is supposed to do?
//  */
// bool getGPUConfig() {
//   bool status = true;
//   int deviceCount = 0;
//   cudaGetDeviceCount(&deviceCount);
//   for (int i = 0; i < deviceCount; ++i) {
//     cudaDeviceProp deviceProperties;
//     cudaGetDeviceProperties(&deviceProperties, i);
//     // ... (rest of the code)
//     // total memory in bytes
//     totalAvailableMemory = deviceProperties.totalGlobalMem;
//     if (deviceProperties.l2CacheSize > 0) {
//       gpuL2Size = deviceProperties.l2CacheSize;
//     } else {
//       cout << "Device " << i << " No L2 Cache\n";
//       status = false;
//     }
//   }
//   return status;
// }
