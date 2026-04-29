/** Define all project-wide constants here */

#pragma once

#include <cstdint>
#include <string>
#include <thread>

#define HASH_TAB_SIZE 100000
#define WARP_SIZE 32
#define SLAB_NODE_SIZE WARP_SIZE
using std::string;
/** Random number generation */
static constexpr uint64_t RANDOM_SEED = 42;
static constexpr uint32_t BlockSize = 512;
/** Number of buckets in the hash table */
static const uint32_t bucket_count = 1000;
static constexpr uint64_t MAX_OPERATIONS = 1e9;
static const uint32_t SENTINEL_KEY = 0;
static const uint32_t SENTINEL_VALUE = 0;
static const uint32_t PRIME_DIVISOR = 4294967291u;
// keep 2 CPU threads free to handle the page fault by GPU
static const uint32_t NUM_THREADS = std::thread::hardware_concurrency();
static const uint64_t SENTINEL_PAIR = 0;
static const uint32_t MAX_PROBING_RETRIES = (1 << 30);
static const uint32_t GPU_BATCH_SIZE = 690000000;
/** Tombstone to identify deleted keys */
static const uint32_t TOMBSTONE_KEY = UINT32_MAX;
static const uint32_t TOMBSTONE_VALUE = UINT32_MAX;
/** Double the size of hash table when size reaches loadFactor*capacity */
const float LOAD_FACTOR = 0.9F;

// HETERODS: constant for prefetching
// prefetching distance, tune by running experiments
const int PDIST = 4;
const int BLOCKDIMX = blockDim.x;

// environment variable name for trace location
static const string PROJECT_ROOT_DIR = "TRACE_ROOT";
const uint32_t BATCH_SIZE = 1073741824; //1G, 2G = 2147483648;
const uint64_t PROBE_RETRIES = (~uint64_t(0) - 1024);
const uint32_t SL_BATCH_SIZE = 500000000; //536870912 5e8, 512M
