#pragma once

#include <atomic>
#include <cstdint>

#include "constants.h"

using std::atomic_uint32_t;
using std::atomic_uint64_t;

/** Key-Value type for the GPU map */
struct key_value_s {
  uint32_t key;
  uint32_t value;
};
using KeyValue = struct key_value_s;

using KVPair = atomic_uint64_t;

struct Node_s {
  KeyValue arr[SLAB_NODE_SIZE];
  Node_s *next;
} slabNode_default = {{0}, nullptr};
using SlabNode = struct Node_s;

struct d_HtContent_s {
  SlabNode *node;
  bool required;
} d_HtContent_default = {nullptr, false};
using d_HtContent = struct d_HtContent_s;

/** Node type for CPU hash table */
struct cpu_hash_node_s {
  atomic_uint32_t key;
  atomic_uint32_t value;
};
using CPUNode = struct cpu_hash_node_s;

enum TRACE_PATTERN {
  SPARSE_UNIQUE = 0,
  SPARSE_REPEAT = 1,
  DENSE_UNIQUE = 2,
  DENSE_REPEAT = 3,
  PHASE_REPETITION = 4,
  MONOTONIC_INCREASE = 5,
  MONOTONIC_DECREASE = 6
};

/** mode=0 implies partition based on offload percentage, mode=1 implies
    partition based on maximum GPU memory capacity, and mode=2 implies partition
    based on equal time-consuming chunks */
enum class PartitionMode {
  OFFLOAD = 0,
  GPU_MEM_CAPACITY = OFFLOAD + 1,
  EQUAL_TIME_CHUNKS = GPU_MEM_CAPACITY + 1,
  MAX_MODES = EQUAL_TIME_CHUNKS + 1
};

using HoHCpu = struct HoHCpu {
#if defined(ATOMIC_WRAPPER)
  uint32_t range;
  uint32_t unique_keys;
  KeyValue *inner_hashtable;
#else
  atomic_uint32_t range;
  atomic_uint32_t unique_keys;
  CPUNode *inner_hashtable;
#endif
};

using HoHGpu = struct HoHGpu {
  uint32_t range;
  uint32_t unique_keys;
  KeyValue *inner_hashtable;
};

using HoHGpuSN = struct HoHGpuSN {
  HoHGpu data[32];
  HoHGpuSN *next;
};

using HoHGpuSNmain = struct HoHGpuSNmain {
  HoHGpu data[524288];
  HoHGpuSN *next;
};

// Definition of Hashtable of Hashtable structure
// Will later merge to a single definition used across CPU and GPU
using HTNode = struct HoHStruct {
  uint32_t range;
  uint32_t unique_keys;
  KeyValue *inner_hashtable;
};
