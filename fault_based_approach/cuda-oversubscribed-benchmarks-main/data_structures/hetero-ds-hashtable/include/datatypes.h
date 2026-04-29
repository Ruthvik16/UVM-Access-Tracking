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
} slabNode_default = {{0}, NULL};
using SlabNode = struct Node_s;

struct d_HtContent_s {
  SlabNode *node;
  bool required;
} d_HtContent_default = {NULL, false};
using d_HtContent = struct d_HtContent_s;

/** Node type for CPU hash table */
struct cpu_hash_node_s {
  atomic_uint32_t key;
  atomic_uint32_t value;
};
using CPUNode = struct cpu_hash_node_s;
