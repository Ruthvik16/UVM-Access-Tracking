/** This file includes global definitions that are shared across drivers but are
    mutable. */

#pragma once

#include "constants.h"
#include "datatypes.h"
#include <cstdint>
#include <cstring>

uint64_t NUM_OPS = 1e8;         // Total operations
uint64_t PERCENT_INSERT = 100;  // Percentage of insert
uint64_t PERCENT_DELETE = 0;    // Percentage of delete
uint32_t PERCENT_OFFLOAD = 100; // Percentage of operations executed by the GPU

uint32_t USE_TRACE_FILE = 1; // Read the trace file for operation list
// Should we prepopulate a portion (in percentage) of the  hash table?
float PRE_POPULATE_HT_PERCENT = 0;

int runs = 2; // Total trials, one trial for warm up
int hashflag = 0;
uint32_t stride = 0;
uint32_t maximumThread = 0;

// FIXME: SB: What is the purpose of these variables?
uint32_t power_of_two = 10;
uint64_t smallerPrimeCPU = 32;
#if defined(CG)
uint64_t smallerPrimeGPU = COOP_GROUP_SIZE;
uint64_t smallerPrimeInner = COOP_GROUP_SIZE;
#else
uint64_t smallerPrimeInner = 32;
uint64_t smallerPrimeGPU = 32;
#endif
// Options to control the trace generation

uint32_t duplicateInAdd = 0;
uint32_t duplicateInRem = 0;
uint32_t duplicateInFind = 0;
uint32_t nonExistingDeleteKeysPercent = 0;
uint32_t nonExistingSearchKeysPercent = 0;
uint32_t tracePtr = 0;

/** If hash table overflows GPU global memory, then insertion is in batches of
    size 1GB. */
uint32_t cpuBatchSIZE = (1 << 30); // 1G

uint32_t gpuBatchSize = 1e8;
/** Size of a range, to limit the bitmap tracking metadata */
uint32_t rangeSize = (1 << 10);

size_t gpuGlobalMem = 0;
size_t gpuL2CacheSize = 0;

std::string addTrace, delTrace, findTrace;
PartitionMode mode = PartitionMode::OFFLOAD;

//Option for skiplist
uint32_t NUM_BLOCKS = 512;  // Number of Blocks
uint32_t BLOCK_SIZE = 256;  // Number of theads per block
uint32_t OVERSUB_RATIO = 0; // Oversubscription ratio
uint32_t NUM_CHUNKS = 1024;
uint32_t KEYS_PER_WARP = 1;   // keys per warp min:1(gfsl baseline) max:30
uint32_t WAITING_WARPS = 512; // control the warp that proceed sequentially

struct iterationTime {
  uint32_t iteration;
  float total_time;
  float total_insert_time;
  float total_delete_time;
  float total_search_time;
  float total_batch_time;
  float total_sort_time;
};
