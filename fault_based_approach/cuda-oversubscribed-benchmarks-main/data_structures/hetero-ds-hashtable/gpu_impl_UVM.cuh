#include "include/constants.h"
#include "include/datatypes.h"
#include "include/functions.h"
#include <algorithm>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

__global__ void initialize_hash(KeyValue *hashtable, uint64_t capacity) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < capacity) {
    hashtable[tid].key = SENTINEL_KEY;
    hashtable[tid].value = SENTINEL_VALUE;
  }
  if (tid == 0)
    printf("Capacity is : %ld\n", capacity);
}

__device__ uint32_t lookupgpuimpl(KeyValue *hashTable, uint32_t key, int lane,
                                  uint64_t capacity, uint32_t rand_int,
                                  uint64_t primeDH) {
  uint64_t pos = lane;
  uint64_t i = 0;
  uint64_t probingAttempt = 0;
#if defined(HH)
  i = (hashFuncIdentity(key) + lane) % capacity;
#elif defined(SH)
  i = (hashFuncSH(key, rand_int) + lane) % capacity;
#else
  i = (hashFuncWC(key) + lane) % capacity;
#endif
#if defined(DOUBLE_HASHING)
  uint64_t secondary_hash = primeDH - (key % primeDH);
#endif
  while (i < ~uint32_t(0)) {
    uint32_t key_p = hashTable[i].key;
    const bool hit = (key_p == key);
    const auto hitmask = __ballot_sync(0xFFFFFFFF, hit);
    if (hitmask) {
      const auto leader = __ffs(hitmask) - 1;
      const auto leader_index = __shfl_sync(0xFFFFFFFF, i, leader, 32);
      // printf("The value is :%d\n", hashTable[leader_index].value);
      return hashTable[leader_index].value;
    }
    if (__any_sync(0xFFFFFFFF, (key_p == SENTINEL_KEY))) {
      return 0;
    }
    pos += 32;
    probingAttempt++;
#if defined(QUADRATIC_PROBING)
    i = ((i * i) + 32) % capacity;
#elif defined(DOUBLE_HASHING)
    i = (i + secondary_hash) % capacity;
#else
    i = (i + 32) % capacity;
#endif
    // if (pos >= 1e+70)
    if (probingAttempt >= PROBE_RETRIES)
      i = ~uint32_t(0);
  }
  return 0;
}
__global__ void batch_lookup_gpu_kernel(KeyValue *hashtable, uint32_t *d_keys,
                                        uint32_t *searched_value,
                                        uint64_t num_queries, uint64_t capacity,
                                        uint32_t rand_int, uint64_t primeDH) {
  uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t lane = tid & 0x1F;
  uint32_t wid = tid >> 5;
  if (wid < num_queries) {
    // if (tid == 0) {
    //   printf("Key is %d and queries is %ld\n", d_keys[tid], num_queries);
    // }
    uint32_t value;
    value = lookupgpuimpl(hashtable, d_keys[wid], lane, capacity, rand_int,
                          primeDH);
    if (lane == 0) {
      if (value != 0) {
        // printf("Values are: %d and wid is: %d\n", value, wid);
        searched_value[wid] = value;
        // printf("Searched value contains: %d\n", searched_value[wid]);
      }
    }
  }
}

float batch_lookup_gpu(KeyValue *mHashTable, uint32_t *search_queries,
                       uint32_t *searched_values, uint64_t num_queries,
                       uint64_t capacity) {
  uint32_t num_blocks = SDIV((num_queries << 5), BlockSize);
#if defined(GPU_DEBUG) || defined(DEBUG)
  printf("HASHTABLE: launching search kernel on GPU\n");
#endif
  uint32_t rand_int = 0;
#if defined(SH)
  std::mt19937 rng(0);
  rand_int = rng() % PRIME_DIVISOR;
#endif
#ifdef SEARCH_KEY
  printf("Keys to search:\n");
  for (int index = 0; index < num_queries; index++)
    printf("%u\t", search_queries[index]);
  printf("\n");
#endif // SEARCH_KEY

#if defined(UVM_BATCH_SEARCH)
  // hashtable size in bytes based on capacity
  uint64_t htSizeInBytes = capacity * sizeof(KeyValue);
  uint64_t searchArraySize = num_queries * sizeof(uint32_t);
  uint64_t batchSize = BATCH_SIZE;
  uint64_t totalSearchBatches;
  uint64_t availableMemory = 0;
  // If hashtable overflows, batch size is 1GB
  if (htSizeInBytes >= totalAvailableMemory) {
    availableMemory = totalAvailableMemory - htSizeInBytes;
    if (availableMemory < searchArraySize)
      batchSize = (availableMemory >> 1);
  }
  totalSearchBatches = searchArraySize / batchSize;
  printf("SEARCH: Batch size: %ld, AvailableMemory: %ld, BatchCount:%ld\n",
         batchSize, availableMemory, totalSearchBatches);
#endif // UVM_BATCH_SEARCH

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // printf("No of blocks in search is %d\n", num_blocks);

#if defined(UVM_BATCH_SEARCH)
  // float per_batch_insert_time = 0.0f;
  uint32_t *searchArr;
  uint32_t *searchVal;
  uint64_t totalElementInBatch = batchSize / sizeof(uint32_t);
  UVM_ALLOC_BUF(uint32_t, searchArr, totalElementInBatch * sizeof(uint32_t));
  UVM_ALLOC_BUF(uint32_t, searchVal, totalElementInBatch * sizeof(uint32_t));

#if defined(UVM_PREFETCH_HINT)
  cudaCheckErrorMacro(
      cudaMemPrefetchAsync(searchArr, (totalElementInBatch * sizeof(uint32_t)),
                           0),
      "Prefetching hint failure for search Array\n");
#endif
  num_blocks = SDIV((totalElementInBatch << 5), BlockSize);
  for (int iteration = 0; iteration < totalSearchBatches; iteration++) {
    memcpy(searchArr, search_queries + (iteration * totalElementInBatch),
           totalElementInBatch * sizeof(uint32_t));
    batch_lookup_gpu_kernel<<<num_blocks, BlockSize>>>(
        mHashTable, searchArr, searchVal, totalElementInBatch, capacity,
        rand_int, smallerPrimeGPU);
    cudaDeviceSynchronize();
    cudaCheckErrorMacro(cudaGetLastError(),
                        "Search kernel failure for batch " + iteration);
    // copy the result in output array
    memcpy(searched_values + (iteration * totalElementInBatch), searchVal,
           totalElementInBatch * sizeof(uint32_t));
  }
  // If total insert elements not multiple of batchSize
  uint64_t additionalElements =
      (searchArraySize % batchSize) / sizeof(uint32_t);
  if (additionalElements) {
    num_blocks = SDIV((additionalElements << 5), BlockSize);
    memcpy(searchArr,
           search_queries + (totalSearchBatches * totalElementInBatch),
           additionalElements * sizeof(uint32_t));
    batch_lookup_gpu_kernel<<<num_blocks, BlockSize>>>(
        mHashTable, searchArr, searchVal, additionalElements, capacity,
        rand_int, smallerPrimeGPU);
    cudaDeviceSynchronize();
    cudaCheckErrorMacro(cudaGetLastError(),
                        "Search kernel failure for additional elements");
    memcpy(searched_values + (totalSearchBatches * totalElementInBatch),
           searchVal, additionalElements * sizeof(uint32_t));
  }
  cudaFree(searchArr);
  cudaFree(searchVal);
#else
#if defined(UVM_MEM_ADVISE_SA)
  cudaCheckErrorMacro(cudaMemAdvise(search_queries,
                                    num_queries * sizeof(uint32_t),
                                    cudaMemAdviseSetAccessedBy, 0),
                      "Search Queries: Memadvise hint failure\n");
#endif
  batch_lookup_gpu_kernel<<<num_blocks, BlockSize>>>(
      mHashTable, search_queries, searched_values, num_queries, capacity,
      rand_int, smallerPrimeGPU);
#endif // UVM_BATCH_SEARCH

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaCheckErrorMacro(cudaGetLastError(), "Search kernel failure");
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Time for lookup:%f ms \n", elapsedTime);
#if defined(GPU_DEUBG)
  printf("HASHTABLE: search kernel complete\n");
#endif
  // for(int i=0;i<num_queries;i++)
  //   printf("value of searched bool is: %d\n",d_searched_value[i]);
  return elapsedTime;
}

__device__ uint32_t *insertgpuimpl(uint32_t key, int lane, KeyValue *hashtable,
                                   uint64_t capacity, uint64_t maxCollisions,
                                   uint32_t rand_int, uint64_t primeDH) {

  uint64_t pos = lane;
  uint64_t i = 0;
  uint64_t probingAttempt = 0;
#if defined(HH)
  i = (hashFuncIdentity(key) + lane) % capacity;
#elif defined(SH)
  i = (hashFuncSH(key, rand_int) + lane) % capacity;
#else
  i = (hashFuncWC(key) + lane) % capacity;
#endif
#if defined(DOUBLE_HASHING)
  uint64_t secondary_hash = primeDH - (key % primeDH);
#endif // DOUBLE_HASHING
#if defined(GPU_DEBUG)
  assert(secondary_hash > 0);
#endif // GPU_DEBUG                                                            \

#ifdef STATS
  __shared__ uint32_t c;
  c = 0;
#endif
  while (i < ~uint64_t(0)) {
    // Simplified condition
    // while (probingAttempt < capacity) {
    uint32_t key_p = hashtable[i].key;
    const bool hit = (key_p == key);
    // printf("Value of hit is %d\n", hit);
    const auto hitmask = __ballot_sync(0xFFFFFFFF, hit);
    if (hitmask) {
      const auto leader = __ffs(hitmask) - 1;
      const auto leader_index = __shfl_sync(0xFFFFFFFF, i, leader, 32);
// printf("The leader index in hitmask %d\n", leader_index);
#ifdef STATS
      if (lane == leader)
        printf("The collision count(from hit) is %d\n", c);
#endif
#if defined(INSERT_DEBUG)
      printf("The leader index in hitmask %ld and key is: %d\n", leader_index,
             hashtable[leader_index].key);
#endif // INSERT_DEBUG
      return (&hashtable[leader_index].value);
    }
    auto empty_mask = __ballot_sync(0xFFFFFFFF, (key_p == SENTINEL_KEY));
    bool success = false;
    bool duplicate = false;

    while (empty_mask) {
      const auto leader = __ffs(empty_mask) - 1;
      // if (lane == leader)
      //   printf("The hashed place is: %ld\n", i);

      if (lane == leader) {
        const auto old = atomicCAS(&(hashtable[i].key), key_p, key);
        // printf("Value of i is %ld\n",i);
        success = (old == key_p);
        duplicate = (old == key);
      }

      if (__any_sync(0xFFFFFFFF, duplicate)) {

        const auto leader_index = __shfl_sync(0xFFFFFFFF, i, leader, 32);
// printf("In any sync: the leader index is: %d and key is: %d\n",
//        leader_index, hashtable[leader_index].key);
#if defined(INSERT_DEBUG)
        printf("Duplicate key is: %u\n", hashtable[leader_index].key);
#endif // INSERT_DEBUG

#ifdef STATS
        if (lane == leader)
          printf("The collision count(from duplicate) is %d\n", c);
#endif
        return (&hashtable[leader_index].value);
      }

      if (__any_sync(0xFFFFFFFF, success)) {
        const auto leader_index = __shfl_sync(0xFFFFFFFF, i, leader, 32);
#if defined(INSERT_DEBUG)
        printf("The leader index in success %ld and key is: %d\n", leader_index,
               hashtable[leader_index].key);
#endif // INSERT_DEBUG
#ifdef STATS
        if (lane == leader)
          printf("The collision count(from success) is %d\n", c);
#endif
        return (&hashtable[leader_index].value);
      }
#ifdef STATS
      if (lane == leader)
        atomicInc(&c, maxCollisions);
#endif
      empty_mask ^= 1UL << leader;
    }
    pos += 32;
    probingAttempt++;
#if defined(QUADRATIC_PROBING)
    i = ((i * i) + 32) % capacity;
#elif defined(DOUBLE_HASHING)
    i = (i + secondary_hash) % capacity;
#else
    i = (i + 32) % capacity;
#endif
    // if (pos >= 1e70)
    if (probingAttempt > capacity)
      i = ~uint64_t(0);
  }
  return NULL;
}

// PROSPAR: kernel to support the batch insert
__global__ void batch_insert_gpu_kernel(KeyValue *hashtable,
                                        KeyValue *kvs_array, uint64_t gpu_ins,
                                        uint64_t capacity,
                                        uint64_t maxCollisions,
                                        uint32_t rand_int, uint64_t primeDH) {
  uint64_t tid = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;
  uint64_t wid = tid >> 5;
  uint32_t lane = tid & 0x1F;
#if defined(INSERT_DEBUG)
  if (lane == 0) {
    printf("Wid %lu inserting key: %u\n", wid, kvs_array[wid].key);
  }
#endif // INSERT_DEBUG
  if (wid < gpu_ins) {
    uint32_t *value_addr =
        insertgpuimpl(kvs_array[wid].key, lane, hashtable, capacity,
                      maxCollisions, rand_int, primeDH);
    if (lane == 0 && value_addr != NULL)
      *value_addr = kvs_array[wid].value;
    else if (lane == 0)
      printf("Key %u not inserted\n",kvs_array[wid].key);
  }
}

// PROSPAR: added to support batch insert
float batch_insert_gpu_UVM(KeyValue *mHashtable, KeyValue *h_kvpairs,
                           uint64_t gpu_ins, uint64_t capacity) {

  // The array to store key value pair on the
  // memory allocation on device and host
  float elapsedTime;
  uint64_t num_blocks = SDIV((gpu_ins << 5), BlockSize);
  uint32_t rand_int = 0;
#if defined(SH)
  std::mt19937 rng(0);
  rand_int = rng() % PRIME_DIVISOR;
#endif

  // printf("No of blocks: %d\n",num_blocks);
#ifdef DEBUG
  printf("HASHTABLE: launching kernel for bulk insert\n");
  // for (int index = 0; index < gpu_size; index++)
  //   printf("%d ", h_kvpairs[index].key);
#endif
  uint64_t maxCollisions = gpu_ins * (gpu_ins - 1) / 2 + 1;

#if defined(UVM_BATCH_INSERT)
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // hashtable size in bytes based on capacity
  uint64_t htSizeInBytes = capacity * sizeof(KeyValue);
  uint64_t insertArraySize = gpu_ins * sizeof(KeyValue);
  uint64_t batchSize = BATCH_SIZE;
  uint64_t totalInsertBatches;
  uint64_t availableMemory = 0;
  // If hashtable overflows, insertion in batches of 1GB size
  if (htSizeInBytes > totalAvailableMemory) {
    availableMemory = totalAvailableMemory - htSizeInBytes;
    if (availableMemory < insertArraySize)
      batchSize = availableMemory >> 1;
  }
  totalInsertBatches = insertArraySize / batchSize;
  printf("INSERT: Batch size: %ld, AvailableMemory: %ld, BatchCount:%ld\n",
         batchSize, availableMemory, totalInsertBatches);

  KeyValue *insertArr;
  uint64_t totalElementInBatch = batchSize / sizeof(KeyValue);
  UVM_ALLOC_BUF(KeyValue, insertArr, totalElementInBatch * sizeof(KeyValue));

  // Other memadvise options and prefetching on insertion array
  // does not help
#if defined(UVM_MEM_ADVISE_SA)
  // Pranjal: we can remove this as well.
  cudaCheckErrorMacro(cudaMemAdvise(insertArr,
                                    totalElementInBatch * sizeof(KeyValue),
                                    cudaMemAdviseSetAccessedBy, 0),
                      "Mem advise hint failed for insert array\n");
#endif // UVM_MEM_ADVISE_SA
  num_blocks = SDIV((totalElementInBatch << 5), BlockSize);
  for (int iteration = 0; iteration < totalInsertBatches; iteration++) {
    memcpy(insertArr, h_kvpairs + (iteration * totalElementInBatch),
           totalElementInBatch * sizeof(KeyValue));
    batch_insert_gpu_kernel<<<num_blocks, BlockSize>>>(
        mHashtable, insertArr, totalElementInBatch, capacity, maxCollisions,
        rand_int, smallerPrimeGPU);
    cudaDeviceSynchronize();
    cudaCheckErrorMacro(cudaGetLastError(), "Batch insert kernel failure");
  }
  // If total insert elements not multiple of batchSize
  uint64_t additionalElements =
      (insertArraySize % batchSize) / sizeof(KeyValue);
  cout << "Additional elements: " << additionalElements << "\n";
  if (additionalElements) {
    num_blocks = SDIV((additionalElements << 5), BlockSize);
    memcpy(insertArr, h_kvpairs + (totalInsertBatches * totalElementInBatch),
           additionalElements * sizeof(KeyValue));
    batch_insert_gpu_kernel<<<num_blocks, BlockSize>>>(
        mHashtable, insertArr, additionalElements, capacity, maxCollisions,
        rand_int, smallerPrimeGPU);
    cudaDeviceSynchronize();
    cudaCheckErrorMacro(cudaGetLastError(),
                        "Insert kernel failure for additional elements");
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaCheckErrorMacro(cudaFree(insertArr),
                      "Unable to free the batch insert array");
  cudaEventElapsedTime(&elapsedTime, start, stop);
#else
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
#if defined(UVM_MEM_ADVISE_SA)
  cudaCheckErrorMacro(cudaMemAdvise(h_kvpairs, gpu_ins * sizeof(KeyValue),
                                    cudaMemAdviseSetAccessedBy, 0),
                      "Memadvise hint failed for insertion KV pairs\n");
#endif
  batch_insert_gpu_kernel<<<num_blocks, BlockSize>>>(
      mHashtable, h_kvpairs, gpu_ins, capacity, maxCollisions, rand_int,
      smallerPrimeGPU);
  // cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaCheckErrorMacro(cudaGetLastError(), "Insert kernel failure");
  cudaEventElapsedTime(&elapsedTime, start, stop);

#endif // UVM_BATCH_INSERT

  // printf("Insert kernel finished\n");

  printf("Time for insertion:%f ms \n", elapsedTime);
#if defined(GPU_DEBUG)
  printf("HASHTABLE: GPU-batch insertion successfull\n");
#endif
#if defined(GPU_DEBUG)
  printf("Deallocation of memory successful\n");
#endif
  // printf("Elapsed Time: %f\n", elapsedTime);
  return elapsedTime;
}

__device__ bool batchdeleteimpl(KeyValue *hashTable, uint32_t key, int lane,
                                uint64_t capacity, uint32_t rand_int,
                                uint64_t primeDH) {

  // printf("The key here is: %d\n", key);

  uint32_t pos = lane;
  uint64_t i = 0;
  uint64_t probingAttempt = 0;
#if defined(HH)
  i = (hashFuncIdentity(key) + lane) % capacity;
#elif defined(SH)
  i = (hashFuncSH(key, rand_int) + lane) % capacity;
#else
  i = (hashFuncWC(key) + lane) % capacity;
#endif
#if defined(DOUBLE_HASHING)
  uint64_t secondary_hash = primeDH - (key % primeDH);
#endif
  while (i < ~uint32_t(0)) {
    uint32_t key_p = hashTable[i].key;
    bool hit = (key_p == key);
    uint32_t hitmask = __ballot_sync(0xFFFFFFFF, hit);
    if (hitmask) {
      uint32_t leader = __ffs(hitmask) - 1;
      uint32_t leader_index = __shfl_sync(0xFFFFFFFF, i, leader, 32);
#ifdef GPU_DEBUG
      printf("The deleted value is %d\n", hashTable[leader_index].value);
#endif
      // TODO: discuss LD/ST should be replaced by atomic LD/ST?
      if (atomicCAS(&(hashTable[leader_index].key), key, TOMBSTONE_KEY) ==
          key) {
        hashTable[leader_index].value = TOMBSTONE_VALUE;
      }
      return true;
    }
    if (__any_sync(0xFFFFFFFF, (key_p == SENTINEL_KEY))) {
      return false;
    }
    pos += 32;
    probingAttempt++;
#if defined(QUADRATIC_PROBING)
    i = ((i * i) + 32) % capacity;
#elif defined(DOUBLE_HASHING)
    i = (i + secondary_hash) % capacity;
#else
    i = (i + 32) % capacity;
#endif
    // if (pos >= 1e+6)
    if (probingAttempt >= PROBE_RETRIES)
      i = ~uint32_t(0);
  }
  return false;
}

// PROSPAR: function passed unit testing
__global__ void batch_delete_gpu_kernel(KeyValue *hashTable, uint32_t *keys,
                                        bool *deletedValues,
                                        uint64_t num_queries, uint64_t capacity,
                                        uint32_t rand_int, uint64_t primeDH) {
  uint64_t tid = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;
  int wid = tid >> 5;
  int lane = tid & 0x1F;

  if (wid < num_queries) {
    bool del;
    // printf("The tid is %d\n", tid);
    del = batchdeleteimpl(hashTable, keys[wid], lane, capacity, rand_int,
                          primeDH);
    if (lane == 0) {
      if (del == true)
        deletedValues[wid] = del;
    }
  }
}

float batch_delete_gpu(KeyValue *mHashtable, uint32_t *del_keys,
                       bool *deleted_result, uint32_t num_queries,
                       uint64_t capacity) {
  int num_blocks = SDIV((num_queries << 5), BlockSize);
  uint32_t rand_int = 0;
  float elapsedTime = 0.0f;

#if defined(SH)
  std::mt19937 rng(0);
  rand_int = rng() % PRIME_DIVISOR;
#endif
  // printf("The key to be deleted is:%d\n",del_keys[0]);

#if defined(GPU_DEBUG) || defined(DEBUG)
  printf("HASHTABLE: batch deletion call\n");
#endif
#ifdef DELETE_KEY
  printf("Keys to delete:\n");
  for (int index = 0; index < num_queries; index++)
    printf("%u ", h_kvs[index]);
  printf("\n");
#endif // DELETE_KEY

#if defined(UVM_BATCH_DELETE)
  // hashtable size in bytes based on capacity
  uint64_t htSizeInBytes = capacity * sizeof(KeyValue);
  uint64_t deleteArraySize = num_queries * sizeof(uint32_t);
  uint64_t batchSize = BATCH_SIZE;
  uint64_t totalDeleteBatches;
  uint64_t availableMemory = 0;
  // If hashtable overflows, insertion in batches of 1GB size
  if (htSizeInBytes >= totalAvailableMemory) {
    availableMemory = totalAvailableMemory - htSizeInBytes;
    if (availableMemory < deleteArraySize)
      batchSize = (availableMemory >> 1);
  }
  totalDeleteBatches = deleteArraySize / batchSize;
  printf("DELETE: Batch size: %ld, AvailableMemory: %ld, BatchCount:%ld\n",
         batchSize, availableMemory, totalDeleteBatches);
#endif // UVM_BATCH_DELETE

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
#if defined(UVM_BATCH_DELETE)
  // float per_batch_insert_time = 0.0f;
  uint32_t *delArr;
  bool *delStatus;
  uint64_t totalElementInBatch = batchSize / sizeof(uint32_t);
  UVM_ALLOC_BUF(uint32_t, delArr, totalElementInBatch * sizeof(uint32_t));
  UVM_ALLOC_BUF(bool, delStatus, totalElementInBatch * sizeof(bool));

  num_blocks = SDIV((totalElementInBatch << 5), BlockSize);
#if defined(UVM_MEM_ADVISE_SA)
  // Pranjal: can remove this bit as well.
  cudaCheckErrorMacro(cudaMemAdvise(delArr,
                                    totalElementInBatch * sizeof(uint32_t),
                                    cudaMemAdviseSetAccessedBy, 0),
                      "MemAdvise hint failure for deletion array\n");
#endif
#if defined(UVM_PREFETCH_HINT)
  cudaCheckErrorMacro(
      cudaMemPrefetchAsync(delArr, (totalElementInBatch * sizeof(uint32_t)), 0),
      "Prefetching hint failure for delete Array\n");
#endif
  for (int iteration = 0; iteration < totalDeleteBatches; iteration++) {
    memcpy(delArr, del_keys + (iteration * totalElementInBatch),
           totalElementInBatch * sizeof(uint32_t));
    batch_delete_gpu_kernel<<<num_blocks, BlockSize>>>(
        mHashtable, delArr, delStatus, totalElementInBatch, capacity, rand_int,
        smallerPrimeGPU);
    cudaDeviceSynchronize();
    cudaCheckErrorMacro(cudaGetLastError(),
                        "Delete kernel failure for batch " + iteration);
    memcpy(deleted_result + (totalElementInBatch * iteration), delStatus,
           totalElementInBatch * sizeof(bool));
  }
  // If total insert elements not multiple of batchSize
  uint64_t additionalElements =
      (deleteArraySize % batchSize) / sizeof(uint32_t);
  if (additionalElements) {
    num_blocks = SDIV((additionalElements << 5), BlockSize);
    memcpy(delArr, del_keys + (totalDeleteBatches * totalElementInBatch),
           additionalElements * sizeof(uint32_t));
    batch_delete_gpu_kernel<<<num_blocks, BlockSize>>>(
        mHashtable, del_keys, delStatus, additionalElements, capacity, rand_int,
        smallerPrimeGPU);
    cudaDeviceSynchronize();
    cudaCheckErrorMacro(cudaGetLastError(),
                        "Delete kernel failure for additional elements");
    memcpy(deleted_result + (additionalElements * totalDeleteBatches),
           delStatus, additionalElements * sizeof(bool));
  }
  cudaFree(delArr);
  cudaFree(delStatus);
#else
#if defined(UVM_MEM_ADVISE_SR)
  cudaCheckErrorMacro(cudaMemAdvise(del_keys, num_queries * sizeof(uint32_t),
                                    cudaMemAdviseSetReadMostly, 0),
                      "MemAdvise hint failure for delete keys\n");
#endif
  batch_delete_gpu_kernel<<<num_blocks, BlockSize>>>(
      mHashtable, del_keys, deleted_result, num_queries, capacity, rand_int,
      smallerPrimeGPU);

#endif // UVM_BATCH_DELETE

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaCheckErrorMacro(cudaGetLastError(), "Delete kernel failure");
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Time for deletion:%f ms \n", elapsedTime);
#if defined(GPU_DEBUG) || defined(DEBUG)
  printf("HASHTABLE: delete GPU kernel call successfull\n");
#endif
  // cudaDeviceSynchronize();
#if defined(GPU_DEBUG) || defined(DEBUG)
  printf("HASHTABLE: deletion successfull, memory reclaimed from GPU\n");
#endif

  return elapsedTime;
}

__global__ void print_Kernel(KeyValue *hashTable, uint64_t capacity) {
  uint64_t tid = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;
  if (tid == 0) {
    uint64_t count = 0;
    for (uint64_t i = 0; i < capacity; i++) {
      if (hashTable[i].key != SENTINEL_KEY /*&&
          hashTable[i].value != SENTINEL_VALUE*/) {
        count++;
        printf("K: %d  V: %d at slot %ld\n", hashTable[i].key,
               hashTable[i].value, i);
      }
    }
    printf("count is %ld\n", count);
  }
}

void print_gpuHashTable(KeyValue *mHashTable, uint64_t capacity) {
  printf("GPU hashtable\n");
  print_Kernel<<<1, BlockSize>>>(mHashTable, capacity);
  cudaDeviceSynchronize();
}

void Key_Check_GPU(KeyValue *hashTable, uint64_t totalInsert, KeyValue *Kvs,
                   uint64_t gpuHTSize) {
  uint64_t capacity = gpuHTSize;
  cout << "HT Capacity: " << capacity << "\n";
  uint32_t *h_keys_ht = (uint32_t *)malloc(sizeof(uint32_t) * capacity);
  uint64_t h_keys_counter = 0;
  for (uint64_t i = 0; i < capacity; i++) {
    // a valid insert will have key  > 0
    if (hashTable[i].key)
      h_keys_ht[h_keys_counter++] = hashTable[i].key;
  }
  cout << "Total inserts : " << totalInsert
       << " Keys inserted in hashtable : " << h_keys_counter << "\n";
  // assert(h_keys_counter == totalInsert);
  printf("Keys in hashTable: %lu\n", h_keys_counter);
  uint32_t *ins_keys = (uint32_t *)malloc(sizeof(uint32_t) * totalInsert);
  for (uint64_t i = 0; i < totalInsert; i++) {
    ins_keys[i] = Kvs[i].key;
  }
  std::sort(h_keys_ht, h_keys_ht + h_keys_counter);
  std::sort(ins_keys, ins_keys + totalInsert);
  uint32_t unique_size = 1;
  for (uint32_t i = 1; i < totalInsert; i++) {
    if (ins_keys[i] != ins_keys[i - 1])
      unique_size++;
  }
  printf("unique size is: %u\n", unique_size);
  uint32_t *ins_keys_copy = new uint32_t[unique_size];
  uint64_t ins_keys_counter = 0;
  ins_keys_copy[ins_keys_counter++] = ins_keys[0];
  for (uint32_t i = 1; i < totalInsert; i++) {
    if (ins_keys[i] != ins_keys[i - 1])
      ins_keys_copy[ins_keys_counter++] = ins_keys[i];
  }
  uint64_t total_mismatches = 0;
  for (uint64_t i = 0; i < unique_size; i++) {
    if (h_keys_ht[i] != ins_keys_copy[i])
      total_mismatches++;
  }
  printf("Total mismatches: %lu\n", total_mismatches);
}
