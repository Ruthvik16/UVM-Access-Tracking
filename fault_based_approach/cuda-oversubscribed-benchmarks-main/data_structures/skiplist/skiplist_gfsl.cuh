#include "functions.h"
#include "skiplist_stats.cuh"
#include <cstdint>
#include <iostream>
#if defined(PROFILE_RUN)
#include "cudaProfiler.h"
#endif

#define TEAM_SIZE (32) // don't change this as of now
#define DSIZE (TEAM_SIZE - 2)

#define MAX_LEVEL (TEAM_SIZE - 1)

#define TID_NONE (TEAM_SIZE + 1) // for now
#define TID_NEXT (TEAM_SIZE - 2)
#define TID_LOCK (TEAM_SIZE - 1)

// Don't insert these keys
#define NEG_INF (0x0U)
#define POS_INF (0xffffffffU)
#define INV_ULL (0xffffffffffffffffULL)

/* alt?
#define NEG_INF (0xffffffffU)
#define POS_INF (0x7fffffffU)
*/

#define LOCK_LOCKED (0UL)
#define LOCK_ZOMBIE (1UL)
#define LOCK_UNLOCKED (2UL)

#define MERGE_THRESHOLD (DSIZE / 3)
// clang-format off
#define CHK_INVALID ((uint32_t)-1)
// clang-format on
#ifdef DEBUG
#define PRINT0(STR, ...)                                                       \
  do {                                                                         \
    if (getMyTid() == 0)                                                       \
      printf("[%d] [%s:%d] " STR,                                              \
             (blockIdx.x * blockDim.x + threadIdx.x) / 32, __func__, __LINE__, \
             __VA_ARGS__);                                                     \
    __syncwarp();                                                              \
  } while (0)
#define PRINT_TID(TID, STR, ...)                                               \
  if (getMyTid() == TID)                                                       \
  printf("[%d] [%s:%d] " STR, (blockIdx.x * blockDim.x + threadIdx.x) / 32,    \
         __func__, __LINE__, __VA_ARGS__)
#define PRINTALLKV(KV)                                                         \
  printf("[%d] [%2d] %u %u\n", (blockIdx.x * blockDim.x + threadIdx.x) / 32,   \
         getMyTid(), KV.kv.key, KV.kv.value)
#define PRINTCHK(CHK) getChunkFromIdx(CHK)->printChunk()
#define PRINTF(...) printf(__VA_ARGS__)
#else
#define PRINT0(STR, ...)
#define PRINT_TID(TID, STR, ...)
#define PRINTALLKV(KV)
#define PRINTCHK(CHK)
#define PRINTF(...)
#endif // DEBUG

#ifdef EDEBUG
#define EPRINT0(STR, ...)                                                      \
  if (getMyTid() == 0)                                                         \
  printf("[%s:%d] " STR, __func__, __LINE__, __VA_ARGS__)
#define EPRINT_TID(TID, STR, ...)                                              \
  if (getMyTid() == TID)                                                       \
  printf("[%s:%d] " STR, __func__, __LINE__, __VA_ARGS__)
#define EPRINTALLKV(KV)                                                        \
  printf("[%2d] %u %u\n", getMyTid(), KV.kv.key, KV.kv.value)
#define EPRINTCHK(CHK) getChunkFromIdx(CHK)->printChunk()
#define EPRINTF(STR, ...)                                                      \
  printf("[%s:%d] " STR, __func__, __LINE__, __VA_ARGS__)
#else
#define EPRINT0(STR, ...)
#define EPRINT_TID(TID, STR, ...)
#define EPRINTALLKV(KV)
#define EPRINTCHK(CHK)
#define EPRINTF(...)
#endif // EDEBUG

#ifdef EXDEBUG
#define EXPRINT0(STR, ...)                                                     \
  do {                                                                         \
    if (getMyTid() == 0)                                                       \
      printf("[%d] [%s:%d] " STR,                                              \
             (blockIdx.x * blockDim.x + threadIdx.x) / 32, __func__, __LINE__, \
             __VA_ARGS__);                                                     \
    __syncwarp();                                                              \
  } while (0)
#define EXPRINT_TID(TID, STR, ...)                                             \
  do {                                                                         \
    if (getMyTid() == TID)                                                     \
      printf("[%s:%d] " STR, __func__, __LINE__, __VA_ARGS__);                 \
    __syncwarp();                                                              \
  } while (0)
#define EXPRINTALLKV(KV)                                                       \
  printf("[%2d] %u %u\n", getMyTid(), KV.kv.key, KV.kv.value)
#define EXPRINTCHK(CHK) getChunkFromIdx(CHK)->printChunk()
#define EXPRINTF(STR, ...)                                                     \
  printf("[%s:%d] " STR, __func__, __LINE__, __VA_ARGS__)
#else
#define EXPRINT0(STR, ...)
#define EXPRINT_TID(TID, STR, ...)
#define EXPRINTALLKV(KV)
#define EXPRINTCHK(CHK)
#define EXPRINTF(...)
#endif // EXDEBUG

// typedef unsigned long long uint64_t; //TODO: yes or no?
typedef unsigned uint32_t;

typedef unsigned long long _ull;

typedef uint32_t K;
typedef uint32_t V;
typedef uint32_t ChunkIdx;

class GFSL;

struct __attribute__((aligned(8))) _kv {
  K key;
  V value;
} __attribute__((packed));

struct __attribute__((aligned(8))) KV {
  union {
    _kv kv;
    uint64_t raw;
  };
} __attribute__((packed));
//    __device__ KV() = default;
//     __device__ KV(uint32_t _key, uint32_t _value): key(_key), value(_value)
//     {}
//     __device__ explicit KV(uint64_t packed){
//         key = static_cast<K>(packed >> 32);
//         value = static_cast<V>(packed & 0x0ffffffffL);
//
//         // unsafe, but could be faster?
//         // key = *reinterpret_cast<uint32*>(&packed);
//         // val = *(reinterpret_cast<uint32*>(&packed) + 1);
//     }
//
//     __device__ explicit operator uint64_t() const{
//         return ((static_cast<uint64_t>(key) << 32) |
//         static_cast<uint64_t>(value));
//         // or could be unsafe and do, *reinterpret_cast<uint64_t*>(this), but
//         since it's packed it should work anyways;
//     }
//
//     // since cuda thinks that they are different
//     __device__ explicit operator unsigned long long() const{
//         return ((static_cast<uint64_t>(key) << 32) |
//         static_cast<uint64_t>(value));
//     }

// TODO: move __syncwarps from where they currently are to whenever threads
// actually need to be in sync? (whenever that is)

__device__ uint32_t getTeamMask() {
  return 0xffffffffU; // for now
}
// VIPIN: return tid based on the group size
// inline the function
__device__ int getMyTid() {
  //     printf("wtf???\n");
  return threadIdx.x %
         TEAM_SIZE; // for now - make more flexible using cooperative groups?
}

// the last thread checks for the zombie
__device__ bool isZombie(KV pair) {
  uint32_t teamMask = getTeamMask();
  return (__ballot_sync(teamMask, (getMyTid() == TID_LOCK) &&
                                      (pair.raw == LOCK_ZOMBIE)) != 0);
}

__device__ ChunkIdx getPtrFromTid(int tid, KV pair) {
  uint32_t teamMask = getTeamMask();
  return __shfl_sync(teamMask, static_cast<ChunkIdx>(pair.kv.value), tid,
                     TEAM_SIZE);
}

// __device__ ChunkIdx getKeyFromTid(int tid, KV pair) {
// TODO: is this really necessary?
//     uint32_t teamMask = getTeamMask();
//     return __shfl_sync(teamMask, static_cast<ChunkIdx>(pair.kv.key), tid,
//     TEAM_SIZE);
// }

__device__ void incrementLateralMovement(SkiplistStats *stats) {
  int tid = getMyTid();
  if (tid == 0) {
    stats->recordLateralMovement();
  }
  __syncwarp();
}

__device__ void incrementDownwardMovement(SkiplistStats *stats) {
  int tid = getMyTid();
  if (tid == 0) {
    stats->recordDownwardMovement();
  }
  __syncwarp();
}
__device__ void incrementLateralMv(int level, SkiplistStats *stats) {
  int tid = getMyTid();
  if (tid == 0 || tid == TID_NEXT) {
    stats->recordLateralMv(level);
  }
  __syncwarp();
}

__device__ void incrementDownwardMv(int level, SkiplistStats *stats) {
  int tid = getMyTid();
  if (tid == 0) {
    stats->recordDownMv(level);
  }
  __syncwarp();
}

__device__ void incrementLockAttempt(SkiplistStats *stats, uint32_t nodeId,
                                     int lvl) {
  int tid = getMyTid();
  if (tid == TID_LOCK) {
    stats->recordLockAttempt();
    stats->recordPerNodeLockStat(nodeId, false, lvl);
  }
  // __syncwarp();
}

__device__ void incrementLockSuccess(SkiplistStats *stats, uint32_t nodeId,
                                     int lvl) {
  int tid = getMyTid();
  if (tid == TID_LOCK) {
    stats->recordLockSuccess();
    stats->recordPerNodeLockStat(nodeId, true, lvl);
  }
  // __syncwarp();
}

// gets segment of traversal path from array stored across registers of threads
__device__ ChunkIdx getPathFromTid(ChunkIdx path, int tid) {
  uint32_t teamMask = getTeamMask();
  // needs to be more complex if TEAM_SIZE < 32?
  //     PRINT0("(%d)\n", tid);
  //     printf("[%2d]: %u\n", getMyTid(), path);
  return __shfl_sync(teamMask, path, tid, TEAM_SIZE);
}

// VIPIN: inline the function
__device__ int getTidForNextStep(K key, KV pair) {
  uint32_t teamMask = getTeamMask();
  int myTid = getMyTid();
  //     PRINTALLKV(pair);
  // identify index with key smaller than searched key
  bool elem = (myTid < DSIZE) && (pair.kv.key <= key);
  // identify whether the max element is smaller than key, need lateral traversal
  bool next = (myTid == TID_NEXT) && (pair.kv.key < key);

  uint32_t ballot = __ballot_sync(teamMask, (next || elem));
  //     int active = __activemask();
  //     PRINT0("active: %x ballot: %x return: %d\n", active, ballot, (TEAM_SIZE
  //     - 1) - __clz(ballot));
  if (!ballot)
    return TID_NONE;

  return (TEAM_SIZE - 1) - __clz(ballot);
}

__device__ int getTidOfDownStep(K key, KV pair) {
  uint32_t teamMask = getTeamMask();
  int myTid = getMyTid();
  bool elem =
      (myTid < DSIZE) &&
      (pair.kv.key <= key); // should be all elements except lock and next

  uint32_t ballot = __ballot_sync(teamMask, elem);
  if (!ballot) { // should not happen
    // report error here
    return TID_NONE;
  }

  return (TEAM_SIZE - 1) - __clz(ballot);
}

__device__ int getTidWithKey(K key, KV pair) {
  uint32_t teamMask = getTeamMask();
  int myTid = getMyTid();
  PRINT0("key: %u\n", key);
  //     PRINTALLKV(pair);
  bool key_found = (myTid < DSIZE) && (pair.kv.key == key);
  //     bool next = (myTid == TID_NEXT) && (pair.kv.key < key);

  uint32_t ballot = __ballot_sync(teamMask, key_found);
  if (!ballot)
    return TID_NONE;
  else
    return (TEAM_SIZE - 1) - __clz(ballot);
}

__device__ bool chunkContains(K key, KV pair) {
  if (getTidWithKey(key, pair) == TID_NONE)
    return false;
  else
    return true;
}

__device__ int occupiedSlots(KV pair) {
  uint32_t teamMask = getTeamMask();
  int myTid = getMyTid();
  bool isOccupied = (myTid < DSIZE) && (pair.kv.key != POS_INF);
  uint32_t ballot = __ballot_sync(teamMask, isOccupied);
  return __popc(ballot);
}

__device__ bool isLastChunk(KV pair) {
  uint32_t teamMask = getTeamMask();
  int myTid = getMyTid();
  return __shfl_sync(teamMask,
                     (pair.kv.value == POS_INF) && (pair.kv.key == POS_INF),
                     TID_NEXT, TEAM_SIZE);
  //     bool isLast = (myTid == TID_NEXT) && (pair.kv.value == POS_INF) &&
  //     (pair.kv.key == POS_INF); uint32_t ballot = __ballot_sync(teamMask,
  //     isLast); return (ballot != 0);
}

__device__ KV getChunkValFromLeftNeighbor(KV chunkKV) {
  uint32_t teamMask = getTeamMask();
  int myTid = getMyTid();
  return {.raw = (__shfl_up_sync(teamMask, chunkKV.raw, 1))};
}

__device__ KV getChunkValFromRightNeighbor(KV chunkKV) {
  uint32_t teamMask = getTeamMask();
  int myTid = getMyTid();
  KV right = {.raw = (__shfl_down_sync(teamMask, chunkKV.raw, 1))};
  return right;
}

__device__ int getInsertionIdx(KV insertKv, K key) {
  // TODO: check if this is correct
  /*Example:
  0 1 2 3 4
  1 3 5 7 x
  key = 4:
  ballot = 00011
  return 5 - 3 = 2
  key = 0:
  ballot = 00000
  return 5 - 5 = 0
  key = 8:
  ballot = 01111
  return 5 - 1 = 4
  after taking from left
  1 1 3 5 7
  key = 4:
  00111
  return 4 - 2 = 2
  key = 0:
  00000
  return 4 - 5 = -1??
  */
  uint32_t teamMask = getTeamMask();
  int tid = getMyTid();
  uint32_t ballot =
      __ballot_sync(teamMask, insertKv.kv.key < key && tid < DSIZE);
  PRINT0("ballot: %x key: %d\n", ballot, key);
  //     printf("[%2d] my kv: %u %u\n", tid, insertKv.kv.key,
  //     insertKv.kv.value);

  return TEAM_SIZE - __clz(ballot);
}

__device__ bool isKeyRaised() {
  // for now, always returning true
  // to implement: rng
  return true;
}

__device__ void debugPath(ChunkIdx segment);

class __attribute__((aligned(256))) Chunk { // chunks are 32 entries of 8B each
public:
  __device__ KV read();
  __device__ void Lock(SkiplistStats *, uint32_t, int);
  __device__ bool TryLock(SkiplistStats *, uint32_t, int);
  __device__ void Unlock();
  __device__ void UpdateNextKey(K);
  __device__ void UpdateNextVal(V);
  __device__ void UpdateNextBoth(KV);
  __device__ void AtomicWrite(int, KV);
  __host__ __device__ uint64_t *all_data_ptr();
  __device__ void AtomicKVWrite(int, KV);
  __device__ void AtomicKWrite(int, K);
  __device__ void AtomicVWrite(int, V);
  __device__ void ShiftData(int);
  __device__ void CopyDataTo(Chunk *dst, int ignoreIdx, K &min_key, K &max_key);
  __device__ void MarkZombie();
  __device__ void printChunk();

private:
  union {
    uint64_t all_data[TEAM_SIZE];
    struct {
      KV data[DSIZE];
      KV next;
      uint64_t lock;
    } __attribute__((packed));
  };
  friend GFSL;

} __attribute__((packed));

__host__ __device__ uint64_t *Chunk::all_data_ptr() {
  return &(this->all_data[0]);
}

__device__ void Chunk::AtomicWrite(int idx, KV value) {
  // The compiler is complaining if uint64_t* is passed and static_cast not
  // working :/
  atomicExch(reinterpret_cast<unsigned long long *>(&all_data[idx]),
             static_cast<unsigned long long>(value.raw));
  __threadfence();
}

__device__ void Chunk::AtomicKVWrite(int idx, KV value) {
  // The compiler is complaining if uint64_t* is passed and static_cast not
  // working :/
  atomicExch(reinterpret_cast<unsigned long long *>(&all_data[idx]), value.raw);
  __threadfence();
}

__device__ void Chunk::AtomicKWrite(int idx, K key) {
  atomicExch(reinterpret_cast<unsigned int *>(&all_data[idx]), key);
  __threadfence();
}

__device__ void Chunk::AtomicVWrite(int idx, V value) {
  atomicExch(reinterpret_cast<unsigned int *>(&all_data[idx]) + 1, value);
  __threadfence();
}

__device__ void Chunk::ShiftData(int shift) {
  int tid = getMyTid();
  KV kv = read();
  int size = occupiedSlots(kv);
  for (int i = size - 1; i >= 0; i--) {
    if (tid == i)
      AtomicKVWrite(i + shift, kv);
  }
  __syncwarp();
}

__device__ void Chunk::CopyDataTo(Chunk *dst, int ignoreIdx, K &min_key,
                                  K &max_key) {
  int tid = getMyTid();
  KV data = read();
  int size = occupiedSlots(data);
  if (tid < DSIZE && tid != ignoreIdx && data.kv.key != POS_INF) {
    if (tid < ignoreIdx)
      dst->AtomicKVWrite(tid, data);
    else
      dst->AtomicKVWrite(tid - 1, data);
  }
  min_key = __shfl_sync(getTeamMask(), data.kv.key, 0);
  max_key = __shfl_sync(getTeamMask(), data.kv.key, size - 1);
}

__device__ KV Chunk::read() {
  int tid = getMyTid();
  //     __syncwarp();
  // printf("read %i %p\n", tid, this);
  return {.raw = all_data[tid]};
}

// spinlock
__device__ void Chunk::Lock(SkiplistStats *stats, uint32_t nodeId, int lvl) {
  int tid = getMyTid();
  PRINT0("locking %p\n", this);
  //     if (getMyTid() == 0) printf("[%d] [%s:%d] " "locking %p\n",
  //     (blockIdx.x*blockDim.x+threadIdx.x)/32, __func__, __LINE__, this);
  //     this->printChunk();
  if (tid == TID_LOCK) {
    //         uint64_t oldVal = this->lock,
    uint64_t oldValOut;
    int i = (blockIdx.x % 1023), j = 0;
    while (1) {
#if defined(ENABLE_STATS)
      incrementLockAttempt(stats, nodeId, lvl);
#endif
      oldValOut =
          atomicCAS(reinterpret_cast<unsigned long long *>(&(this->lock)),
                    LOCK_UNLOCKED, LOCK_LOCKED);
      if (oldValOut == LOCK_UNLOCKED) {
#if defined(ENABLE_STATS)
        incrementLockSuccess(stats, nodeId, lvl);
#endif
        break;
      } else if (oldValOut == LOCK_ZOMBIE)
        break;
      for (; i > 0; i--) {
        j++;
      }
      i = j << 1;
      if (i > 65536)
        i = 1;
      j = 0;
    }
  }
  __syncwarp();
  __threadfence();
  PRINT0("locked %p\n", this);
}

__device__ bool Chunk::TryLock(SkiplistStats *stats, uint32_t nodeId, int lvl) {

  int tid = getMyTid();
  uint32_t teamMask = getTeamMask();
  bool result = false;
  if (tid == TID_LOCK) {
    uint64_t oldVal, oldValOut;
    oldVal = this->lock;
    if (oldVal == LOCK_UNLOCKED) {
#if defined(ENABLE_STATS)
      incrementLockAttempt(stats, nodeId, lvl);
#endif
      oldValOut = atomicCAS(reinterpret_cast<unsigned long long *>(&this->lock),
                            oldVal, LOCK_LOCKED);
      __threadfence();
      if (oldValOut == oldVal) {
#if defined(ENABLE_STATS)
        incrementLockSuccess(stats, nodeId, lvl);
#endif
        result = true;
      }
    }
  }
  return __shfl_sync(teamMask, result, TID_LOCK, TEAM_SIZE);
}

__device__ void Chunk::Unlock() {
  __threadfence();
  PRINT0("unlocking %p\n", this);
  //     if (getMyTid() == 0) printf("[%d] [%s:%d] " "unlocking %p\n",
  //     (blockIdx.x*blockDim.x+threadIdx.x)/32, __func__, __LINE__, this);
  int tid = getMyTid();
  if (tid == TID_LOCK) {
    if (this->lock == LOCK_LOCKED) {
      atomicExch(reinterpret_cast<unsigned long long *>(&this->lock),
                 LOCK_UNLOCKED);
      ///             __threadfence();
    }
  }
  __syncwarp();
  PRINT0("lock value: %lld\n", this->lock);
}

__device__ void Chunk::UpdateNextKey(K new_key) {
  if (getMyTid() == TID_NEXT) {
    //         int active = __activemask();
    //         PRINT_TID(TID_NEXT, "current threads: %x\n", active);
    // TODO: need to make this atomic?
    // this->next.kv.key = new_key;
    AtomicKWrite(TID_NEXT, new_key);
  }
  //     int active = __activemask();
  //     if (1) PRINTF("[%s:%d] current threads: %x\n", __func__, __LINE__,
  //     active);
}

__device__ void Chunk::UpdateNextVal(V new_val) {
  // no need to change max field, is same for current chunk
  int tid = getMyTid();
  if (tid == TID_NEXT) {
    //         KV new_next = KV(this->next);
    //         new_next.kv.value = new_val;
    //         this->next = static_cast<uint64_t>(new_next);
    // TODO: need to make this atomic?
    // this->next.kv.value = new_val; // Keeping this around incase we switch to
    // storing KVs
    AtomicVWrite(TID_NEXT, new_val);
  }
  __syncwarp();
}

__device__ void Chunk::UpdateNextBoth(KV new_kv) {
  if (getMyTid() == TID_NEXT) {
    //         printf("[%s:%d] \nthis ptr: %p\nnext bit: %p\ncalcslot:
    //         %p\ncalslot2: %p\ncalslot2: %p\ndiff way: %p\ndiffway2: %p\n",
    //         __func__, __LINE__, this, &(this->next),
    //         &(this->all_data[TID_NEXT]), &(this->all_data[0]),
    //         &(this->all_data[1]), &(this->data[0]), &(this->data[1]));
    //         printf("sizeof kv: %lu\n", sizeof(KV));
    //         printf("[%s:%d] old next: %u %u\n", __func__, __LINE__,
    //         this->next.kv.key, this->next.kv.value);
    // _ull old = this->next.raw;
    // _ull ret;
    // while(old != (ret =
    // atomicCAS(reinterpret_cast<_ull *>(&(this->next.raw)), old, new_kv.raw)))
    // {
    //     old = ret;
    // }
    AtomicKVWrite(TID_NEXT, new_kv);
    //         printf("[%s:%d] new next: %u %u\n", __func__, __LINE__,
    //         this->next.kv.key, this->next.kv.value);
  }
}

class GFSL {
public:
  Chunk *head;
  // uint32_t height; //max == TEAM_SIZE - 1, min == 0 (one level only)
  //* no height parameter to match paper
  Chunk *memory_pool;
  uint32_t pool_size;
  uint32_t num_allocated; // counter for getNewChunk
  __device__ int getHeight();
  __device__ ChunkIdx getNewChunk();
  __device__ Chunk *getChunkFromIdx(ChunkIdx);
  __device__ ChunkIdx searchDown(K, SkiplistStats *);
  __device__ KV searchLateral(K, ChunkIdx, SkiplistStats *);
  __device__ ChunkIdx firstChunkAtLevel(int);
  __device__ ChunkIdx backTrack(KV &, K);
  __device__ uint64_t searchSlow(K key, /* ChunkIdx &, K &,*/ SkiplistStats *);
  __device__ bool insertToLevel(int, ChunkIdx &, K, V, bool &, SkiplistStats *);
  __device__ bool isLevelEmpty(int);
  __device__ void executeInsert(ChunkIdx, KV, K, V);
  __device__ void incrementNumChunksAtLevel(int);
  __device__ void decrementNumChunksAtLevel(int);
  __device__ uint64_t splitInsert(ChunkIdx, K, V, int, SkiplistStats *);
  __device__ ChunkIdx preSplit(ChunkIdx, int, SkiplistStats *);
  __device__ KV splitCopy(ChunkIdx, ChunkIdx, int);
  __device__ ChunkIdx insertNewData(K, V, ChunkIdx, ChunkIdx, KV);
  __device__ ChunkIdx lockNextChunk(ChunkIdx, SkiplistStats *);
  __device__ bool eraseFromLevel(int level, ChunkIdx &pEnc, K key,
                                 SkiplistStats *);
  __device__ void executeDelete(ChunkIdx, KV, int, K, uint32_t);
  __device__ void mergeDelete(ChunkIdx, KV, int, K, uint32_t, SkiplistStats *);
  __device__ ChunkIdx findAndLockEnclosing(ChunkIdx, K, int, SkiplistStats *);
  // note: head structure: keys -> counters, values -> chunk index, height = 0 @
  // all_data 0
  __device__ ChunkIdx findAndLockNextNonZombie(ChunkIdx, int, SkiplistStats *);
  __device__ void updateDownPtrs(int level, V new_val, K min_key, K max_key,
                                 SkiplistStats *);
  //     public:
  __host__ GFSL(int max_nodes, bool host_side);
  __host__ GFSL();
  __host__ ~GFSL();
  __host__ void freeGFSL();
  __device__ GFSL(int max_nodes);
  __device__ KV contains(K, SkiplistStats *);
  __device__ bool insert(K, V, SkiplistStats *);
  __device__ bool erase(K, SkiplistStats *);
  __host__ void print(bool uvm);
  __device__ void dumpList();
  __device__ void debugPath(ChunkIdx segment, ChunkIdx, K);
  __device__ bool deleteCorrectness(K key);
  __host__ bool initializeGFSL(int max_nodes, bool host_side);

  friend Chunk;
};

__host__ GFSL::GFSL() {}

__host__ GFSL::GFSL(int max_nodes, bool host_side) {
  // restrict to one thread if on device? or do on CPU?
  pool_size = max_nodes;
  num_allocated = TEAM_SIZE; // first nodes for each level
  cudaMallocManaged(&memory_pool,
                    sizeof(Chunk) * (max_nodes + 1)); // extra node for head

  uint64_t *head_data = memory_pool->all_data_ptr();
  head = memory_pool++; // next Chunk is the start of the mem pool

  // set up head node, starts with ctr = 0 for all but lowest level (where ctr = 1)
  cudaMemset(head_data, 0x00, sizeof(Chunk));
  KV temp = {.kv = {.key = 1, .value = 0}};
  cudaMemcpy(&head_data[0], &(temp.raw), sizeof(uint64_t),
             cudaMemcpyHostToDevice);
  temp.kv.key = 0;
  for (int i = 1; i < TEAM_SIZE; i++) {
    temp.kv.value = i;
    cudaMemcpy(&head_data[i], &(temp.raw), sizeof(uint64_t),
               cudaMemcpyHostToDevice);
  }

  // set up first nodes on each level
  // set up first node: {{-inf, CHK_INVALID}, {inf, CHK_INVALID} ..., {0, 0}
  // (last is lock)}
  // VIPIN:using managed memory, use copy constuctor or initializer list
  Chunk device_side_chunk;
  uint64_t *device_chunk_data = device_side_chunk.all_data_ptr();
  device_chunk_data[0] = (KV{.kv = {.key = NEG_INF, .value = CHK_INVALID}}).raw;
  for (int i = 1; i < TID_LOCK; i++) {
    device_chunk_data[i] =
        (KV{.kv = {.key = POS_INF, .value = CHK_INVALID}}).raw;
  }
  device_chunk_data[TID_LOCK] = LOCK_UNLOCKED;
  cudaMemcpy(memory_pool[0].all_data_ptr(), device_chunk_data, sizeof(Chunk),
             cudaMemcpyHostToDevice);
  for (int i = 1; i < TEAM_SIZE; i++) {
    // node structure: first entry is {-inf, pointer to node in level below},
    // rest are {inf, CHK_INVALID}
    device_chunk_data[0] =
        (KV{.kv = {.key = NEG_INF, .value = static_cast<unsigned int>(i - 1)}})
            .raw;
    cudaMemcpy(memory_pool[i].all_data_ptr(), device_chunk_data, sizeof(Chunk),
               cudaMemcpyHostToDevice);
  }

  // we need to copy this to cuda mem now....... or do we?
}

__host__ bool GFSL::initializeGFSL(int max_nodes, bool host_side) {
  // restrict to one thread if on device? or do on CPU?
  bool status = true;
  pool_size = (max_nodes + TEAM_SIZE + 1);
  // additional node at each level and head node
  num_allocated = TEAM_SIZE; // first nodes for each level

  uint64_t *head_data = this->memory_pool->all_data_ptr();
  head = this->memory_pool++; // next Chunk is the start of the mem pool

  // set up head node, starts with ctr = 0 for all but lowest level (where ctr =
  // 1)
  // memory pool is zeroed by the driver no need for memset
  // cudaMemset(head_data, 0x00, sizeof(Chunk));
  KV temp = {.kv = {.key = 1, .value = 0}};
  head_data[0] = temp.raw;
  temp.kv.key = 0;
  for (int i = 1; i < TEAM_SIZE; i++) {
    temp.kv.value = i;
    head_data[i] = temp.raw;
  }

  // set up first nodes on each level
  // set up first node: {{-inf, CHK_INVALID}, {inf, CHK_INVALID} ..., {0, 0}
  // (last is lock)}
  // intialization of level 0
  this->memory_pool[0].all_data[0] =
      (KV{.kv = {.key = NEG_INF, .value = CHK_INVALID}}).raw;
  for (int i = 1; i < TID_LOCK; i++) {
    this->memory_pool[0].all_data[i] =
        (KV{.kv = {.key = POS_INF, .value = CHK_INVALID}}).raw;
  }
  memory_pool[0].all_data[TID_LOCK] = LOCK_UNLOCKED;
  // intialization of higher levels
  for (int i = 1; i < TEAM_SIZE; i++) {
    // node structure: first entry is {-inf, pointer to node in level below},
    // rest are {inf, CHK_INVALID}
    memory_pool[i].all_data[0] =
        (KV{.kv = {.key = NEG_INF, .value = static_cast<unsigned int>(i - 1)}})
            .raw;
    for (int j = 1; j < TID_LOCK; j++) {
      memory_pool[i].all_data[j] =
          (KV{.kv = {.key = POS_INF, .value = CHK_INVALID}}).raw;
    }
    memory_pool[i].all_data[TID_LOCK] = LOCK_UNLOCKED;
  }

  return status;
}

__host__ GFSL::~GFSL() {
  cudaFree(memory_pool);
  cudaFree(head);
}

__host__ void GFSL::freeGFSL() {
  cudaFree(memory_pool);
  cudaFree(head);
}
__device__ ChunkIdx GFSL::getNewChunk() {
  // TODO: check, finish
  // arbitrary, but we want only one thread to do this
  uint32_t idx, teamMask = getTeamMask();
  int tid = getMyTid();
  if (tid == TID_LOCK) {
    // VIPIN: changing limit to pool_size
    idx = atomicInc(static_cast<unsigned int *>(&(this->num_allocated)),
                    pool_size);
  }
  idx = __shfl_sync(teamMask, idx, TID_LOCK, TEAM_SIZE);
  // set up the chunk, each key gets POS_INF, each value gets CHK_INVALID, and
  // lock gets LOCK_LOCKED
  if (idx == CHK_INVALID)
    EPRINTF("INVALID HERE[%d]\n", 0);
  uint64_t *entry = &(this->getChunkFromIdx(idx)->all_data_ptr()[tid]);
  if (tid == TID_LOCK) {
    *entry = LOCK_LOCKED;
  } else {
    *entry = static_cast<uint64_t>(POS_INF) << 32 |
             static_cast<uint64_t>(CHK_INVALID);
  }
  return idx;
}

__device__ Chunk *GFSL::getChunkFromIdx(ChunkIdx idx) {
  //     if(idx == CHK_INVALID){
  //         printf("Something is very wrong\n");
  //     }
  return &(this->memory_pool[idx]);
}

__device__ KV GFSL::contains(K key, SkiplistStats *stats) {
  //     if (getMyTid() == 0) printf("contains pre-searchDown\n");
  ChunkIdx pCurr = this->searchDown(key, stats);
  //     if (getMyTid() == 0) printf("contains pre-searchLateral\n");
  return this->searchLateral(key, pCurr, stats);
}

__device__ ChunkIdx GFSL::searchDown(K key, SkiplistStats *stats) {
restart:
  KV prevKv = {.raw = NULL};
  bool prevKvNotSet = true;
  // can combine below two into the same function
  int height = getHeight();
  ChunkIdx pCurr = firstChunkAtLevel(height);

  PRINT0("height: %d pCurr: %d\n", height, pCurr);

  while (height > 0) {
    PRINT0("pCurr: %d\n", pCurr);
    int active = __activemask();
    PRINT0("threads: %x\n", active);
    KV currKv = this->getChunkFromIdx(pCurr)->read();
    if (isZombie(currKv)) {
#if defined(ENABLE_STATS)
      incrementLateralMv(height, stats);
#endif
      pCurr = getPtrFromTid(TID_NEXT, currKv);
      if (pCurr == CHK_INVALID) { // i.e. end of the line, so backtrack
        if (prevKvNotSet)
          goto restart;
#if defined(ENABLE_STATS)
        incrementDownwardMv(height, stats);
#endif
        height--;
        pCurr = backTrack(prevKv, key);
      }
      continue;
    }

    int stepTid = getTidForNextStep(
        key, currKv); // this will return TID_NONE if nowhere else to go, so no
                      // need to worry about CHK_INVALID

    if (stepTid == TID_NEXT) {
#if defined(ENABLE_STATS)
      incrementLateralMv(height, stats);
#endif
      prevKv = currKv;
      prevKvNotSet = false;
      pCurr = getPtrFromTid(TID_NEXT, currKv);
    } else if (stepTid != TID_NONE) {
#if defined(ENABLE_STATS)
      // TODO: allow only one thread to update
      incrementDownwardMv(height, stats);
#endif
      height--;
      prevKv = {.raw = NULL};
      prevKvNotSet = true;
      pCurr = getPtrFromTid(stepTid, currKv);
    } else {
      if (prevKvNotSet) {
        PRINT0("restarting, last pCurr: %d, max: %u\n", pCurr, num_allocated);
        // PRINTALLKV(currKv);
        goto restart;
      }
#if defined(ENABLE_STATS)
      incrementDownwardMv(height, stats);
#endif
      height--;
      pCurr = backTrack(prevKv, key);
    }
  }
  PRINT0("returning %d\n", pCurr);
  return pCurr;
}

__forceinline__ __device__ int GFSL::getHeight() {
  uint32_t teamMask = getTeamMask();
  int myTid = getMyTid();

  KV mykv = this->head->read();
  // thread with key > 0
  uint32_t ballot = __ballot_sync(teamMask, mykv.kv.key > 0);
  //clz: number of consecutive high-order zero bits in a 32-bit integer
  return (TEAM_SIZE - 1) - __clz(ballot);
}

__device__ ChunkIdx GFSL::firstChunkAtLevel(int height) {
  uint32_t teamMask = getTeamMask();
  int myTid = getMyTid();
  KV mykv = this->head->read();
  ChunkIdx ret;
  if (myTid == height) {
    ret = mykv.kv.value;
  }
  return __shfl_sync(teamMask, ret, height, TEAM_SIZE);
}

__device__ ChunkIdx GFSL::backTrack(KV &prevKv, K key) {
  int stepTid = getTidOfDownStep(key, prevKv);
  if (stepTid == TID_NONE) {
    return CHK_INVALID;
  }
  ChunkIdx pNextStep = getPtrFromTid(stepTid, prevKv);
  prevKv = {.kv = {.key = 0, .value = 0}};
  return pNextStep;
}

__device__ KV GFSL::searchLateral(K key, ChunkIdx pCurr, SkiplistStats *stats) {
  KV currKv, ret;
  int foundTid;
  uint32_t teamMask = getTeamMask();
  do {
    currKv = this->getChunkFromIdx(pCurr)->read();
    foundTid = getTidForNextStep(key, currKv);
    //         PRINT0("pCurr: %d foundTid: %d\n", pCurr, foundTid);
    //         PRINTALLKV(currKv);
    if (foundTid == TID_NEXT || isZombie(currKv)) {
#if defined(ENABLE_STATS)
      // stats->recordLateralMv(0);
      incrementLateralMv(0, stats);
#endif
      foundTid = TID_NEXT;
      pCurr = getPtrFromTid(TID_NEXT, currKv);
    }
    // PRINT0("pCurr: %d foundTid: %d\n", pCurr, foundTid);
  } while (foundTid == TID_NEXT);

  if (foundTid == TID_NONE) {
    ret.raw = -1ULL;
  } else {
    ret.raw = __shfl_sync(teamMask, currKv.raw, foundTid, TEAM_SIZE);
    if (ret.kv.key != key) {
      ret.raw = -1ULL;
    }
  }
  return ret;
}

__device__ bool GFSL::insert(K key, V value, SkiplistStats *stats) {
  //     if (getMyTid() == 0) printf("insert(%u, %u)\n", key, value);
  PRINT0("(%u %u)\n", key, value);
  // ChunkIdx useless = 0;
  // K useless2;
  // check whether key already exists, return the traversal path
  uint64_t flag_ptr = this->searchSlow(key, /* useless, useless2,*/ stats);
  uint64_t flag = (flag_ptr & 0x0100000000UL);
  if (flag) {
    PRINT0("(%u %u) fin\n", key, value);
    return false;
  }

  ChunkIdx path = (flag_ptr & 0x00ffffffffUL);
  ChunkIdx pBottom = getPathFromTid(path, 0);
  bool raiseKey = false;
  PRINT0("pBottom: %u\n", pBottom);
  // warning: pBottom may be updated after this!
  if (!this->insertToLevel(0, pBottom, key, value, raiseKey, stats)) {
    this->getChunkFromIdx(pBottom)->Unlock();
    PRINT0("(%u %u) fin\n", key, value);
    return false;
  }
  PRINT0("pBottom: %u\n", pBottom);
  value = pBottom;
  // If raise key is true after insertion at level 0, insert to upper level
  for (int level = 1; raiseKey && level < MAX_LEVEL; level++) {
    PRINT0("value: %u\n", value);
    ChunkIdx pEnclose = getPathFromTid(path, level);
    insertToLevel(level, pEnclose, key, value, raiseKey, stats);
    value = pEnclose;
    getChunkFromIdx(pEnclose)->Unlock();
  }
  getChunkFromIdx(pBottom)->Unlock();
  ///     __syncwarp();
  PRINT0("(%u %u) fin\n", key, value);
  return true;
}

__device__ uint64_t
GFSL::searchSlow(K key, /*ChunkIdx &debug, K &debug_chosen,*/
                 SkiplistStats *stats) {
  PRINT0("(%u)\n", key);
  int myTid = getMyTid();
  uint32_t teamMask = getTeamMask();
  int restart_count = 0;
restart:
  if (restart_count > 10000000) {
    EXPRINT0("exceeded %d restarts, failing\n", 10000000);
    return 0x00ffffffff;
  }
  restart_count++;
  KV prevKv = {.kv = {.key = 0, .value = 0}};
  int height = this->getHeight();
  ChunkIdx mySegment =
      (head->read()).kv.value; // first chunk at that level, null case
  ChunkIdx pPrev = CHK_INVALID, pCurr = this->firstChunkAtLevel(height);

  while (height > 0) {
    if (pCurr == CHK_INVALID)
      EPRINTF("INVALID HERE[%d]\n", 0);
    KV currKv = this->getChunkFromIdx(pCurr)->read();
    if (isZombie(currKv)) {
      if (pPrev != CHK_INVALID &&
          !isZombie(prevKv) // this can happen if a pPrev couldn't be locked
                            // earlier so we continued
          && this->getChunkFromIdx(pPrev)->TryLock(stats, pPrev, height)) {
        // make sure this node hasn't been split while we were checking stuff
        prevKv = getChunkFromIdx(pPrev)->read();
        if (__shfl_sync(teamMask, prevKv.kv.value, TID_NEXT, TEAM_SIZE) ==
            pCurr) {
          // nothing happened to insert a new node between pPrev and pCurr
          // somehow, so carry on redirect next pointer of prev to next
          // non-zombie node
          while (pCurr != CHK_INVALID && isZombie(currKv)) {
#if defined(ENABLE_STATS)
            incrementLateralMv(height, stats);
#endif
            pCurr = getPtrFromTid(TID_NEXT, currKv);
            if (pCurr != CHK_INVALID) {
              currKv = this->getChunkFromIdx(pCurr)->read();
            }
          }
          // question: should all zombie nodes in between also be redirected?
          // I don't think so, because there isn't much point
          if (pCurr == CHK_INVALID)
            EPRINTF("INVALID HERE[%d]\n", 0);
          this->getChunkFromIdx(pPrev)->UpdateNextVal(
              pCurr); // same whether CHK_INVALID or not
          getChunkFromIdx(pPrev)->Unlock();
          // TODO: should the below ever happen?
          if (pCurr == CHK_INVALID) {
            // backtrack, but we know prevKv isn't null
            EXPRINT0("Do we ever hit this?[%d]\n",
                     0); // incredibly rarely, it seems
#if defined(ENABLE_STATS)
            incrementDownwardMv(height, stats);
#endif
            pCurr = backTrack(prevKv, key);
            if (pCurr == CHK_INVALID)
              goto restart;
            if (myTid == height) {
              mySegment = pPrev;
              // debug = pCurr;
              // debug_chosen = (uint32_t)-1;
            }
            //                         __syncwarp();
            height--;
          }
        } else {
#if defined(ENABLE_STATS)
          incrementLateralMv(height, stats);
#endif
          getChunkFromIdx(pPrev)->Unlock();
          pPrev = pCurr;
          prevKv = currKv;
          pCurr = getPtrFromTid(TID_NEXT, currKv);
          continue;
        }
      } else {
#if defined(ENABLE_STATS)
        incrementLateralMv(height, stats);
#endif
        // either pPrev isn't a chunk e.g. first chunk of level
        // or pPrev is also a zombie, or we couldn't lock pPrev, either way
        // just continue onwards
        pPrev = pCurr;
        prevKv = currKv;
        // TODO: ^these two necessary?
        pCurr = getPtrFromTid(TID_NEXT, currKv);
        continue;
      }
    }

    currKv = getChunkFromIdx(pCurr)->read();
    int stepTid = getTidForNextStep(key, currKv);
    if (stepTid == TID_NEXT) {
#if defined(ENABLE_STATS)
      incrementLateralMv(height, stats);
#endif
      // max ele in node smaller than key, lateral search
      prevKv = currKv;
      pPrev = pCurr;
      pCurr = getPtrFromTid(TID_NEXT, currKv);
    } else if (stepTid != TID_NONE) {
#if defined(ENABLE_STATS)
      incrementDownwardMv(height, stats);
#endif
      prevKv = {.kv = {.key = 0, .value = 0}};
      // store this chunk into a thread's mySegment
      if (myTid == height) {
        mySegment = pCurr;
        // debug = getPtrFromTid(stepTid, currKv);
      }
      ///             __syncwarp();
      height--;
      pPrev = CHK_INVALID;
      // K temp = __shfl_sync(teamMask, currKv.kv.key, stepTid, TEAM_SIZE);
      pCurr = getPtrFromTid(stepTid, currKv);
      // if (myTid == height) { // debug
      //   debug = pCurr;
      //   debug_chosen = temp;
      // }

    } else { // backtrack
      if (pPrev == CHK_INVALID)
        goto restart;
        // store chunk
        //             EXPRINT0("Do we ever hit this?[%d]\n", key); //quite a lot,
        //             in fact
        // debugging info
        // int tempTid = getTidOfDownStep(key, prevKv);
        // K temp = (uint32_t)-1;

        // if (tempTid == TID_NONE) {
        //   EXPRINT0("tempTid is TID_NONE for key %d\n", key);
        // } else {
        //   temp = __shfl_sync(teamMask, prevKv.kv.key, tempTid, TEAM_SIZE);
        // }
#if defined(ENABLE_STATS)
      incrementDownwardMv(height, stats);
#endif
      pCurr = backTrack(prevKv, key);
      if (pCurr == CHK_INVALID)
        goto restart;

      if (myTid == height) {
        mySegment = pPrev;
        // debug = pCurr;
        // debug_chosen = temp;
      }
      __syncwarp();
      height--;
      pPrev = CHK_INVALID;
      //             pCurr = backTrack(prevKv, key);
    }
  }

  // here: should have reached lowest level
  // check for CHK_INVALID? Shouldn't need to if previous step was a downstep
  PRINT0("lowest level, pCurr: %d\n", pCurr);
  int foundTid = -1;
  prevKv = {.kv = {.key = 0, .value = 0}};
  pPrev = CHK_INVALID;
  KV currKv;
  do {
    if (pCurr == CHK_INVALID)
      EPRINTF("INVALID HERE[%d]\n", 0);
    currKv = this->getChunkFromIdx(pCurr)->read();
    PRINT0("pCurr: %d\n", pCurr);
    // rPRINTALLKV(currKv);
    foundTid = getTidForNextStep(key, currKv);
    PRINT0("foundTid: %d\n", foundTid);
    if (isZombie(currKv)) {
      // same old, see above in this function
      if (pPrev != CHK_INVALID && !isZombie(prevKv) &&
          this->getChunkFromIdx(pPrev)->TryLock(stats, pPrev, 0)) {
        prevKv = getChunkFromIdx(pPrev)->read();
        if (__shfl_sync(teamMask, prevKv.kv.value, TID_NEXT, TEAM_SIZE) ==
            pCurr) {
          // skipping all zombies
          while (pCurr != CHK_INVALID && isZombie(currKv)) {
#if defined(ENABLE_STATS)
            // stats->recordLateralMv(0);
            incrementLateralMv(0, stats);
#endif
            PRINT0("pCurr: %d\n", pCurr);
            //                     PRINTALLKV(currKv);
            pCurr = getPtrFromTid(TID_NEXT, currKv);
            if (pCurr != CHK_INVALID)
              currKv = this->getChunkFromIdx(pCurr)->read();
          }
          // question: need to decrement counter by these many chunks too,
          // right? actually, probably not since zombie chunks should already be
          // logically removed when erase is called, this is just the physical
          // removal part

          if (pPrev == CHK_INVALID)
            EPRINTF("INVALID HERE[%d]\n", 0);
          this->getChunkFromIdx(pPrev)->UpdateNextVal(pCurr);
          getChunkFromIdx(pPrev)->Unlock();
          if (pCurr == CHK_INVALID) {
            // no choice but to return pPrev...?
            pCurr = pPrev;
            foundTid = TID_NONE;
          }
        } else {
#if defined(ENABLE_STATS)
          // stats->recordLateralMv(0);
          incrementLateralMv(0, stats);
#endif
          getChunkFromIdx(pPrev)->Unlock();
          pPrev = pCurr;
          prevKv = currKv;
          pCurr = getPtrFromTid(TID_NEXT, currKv);
          continue;
        }
      } else {
#if defined(ENABLE_STATS)
        // stats->recordLateralMv(0);
        incrementLateralMv(0, stats);
#endif
        pPrev = pCurr;
        prevKv = currKv;
        pCurr = getPtrFromTid(TID_NEXT, currKv);
        continue;
      }
    } else if (foundTid == TID_NEXT) {
#if defined(ENABLE_STATS)
      // stats->recordLateralMv(0);
      incrementLateralMv(0, stats);
#endif
      foundTid = TID_NEXT;
      pPrev = pCurr;
      prevKv = currKv;
      pCurr = getPtrFromTid(TID_NEXT, currKv);
    }
    PRINT0("foundTid: %d\n", foundTid);
  } while (foundTid == TID_NEXT);

  // testing: make sure that pCurr is the enclosing chunk
  if (myTid == 0) {
    mySegment = pCurr;
    // debug = pCurr;
    // debug_chosen = 0;
  }

  // mySegment should be different for each thread -> path
  if (foundTid != TID_NONE) {
    bool found = getTidWithKey(key, currKv) != TID_NONE;
    PRINT0("returning %llx\n", (static_cast<uint64_t>(found) << 32) |
                                   static_cast<uint64_t>(mySegment));
    return (static_cast<uint64_t>(found) << 32) |
           static_cast<uint64_t>(mySegment);
  } else {
    PRINT0("returning %llx\n", static_cast<uint64_t>(mySegment));
    return static_cast<uint64_t>(mySegment);
  }
} // SEARCH_SLOW function ends here

// warning: this can update pEnc! updates in case of a split happening (the
// update needs to propagate to GFSL::insert too)
__device__ bool GFSL::insertToLevel(int level, ChunkIdx &pEnc, K key, V value,
                                    bool &raiseKey, SkiplistStats *stats) {
  PRINT0("(%d, %u, %u, %u, %u)\n", level, pEnc, key, value, raiseKey);
  int tid = getMyTid();
  // if (tid == 0) printf("1\n");
  pEnc = findAndLockEnclosing(pEnc, key, level, stats);
  // if (tid == 0) printf("2\n");
  KV encKv = this->getChunkFromIdx(pEnc)->read();

  // PRINTCHK(pEnc);

  if (chunkContains(key, encKv))
    return false;
  // if (tid == 0) printf("3\n");
  raiseKey = false;
  // if (tid == 0) printf("4\n");
  //     PRINT0("[%d] crazy pills\n", 0)
  if (occupiedSlots(encKv) < DSIZE) {
    //         if (tid == 0) printf("executeInsert chosen\n");
    executeInsert(pEnc, encKv, key, value);
    bool decision = isLevelEmpty(level);
    //         printf("[%d] decision: %d\n", tid, decision);
    if (level > 0 && decision) {
      this->incrementNumChunksAtLevel(level);
    }
    PRINT0("reached here %d\n", 0);
    ///         __syncwarp();
  } else {
    //         if (tid == 0) printf("splitInsert chosen\n");
    int active = __activemask();
    PRINT0("current threads: %x\n", active);
    uint64_t pEnc_key = this->splitInsert(pEnc, key, value, level, stats);
    // better way to do the below? maybe std::tuple?
    pEnc = static_cast<ChunkIdx>(pEnc_key >> 32);
    key = static_cast<K>(pEnc_key & 0x0ffffffffUL);
    PRINT0("[%d] calling incrementNumChunks\n", 0);
    this->incrementNumChunksAtLevel(level);
    raiseKey = isKeyRaised();
  }
  PRINT0("(%d, %u, %u, %u, %u) fin\n", level, pEnc, key, value, raiseKey);
  ///     __syncwarp();
  return true;
} // INSERT_TO_LEVEL function ends here

__device__ bool GFSL::isLevelEmpty(int level) {
  PRINT0("called for level %d\n", level);
  int tid = getMyTid();
  uint32_t teamMask = getTeamMask();
  K ctrs = this->head->read().kv.key;
  bool ret;
  if (tid == level) {
    ret = ctrs == 0;
  }
  ret = __shfl_sync(teamMask, ret, level, TEAM_SIZE);
  PRINT0("returning %d\n", ret);
  return ret;
}

__device__ void GFSL::executeInsert(ChunkIdx pEnc, KV encKv, K key, V value) {
  int tid = getMyTid();
  KV insertKv = getChunkValFromLeftNeighbor(encKv);
  // TODO: check if this is correct, because paper uses insertKv instead here
  int insertIdx = getInsertionIdx(encKv, key);
  PRINT0("insertIdx: %d key: %d\n", insertIdx, key);
  // PRINT0("pEnc [%d] before:\n", pEnc);
  // PRINTCHK(pEnc);
  if (tid == insertIdx)
    insertKv = {.kv = {.key = key, .value = value}};

  for (int i = DSIZE - 1; i >= insertIdx; i--) {
    if ((insertKv.kv.key != POS_INF /*i.e. EMPTY slot*/) && (tid == i))
      this->getChunkFromIdx(pEnc)->AtomicWrite(tid, insertKv);
  }
  // PRINT0("pEnc [%d] after:\n", pEnc);
  // PRINTCHK(pEnc);
}

__device__ void GFSL::incrementNumChunksAtLevel(int level) {
  PRINT0("incrementing chunks at %d\n", level);
  int tid = getMyTid();
  uint32_t *addr =
      &((reinterpret_cast<KV *>(&(head->all_data[level])))->kv.key);
  PRINT0("head: %p all_data[level]: %p key addr: %p\n", head,
         &(head->all_data[level]), addr);
  if (tid == level) {
    // uint32_t old_val = *addr;
    //         while (old_val != atomicInc(addr, old_val)) {
    //             old_val = *addr;
    //         }
    atomicInc(addr, POS_INF); // use this or above?
  }
  __syncwarp();
}

__device__ void GFSL::decrementNumChunksAtLevel(int level) {
  int tid = getMyTid();
  uint32_t *addr =
      &((reinterpret_cast<KV *>(&(head->all_data[level])))->kv.key);
  if (tid == level) {
    // uint32_t old_val = *addr;
    //         while (old_val != atomicDec(addr, old_val)) {
    //             old_val = *addr;
    //         }
    atomicDec(addr, POS_INF); // use this or above?
  }
  __syncwarp();
}

__device__ uint64_t GFSL::splitInsert(ChunkIdx pSplit, K key, V value,
                                      int level, SkiplistStats *stats) {

  // note: pSplit is already locked here!
  uint32_t teamMask = getTeamMask();
  int tid = getMyTid();
#if defined(ENABLE_STATS)
  if (tid == 0)
    stats->recordSplit();
#endif
  PRINT0("state of pSplit [%d] at start:\n", pSplit);
  PRINTCHK(pSplit);

  ChunkIdx pNew = this->preSplit(pSplit, level, stats);

  PRINT0("preSplit complete %d\n", key);

  KV splitKv = this->splitCopy(pSplit, pNew, DSIZE);
  ChunkIdx pInsert = this->insertNewData(key, value, pNew, pSplit, splitKv);

  PRINT0("state of pSplit [%d] after insertNewData:\n", pSplit);
  PRINTCHK(pSplit);
  PRINT0("state of pNew [%d] after InsertNewData:\n", pNew);
  PRINTCHK(pNew);

  if (pInsert == pSplit) {
    getChunkFromIdx(pNew)->Unlock();
  } else {
    getChunkFromIdx(pSplit)->Unlock();
  }
  //     if (tid == 0) printf("chunk unlocked\n");
  // keyForNextLevel
  K minK = __shfl_sync(teamMask, splitKv.kv.key, DSIZE / 2 + 1, TEAM_SIZE);
  if (minK > key)
    key = minK;

  K maxK = __shfl_sync(teamMask, splitKv.kv.key, DSIZE - 1, TEAM_SIZE);

  PRINT0("updating down ptrs: %d %d %d %d\n", level + 1, pNew, minK, maxK);

  this->updateDownPtrs(level + 1, pNew, minK, maxK, stats);
  //     if (tid == 0) printf("updateDownPtrs complete\n");

  PRINT0("state of pSplit [%d] at end:\n", pSplit);
  PRINTCHK(pSplit);
  PRINT0("state of pNew [%d] at end:\n", pNew);
  PRINTCHK(pNew);

  return static_cast<uint64_t>(pInsert) << 32 | static_cast<uint64_t>(key);
}

__device__ ChunkIdx GFSL::preSplit(ChunkIdx pSplit, int level,
                                   SkiplistStats *stats) {
  PRINT0("[%d] entered preSplit\n", 0);
  ChunkIdx pNext = findAndLockNextNonZombie(pSplit, level, stats);
  PRINT0("locked next chunk %d\n", pNext);
  // note: now, pSplit AND pNext are locked, pNew starts off locked anyway
  ChunkIdx pNew = getNewChunk();
  PRINT0("got new chunk %d %p\n", pNew, getChunkFromIdx(pNew));
  getChunkFromIdx(pNew)->UpdateNextVal(pNext);
  PRINT0("updated next of new chunk %d\n", pNew);
  if (pNext != CHK_INVALID) {
    getChunkFromIdx(pNext)->Unlock();
    PRINT0("unlocked pNext: %d\n", pNext);
  }
  return pNew;
}

// TODO: big question: WHY are we doing this? Is it necessary to lock the next
// chunk?
// VIPIN: function is unused
__device__ ChunkIdx GFSL::lockNextChunk(ChunkIdx pSplit, SkiplistStats *stats) {
  PRINT0("(%d)\n", pSplit);
  uint32_t teamMask = getTeamMask();
  KV currKv;
  ChunkIdx pNext = pSplit;
  do {
  restart:
    if (pNext == CHK_INVALID)
      EPRINTF("INVALID HERE[%d]\n", 0);
    currKv = this->getChunkFromIdx(pNext)->read();
    pNext = static_cast<ChunkIdx>(
        __shfl_sync(teamMask, currKv.kv.value, TID_NEXT, TEAM_SIZE));
  } while (isZombie(currKv) && pNext != CHK_INVALID);
  // note: should check for zombie again after locking, otherwise this is wrong
  // lock the found chunk
  PRINT0("pNext: %d\n", pNext);
  if (pNext != CHK_INVALID) {
    getChunkFromIdx(pNext)->Lock(stats, pNext,
                                 0); // VIPIN: putting 0 as func is unused
    currKv = getChunkFromIdx(pNext)->read();
    if (isZombie(currKv)) {
      PRINT0("%d zombie, restarting", pNext);
      goto restart;
    }
  }
  // note: we don't update the next field here because it'll have to be updated
  // to pNew later anyway
  PRINT0("returning %d\n", pNext);
  return pNext;
}

__device__ KV GFSL::splitCopy(ChunkIdx pSplit, ChunkIdx pNew, int numKeys) {
  uint32_t teamMask = getTeamMask();
  int tid = getMyTid();
  //     int thresholdTid = numKeys/2 - 1;
  int thresholdTid = numKeys / 2;
  int active = __activemask();
  PRINT0("current threads: %x\n", active);
  if (pSplit == CHK_INVALID)
    EPRINTF("INVALID HERE[%d]\n", 0);
  KV splitKv = this->getChunkFromIdx(pSplit)->read();
  // TODO: replace with getKeyFromTid function?
  K thresh = __shfl_sync(teamMask, splitKv.kv.key, thresholdTid, TEAM_SIZE);
  active = __activemask();
  PRINT0("current threads: %x\n", active);
  if (splitKv.kv.key > thresh /*TODO: this needed? -> */ && tid != TID_LOCK) {
    // this->copyToNewChunk(pNew, splitKv);
    // note: next should only copy key, not value too
    if (tid == TID_NEXT) {
      if (pNew == CHK_INVALID)
        EPRINTF("INVALID HERE[%d]\n", 0);
      this->getChunkFromIdx(pNew)->UpdateNextKey(splitKv.kv.key);
    } else if (tid < numKeys) {
      // write to start of pNew
      if (pNew == CHK_INVALID)
        EPRINTF("INVALID HERE[%d]\n", 0);
      // old atomic write
      this->getChunkFromIdx(pNew)->AtomicWrite(tid - (thresholdTid)-1, splitKv);
      // trying out non-atomic
      //             getChunkFromIdx(pNew)->data[tid - thresholdTid - 1] =
      //             splitKv;
    }
  }
  // TODO: should this syncwarp be here?
  __syncwarp();
  // update next of pSplit - this needs to happen before old values are emptied
  if (tid == TID_NEXT) {
    KV newnext = {.kv = {.key = thresh, .value = pNew}};
    PRINT_TID(TID_NEXT, "new next: %u %u\n", newnext.kv.key, newnext.kv.value);
    if (pSplit == CHK_INVALID)
      EPRINTF("INVALID HERE[%d]\n", 0);
    this->getChunkFromIdx(pSplit)->UpdateNextBoth(newnext);
  }
  // setMovedValsEmpty(splitKv);
  // TODO: should this be done here or in insertNewData?
  if (tid < numKeys && splitKv.kv.key > thresh) {
    // empty old position in pSplit
    KV empty = {.kv = {.key = POS_INF, .value = CHK_INVALID}};
    if (pSplit == CHK_INVALID)
      EPRINTF("INVALID HERE[%d]\n", 0);
    this->getChunkFromIdx(pSplit)->AtomicWrite(tid, empty);
    //             getChunkFromIdx(pSplit)->data[tid] = empty;
  }

  //     PRINT0("pSplit [%d] post updates:\n", pSplit);
  //     PRINTCHK(pSplit);
  //     PRINT0("pNew [%d] post updates:\n", pNew);
  //     PRINTCHK(pNew);
#if defined(DEBUG)
  active = __activemask();
  PRINT0("current threads: %x\n", active);
#endif
  return splitKv;
}

__device__ ChunkIdx GFSL::insertNewData(K key, V value, ChunkIdx pNew,
                                        ChunkIdx pSplit, KV splitKv) {
  uint32_t teamMask = getTeamMask();
  int tid = getMyTid();
  //     int thresholdTid = DSIZE/2 - 1;
  int thresholdTid = DSIZE / 2;
  K thresh = __shfl_sync(teamMask, splitKv.kv.key, thresholdTid, TEAM_SIZE);
  PRINT0("found threshold: %d\n", thresh);
  if (key > thresh) {
    // insert into pNew
    //         PRINTF("[%d] [%2d] splitKv: %u %u\n",
    //         (blockIdx.x*blockDim.x+threadIdx.x)/32, tid, splitKv.kv.key,
    //         splitKv.kv.value);
    PRINT0("d%dne\n", 0);
    splitKv.raw =
        __shfl_down_sync(teamMask, splitKv.raw, thresholdTid + 1, TEAM_SIZE);
    //         PRINTF("[%d] [%2d] splitKv: %u %u\n",
    //         (blockIdx.x*blockDim.x+threadIdx.x)/32, tid, splitKv.kv.key,
    //         splitKv.kv.value);
    PRINT0("d%dne\n", 0);
    if (tid >= DSIZE - thresholdTid - 1)
      splitKv = {.kv = {.key = POS_INF, .value = CHK_INVALID}}; // empty
    PRINTF("[%d] [%2d] splitKv: %u %u\n",
           (blockIdx.x * blockDim.x + threadIdx.x) / 32, tid, splitKv.kv.key,
           splitKv.kv.value);
    this->executeInsert(pNew, splitKv, key, value);

    return pNew;
  } else {
    // insert into pSplit
    if (splitKv.kv.key > thresh) {
      splitKv = {.kv = {.key = POS_INF, .value = CHK_INVALID}}; // empty
    }
    this->executeInsert(pSplit, splitKv, key, value);
  }
  return pSplit;
}

__device__ bool GFSL::erase(K key, SkiplistStats *stats) {
  // ChunkIdx debug = 0;
  // K debug_k = 0;
  uint64_t flag_ptr = searchSlow(key, /* debug, debug_k,*/ stats);
  uint64_t flag = (flag_ptr & 0x0100000000UL);
  ChunkIdx path = (flag_ptr & 0x00ffffffffUL);
  PRINTF("[%lu] [%d]: %lu\n", (blockIdx.x * blockDim.x + threadIdx.x) / 32,
         getMyTid(), path);
  if (!flag) {
    //         EPRINTF("Search Slow Path [%d]\n", path);
    //         ChunkIdx final = __shfl_sync(getTeamMask(), path, 0);
    //         EPRINTCHK(final);
    EPRINTF("Early ret here[%d]\n", 0);
    EXPRINT0("Flag false for %d\n", key);
    // debugPath(path, debug, debug_k);
    return false;
  }

  ChunkIdx pBottom = getPathFromTid(path, 0);
  //     ChunkIdx oldPB = pBottom;
  if (!eraseFromLevel(0, pBottom, key, stats)) {
    if (pBottom == CHK_INVALID)
      EPRINTF("INVALID HERE[%d]\n", 0);
    //         EXPRINTCHK(pBottom);
    //         EXPRINT0(" flag: %lld\n", flag);
    //         EXPRINTF("[%d]: %d\n", getMyTid(), path);
    //         EXPRINT0(" oldPB: %d pBottom: %d\n", oldPB, pBottom);
    //         EXPRINTCHK(oldPB);
    getChunkFromIdx(pBottom)->Unlock();
    EXPRINT0("EraseFromLevel for bottom false for %d\n", key);
    // debugPath(path, debug, debug_k);
    //         EXPRINT0("Early Ret![%d]\n", key);
    //         flag_ptr = searchSlow(key);
    //         flag = (flag_ptr & 0x0100000000UL);
    //         path = (flag_ptr & 0x00ffffffffUL);
    //         EXPRINT0(" restarted searchSlow flag: %lld\n", flag);
    //         EXPRINTF("[%d]: %d\n", getMyTid(), path);
    //         pBottom = getPathFromTid(path, 0);
    //         EXPRINTCHK(pBottom);
    //         EXPRINT0(" new pBottom: %d\n", pBottom);
    return false;
  }

  int height = getHeight();

  for (int level = 1; level <= height; level++) {
    ChunkIdx pEnclose = getPathFromTid(path, level);
    bool present = eraseFromLevel(level, pEnclose, key, stats);
    getChunkFromIdx(pEnclose)->Unlock();
    if (!present) {
      //             EXPRINT0("key %d done?\n", key);
      break;
    }
    if (pEnclose == CHK_INVALID)
      EPRINTF("INVALID HERE[%d]\n", 0);
  }

  if (pBottom == CHK_INVALID)
    EPRINTF("INVALID HERE[%d]\n", 0);
  getChunkFromIdx(pBottom)->Unlock();
  return true;
}

__device__ bool GFSL::eraseFromLevel(int level, ChunkIdx &pEnc, K key,
                                     SkiplistStats *stats) {
  pEnc = findAndLockEnclosing(pEnc, key, level, stats);
  if (pEnc == CHK_INVALID)
    EPRINTF("INVALID HERE[%d]\n", 0);
  //     Chunk* chunkPtr = getChunkFromIdx(pEnc);
  KV chunkKV = getChunkFromIdx(pEnc)->read();
  int numKeys = occupiedSlots(chunkKV);
  if (!chunkContains(key, chunkKV))
    return false;

  if (isLastChunk(chunkKV) || numKeys - 1 > MERGE_THRESHOLD) {
    // remove, no merge, don't mark zombie
    executeDelete(pEnc, chunkKV, numKeys, key, level);
  } else {
    // merge with next, mark zombie
    mergeDelete(pEnc, chunkKV, numKeys, key, level, stats);
  }
  return true;
}

__device__ void GFSL::executeDelete(ChunkIdx chunkIdx, KV chunkKV, int filled,
                                    K key, uint32_t level) {
  // EPRINT0("Execute deleting [%d]\n", key);
  int tid = getMyTid();
  if (chunkIdx == CHK_INVALID)
    EPRINTF("INVALID HERE[%d]\n", 0);
  Chunk *chunkPtr = getChunkFromIdx(chunkIdx);
  KV rightKV = getChunkValFromRightNeighbor(chunkKV);
  int delPos = getTidWithKey(key, chunkKV);
  if (delPos == TID_NONE) {
    // Should NOT happen, raise error;
    EPRINTF("BIG ERROR HERE key: [%d]\n", key);
    return;
  }
  if (tid == filled - 1)
    rightKV = {POS_INF, CHK_INVALID};

  if (delPos == filled - 1 && !isLastChunk(chunkKV)) {
    // Update Max here
    KV left;
    if (filled != 1) {
      left = getChunkValFromLeftNeighbor(chunkKV);
    } else {
      // This could only happen if this was the last chunk
      left = {POS_INF, CHK_INVALID};
      // decrementNumChunksAtLevel(level);
    }
    // Thread delPos holds the second max value
    if (tid == delPos)
      chunkPtr->AtomicKWrite(TID_NEXT, left.kv.key);
  }
  for (int i = delPos; i <= filled - 1; i++) {
    if (tid == i)
      chunkPtr->AtomicKVWrite(i, rightKV);
  }
  __syncwarp();
}

__device__ void GFSL::mergeDelete(ChunkIdx chunkIdx, KV chunkKV, int filled,
                                  K key, uint32_t level, SkiplistStats *stats) {
  // EPRINT0("Merge deleting [%d]\n", key);
#if defined(ENABLE_STATS)
  stats->recordMerge();
#endif
  if (chunkIdx == CHK_INVALID)
    EPRINTF("INVALID HERE[%d]\n", 0);
  Chunk *currentPtr = getChunkFromIdx(chunkIdx);
  ChunkIdx next = findAndLockNextNonZombie(chunkIdx, level, stats);
  if (next == CHK_INVALID)
    EPRINTF("INVALID HERE[%d]\n", 0);
  Chunk *nextPtr = getChunkFromIdx(next);
  KV nextKV = nextPtr->read();
  int delPos = getTidWithKey(key, chunkKV);
  int numKeys = occupiedSlots(nextKV);
  if (numKeys + filled - 1 > DSIZE) {

    // Split the next Chunk here
    ChunkIdx next2next = this->preSplit(next, level, stats);
    KV splitKv = this->splitCopy(next, next2next, numKeys);
    if (next2next == CHK_INVALID)
      EPRINTF("INVALID HERE[%d]\n", 0);
    this->getChunkFromIdx(next2next)->Unlock();
  }
  // Shift the items in the next chunk
  nextPtr->ShiftData(filled - 1);

  // Copy the items (except deleted) from current to the next chunk
  K min_key;
  K max_key;
  currentPtr->CopyDataTo(nextPtr, delPos, min_key, max_key);
  // mark current as zombie, unlock next
  currentPtr->MarkZombie();
  if (numKeys + filled - 1 <= DSIZE)
    decrementNumChunksAtLevel(level);
  // Update DownPtrs of above level, of the moved Keys -- how to do this tho
  updateDownPtrs(level + 1, static_cast<V>(next), min_key, max_key, stats);
  nextPtr->Unlock();
}

__device__ void GFSL::updateDownPtrs(int level, V new_val, K min_key, K max_key,
                                     SkiplistStats *stats) {
  if (level >= TEAM_SIZE)
    return;
  EPRINT0("update called with (%d, %d, %d, %d)\n", level, new_val, min_key,
          max_key);
  int tid = getMyTid();
  // ChunkIdx useless;
  // K useless2;
  ChunkIdx path =
      (searchSlow(min_key, /* useless, useless2,*/ stats) & 0x00ffffffffUL);
  ChunkIdx current = getPathFromTid(path, level);
  // ChunkIdx current = firstChunkAtLevel(level);
  while (1) {
    if (current == CHK_INVALID)
      EPRINTF("INVALID HERE[%d]\n", 0);
    Chunk *currentPtr = getChunkFromIdx(current);
    KV chunk = currentPtr->read();
    // This makes contains slower, maybe cause some keys are left unupdated

    if (isZombie(chunk)) {
#if defined(ENABLE_STATS)
      incrementLateralMv(level, stats);
#endif
      current = getPtrFromTid(TID_NEXT, chunk);
      if (current == CHK_INVALID)
        break;
    }
    int size = occupiedSlots(chunk);
    bool predicate =
        (tid < size) && (chunk.kv.key >= min_key) && (chunk.kv.key <= max_key);
    bool all_pred = __ballot_sync(getTeamMask(), predicate);
    if (all_pred) {
      currentPtr->Lock(stats, current, level);
      chunk = currentPtr->read();
      predicate = (tid < size) && (chunk.kv.key >= min_key) &&
                  (chunk.kv.key <= max_key);

      if (predicate)
        currentPtr->AtomicVWrite(tid, new_val);

      // bool all_updated = !predicate && tid == size-1;
      // all_updated = __shfl_sync(getTeamMask(), all_updated, size-1);
      // if(all_updated){
      //     currentPtr->Unlock();
      //     break;
      // }

      // bool all_updated = (chunk.kv.key > max_key) && tid == 0;
      // all_updated = __shfl_sync(getTeamMask(), all_updated, 0);
      // if(all_updated){
      //     currentPtr->Unlock();
      //     break;
      // }
      currentPtr->Unlock();
    }
    int max_val_chunk = __shfl_sync(getTeamMask(), chunk.kv.key, TID_NEXT);
    if (max_val_chunk > max_key)
      break;
#if defined(ENABLE_STATS)
    incrementLateralMv(level, stats);
#endif
    current = getPtrFromTid(TID_NEXT, chunk);
    if (current == CHK_INVALID)
      break;
  }
}

__device__ ChunkIdx GFSL::findAndLockNextNonZombie(ChunkIdx pEnc, int level,
                                                   SkiplistStats *stats) {
  int tid = getMyTid();
  if (pEnc == CHK_INVALID)
    EPRINTF("INVALID HERE[%d]\n", 0);
  Chunk *initialChunkPtr = getChunkFromIdx(pEnc);
  KV initialKV = initialChunkPtr->read();
#if defined(ENABLE_STATS)
  incrementLateralMv(level, stats);
#endif
  ChunkIdx current = getPtrFromTid(TID_NEXT, initialKV);
  while (1) {
    if (current == CHK_INVALID)
      break;
    KV currKV = getChunkFromIdx(current)->read();
    while (isZombie(currKV)) {
#if defined(ENABLE_STATS)
      incrementLateralMv(level, stats);
#endif
      current = getPtrFromTid(TID_NEXT, currKV);
      if (current == CHK_INVALID)
        EPRINTF("INVALID HERE[%d]\n", 0);
      currKV = getChunkFromIdx(current)->read();
    }
    if (current == CHK_INVALID)
      EPRINTF("INVALID HERE[%d]\n", 0);
    this->getChunkFromIdx(current)->Lock(stats, current, level);
    currKV = getChunkFromIdx(current)->read();
    if (isZombie(currKV)) {
      // getChunkFromIdx(current)->Unlock();
#if defined(ENABLE_STATS)
      incrementLateralMv(level, stats);
#endif
      current = getPtrFromTid(TID_NEXT, initialKV);
    } else {
      // redirect next pointer for original chunk
      initialChunkPtr->UpdateNextVal(static_cast<V>(current));
      break;
    };
  }
  return current;
}

__device__ ChunkIdx GFSL::findAndLockEnclosing(ChunkIdx pEnc, K key, int level,
                                               SkiplistStats *stats) {
  ChunkIdx current = pEnc;
  while (1) {
    if (current == CHK_INVALID)
      EPRINTF("INVALID HERE[%d]\n", 0);
    KV currKV = getChunkFromIdx(current)->read();
    while (isZombie(currKV) || getTidForNextStep(key, currKV) == TID_NEXT) {
      PRINT0("zombie/next: %d\n", 1);
      // if (getMyTid() == 0) printf("3");
#if defined(ENABLE_STATS)
      incrementLateralMv(level, stats);
#endif
      current = getPtrFromTid(TID_NEXT, currKV);
      if (current == CHK_INVALID) {
        EPRINTALLKV(currKV);
        EPRINTF("INVALID HERE[%d] isZombie: [%d]\n", 0, isZombie(currKV));
      }
      currKV = getChunkFromIdx(current)->read();
    }
    // if (getMyTid() == 0) printf("4");
    // Now current should be enclosing, try to lock
    // rPRINTCHK(current);
    if (current == CHK_INVALID)
      EPRINTF("INVALID HERE[%d]\n", 0);
    this->getChunkFromIdx(current)->Lock(stats, current, level);
    // if (getMyTid() == 0) printf("5");
    // Locked, check again if still enclosing and non-zombie
    currKV = getChunkFromIdx(current)->read();
    if (isZombie(currKV) || getTidForNextStep(key, currKV) == TID_NEXT) {
      if (current == CHK_INVALID)
        EPRINTF("INVALID HERE[%d]\n", 0);
      this->getChunkFromIdx(current)->Unlock();
    } else
      break;
  }
  return current;
}

__device__ void Chunk::MarkZombie() {
  int tid = getMyTid();
  if (tid == TID_LOCK) {
    // uint64_t oldVal = lock, oldValOut;
    // while(1){
    //     if(oldVal != LOCK_LOCKED) continue;
    //     oldValOut = atomicCAS(reinterpret_cast<unsigned long long*>(&(lock)),
    //     oldVal, LOCK_ZOMBIE);
    //     __threadfence_system();
    //     printf("Mark Someone's here after: %d\n", lock);
    //     if(oldValOut == oldVal) break;
    // }
    // printf("Mark Someone's here before: %d\n", this->lock);
    atomicExch(reinterpret_cast<unsigned long long *>(&this->lock),
               LOCK_ZOMBIE);
    //         __threadfence();
    // printf("Mark Someone's here after: %d\n", this->lock);
  }
  __syncwarp();
  __threadfence();
}

__host__ void GFSL::print(bool uvm) {
  using std::cout;
  using std::endl;

  cout << "Head Array: " << endl;
  uint64_t *head_ptr = head->all_data_ptr();
  if (!uvm) {
    head_ptr = new uint64_t[TEAM_SIZE];
    cudaMemcpy(reinterpret_cast<void *>(head_ptr), head->all_data_ptr(),
               TEAM_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  }
  cout << "Ctrs: ";
  for (int i = 0; i < TEAM_SIZE; i++) {
    KV kv = {.raw = head_ptr[i]};
    cout << (kv.kv.key) << " ";
  }
  cout << endl;
  // cout << "Ptrs: ";
  // for(int i = 0; i < TEAM_SIZE; i++){
  //     KV kv = {.raw = head_ptr[i]};
  //     cout << (kv.kv.value) << " ";
  // }
  // cout << endl;
  // cout << "Data Arrays:"<< endl;
  // Chunk* pool = memory_pool;
  // if(!uvm){
  //     pool = new Chunk[pool_size];
  //     cudaMemcpy(reinterpret_cast<void*>(pool), memory_pool,
  //     pool_size*sizeof(Chunk), cudaMemcpyDeviceToHost);
  // }

  // for(int i = TEAM_SIZE - 1; i >= 0; i--){
  //     KV cur = {.raw = head_ptr[i]};
  //     if(cur.kv.key == 0) continue;
  //     ChunkIdx next  = cur.kv.value;
  //     cout << "Level " << i << "::\n";
  //     while(1){
  //         cout << next << ":: ";
  //         Chunk chunk = pool[next];
  //         uint64_t* ptr = chunk.all_data_ptr();
  //         for(int j = 0; j < 32; j++){
  //             KV kv = {.raw = ptr[j]};
  //             uint32_t key = kv.kv.key;
  //             uint32_t val = kv.kv.value;
  //             cout << key << ":" << val << " ";
  //         }
  //         cout << endl << endl;
  //         next = KV{.raw = ptr[TID_NEXT]}.kv.value;
  //         if(next == CHK_INVALID){
  //             cout << endl;
  //             break;
  //         }
  //     }
  // }
  // if(!uvm){
  //     delete[] head_ptr;
  //     delete[] pool;
  // }
}

__device__ void GFSL::dumpList() {
  if (getMyTid() == 0)
    printf("\n\nStarting list dump...\n");
  int height = getHeight();
  int teamMask = getTeamMask();
  int tid = getMyTid();
  if (getMyTid() == 0)
    printf("head: %p\n height: %d\n total nodes: %d\n", head, height,
           num_allocated);
  head->printChunk();
  while (height >= 0) {
    if (getMyTid() == 0)
      printf("\n\n---\n\nheight %d:\n", height);
    ChunkIdx pIter = firstChunkAtLevel(height);
    do {
      KV currKv = getChunkFromIdx(pIter)->read();
      if (getMyTid() == 0)
        printf("idx: %d\n", pIter);
      printf("[%2d] k: %10u v: %10u\n", tid, currKv.kv.key, currKv.kv.value);
      pIter = __shfl_sync(teamMask, currKv.kv.value, TID_NEXT, TEAM_SIZE);
      if (getMyTid() == 0)
        printf("[%d]\n", 0);
    } while (pIter != CHK_INVALID);
    height--;
  }
}

__device__ void Chunk::printChunk() {
  //     printf("kms ");
  KV mykv = this->read();
  printf("[%d] [%2d:%p]%10u %10u\n",
         (blockIdx.x * blockDim.x + threadIdx.x) / 32, getMyTid(), this,
         mykv.kv.key, mykv.kv.value);
}

__device__ void GFSL::debugPath(ChunkIdx segment, ChunkIdx debug, K debug_k) {
  // assigning team mask (uin32_t) to integer
  // int teamMask = getTeamMask();
  uint32_t teamMask = getTeamMask();
  for (int i = TEAM_SIZE - 1; i >= 0; i--) {
    ChunkIdx curr = __shfl_sync(teamMask, segment, i, TEAM_SIZE);
    ChunkIdx chosen = __shfl_sync(teamMask, debug, i, TEAM_SIZE);
    K chosenk = __shfl_sync(teamMask, debug_k, i, TEAM_SIZE);
    EXPRINT0("\n\n\nsegment %d, chkidx %d\n", i, curr);
    // VIPIN: lead to lot of printing, so commented out
    // getChunkFromIdx(curr)->printChunk();
    EPRINTCHK(curr);
    EXPRINT0("pair chosen: [%d:%d]\n", chosenk, chosen);
  }
}

__device__ bool GFSL::deleteCorrectness(K key) {
  int tid = getMyTid();
  uint32_t teamMask = getTeamMask();
  int maxheight = getHeight();
  for (int i = 0; i <= maxheight; i++) {
    int height = (blockIdx.x * (threadIdx.x / 32) + i) % (maxheight + 1);
    ChunkIdx curr = firstChunkAtLevel(height);
    int count = 0;
    while (curr != CHK_INVALID) {
      count++;
      if (count > 1000000) {
        EXPRINT0("curr: %d\n", curr);
      }
      KV currKv = getChunkFromIdx(curr)->read();
      uint32_t match =
          __ballot_sync(teamMask, currKv.kv.key == key) &&
          __shfl_sync(teamMask, currKv.raw != LOCK_ZOMBIE, TID_LOCK, TEAM_SIZE);
      if (match)
        return false;
      curr = __shfl_sync(teamMask, currKv.kv.value, TID_NEXT, TEAM_SIZE);
    }
    if (height == 0)
      EXPRINT0(" finished layer %d\n", height);
  }
  return true;
}
