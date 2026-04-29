// FIXME: SB: Remove redundancy across different driver files (e.g., file
// creation and reading logic)

#include <bits/stdc++.h>
#include <cstdint>

#include "include/constants.h"
#include "include/datatypes.h"
#include "include/functions.h"

#include "gpu_impl_UVM.cuh"
#include "cuda-macros-v1.h"

using std::cerr;
using std::cout;

int main(int argc, char *argv[]) {

  if (!getGPUConfig()) {
    cout << "Unable to access GPU configurations\n";
    exit(0);
  }
  for (int i = 1; i < argc; i++) {

    // First check for Gohan and MemAdvise hints. The second set of functions is
    // incompatible with these.
    if (__get_hints(i, argv)) {
      i++;
      continue;
    }

    int error = parse_args(argv[i]);
    if (error == 1) {
      cout << "Argument error, terminating run.\n";
      exit(EXIT_FAILURE);
    }
  }

  std::srand(RANDOM_SEED);
  uint64_t ADD = NUM_OPS * (INSERT / 100.0);
  uint64_t REM = NUM_OPS * (DELETE / 100.0);
  uint64_t FIND = NUM_OPS - (ADD + REM);

  long pre_populate_size = NUM_OPS * (pre_populate_flag / 100.0);
  printf("Prepoluate size is %ld\n", pre_populate_size);

  mickey_clear();
  KeyValue *kvs_ppp = nullptr;
  if (pre_populate_size > 0) {
    UVM_ALLOC_BUF(KeyValue, kvs_ppp, pre_populate_size * sizeof(KeyValue));

    // FIXME: SB: Should we not read from the input traces?
    for (int i = 0; i < pre_populate_size; i++) {
      kvs_ppp[i].key = ((i + 1) * 2);
      kvs_ppp[i].value = ((i + 1) * 2) + 1;
    }
  }

  printf("NUM_OPS: %ld, ADD: %ld, REM: %ld, FIND: %ld\n", NUM_OPS, ADD, REM,
         FIND);

  KeyValue *kvs_insert = nullptr;
  uint32_t *keys_del = nullptr;
  uint32_t *keys_lookup = nullptr;

  UVM_ALLOC_BUF(KeyValue, kvs_insert, sizeof(KeyValue) * ADD);
  mickey_register_va(kvs_insert + (1 << 22));
  if (REM > 0)
    UVM_ALLOC_BUF(uint32_t, keys_del, REM * sizeof(uint32_t));
  if (FIND > 0)
    UVM_ALLOC_BUF(uint32_t, keys_lookup, FIND * sizeof(uint32_t));

  uint64_t add = 0;
  uint64_t rem = 0;
  uint64_t find = 0;

  // TODO: move the fileIO code to common file
  path cwd = std::filesystem::current_path();
  // read the operation list from trace
  if (USE_FILE) {
    if (checkTraceFiles(addTrace, delTrace, findTrace, kvs_insert, keys_del,
                        keys_lookup))
      cout << "Trace file exists, going to read the traces" << std::endl;
    add = ADD;
    rem = REM;
    find = FIND;
  } else { // generate input on run using rand()
    while (add < ADD) {
      // rand()%(max_val-min_val + 1)+min_val to generate in range (min, max)
      kvs_insert[add].key = rand() % ((UINT32_MAX - 1) + 1) + 1;
      kvs_insert[add].value = rand() % ((UINT32_MAX - 1) + 1) + 1;
      add++;
    }
    while (rem < REM) {
      keys_del[rem] = rand() % ((UINT32_MAX - 1) + 1) + 1;
      rem++;
    }
    while (find < FIND) {
      keys_lookup[find] = rand() % ((UINT32_MAX - 1) + 1) + 1;
      find++;
    }
  } // trace into array

  float total_time_gpu = 0.0f;
  float total_init_time_gpu = 0.0f;
  float total_build_time_gpu = 0.0f;
  float total_delete_time_gpu = 0.0f;
  float total_search_time_gpu = 0.0f;

  for (int i = 0; i < runs; i++) {

    float init_time_gpu = 0.0f;
    float build_time_gpu = 0.0f;
    float delete_time_gpu = 0.0f;
    float search_time_gpu = 0.0f;

    uint64_t gpuHTSize = (add / LOAD_FACTOR) + pre_populate_size;
    uint64_t capacity = getCapacityPrime(gpuHTSize);
    // any number smaller than a prime is relatively prime
    smallerPrimeGPU = capacity - 1;
    cout << "HT capacity " << capacity << " second hash" << smallerPrimeGPU
         << "\n";
    // create hashtable
    KeyValue *gt;
    UVM_ALLOC_BUF(KeyValue, gt, capacity * sizeof(KeyValue));
    mickey_register_va(gt + (1 << 24));

    printf("Insert starting\n");
#if defined(DEBUG)
    printf("GPU size is : %ld and vector size is: %ld\n", capacity, add);
#endif

    // MemAdvise: set accessed with prefetching works best
    build_time_gpu = batch_insert_gpu_UVM(gt, kvs_insert, add, capacity);

    // Do not invoke delete kernel if rem is 0
    bool *delete_result = nullptr;
    if (rem) {
      UVM_ALLOC_BUF(bool, delete_result, sizeof(bool) * rem);

      cudaCheckErrorMacro(cudaMemset(delete_result, 0x00, sizeof(bool) * rem),
                          "Initialization of the delete result failed");
      delete_time_gpu =
          batch_delete_gpu(gt, keys_del, delete_result, rem, capacity);

#ifdef DELETE_KEY
      for (int j = 0; j < REM; j++) {
        cout << "DELETE KEY RESULT: " << delete_result[j] << "\t";
      }
      cout << "\n";
#endif
    }

    uint32_t *searched_values = nullptr;
    if (find) { // Do not invoke GPU kernel if find is 0
      UVM_ALLOC_BUF(uint32_t, searched_values, sizeof(uint32_t) * find);
      cudaCheckErrorMacro(
          cudaMemset(searched_values, 0x00, sizeof(uint32_t) * find),
          "Initialization of the search result failed");

      search_time_gpu =
          batch_lookup_gpu(gt, keys_lookup, searched_values, find, capacity);

#ifdef PRINT
      print_gpuHashTable(gt, capacity);
#endif
    }

#ifdef KEY_CHECK
    printf("Calling key check\n");
    Key_Check_GPU(gt, add, kvs_insert, capacity);
#endif

    if (rem) {
      cudaCheckErrorMacro(cudaFree(delete_result),
                          "Unable to free the allocated delete result");
    }

    if (find) {
      // Check the total successful searches
      uint64_t searchInd = 0;
      uint64_t searchSucc = 0;
      for (; searchInd < find; searchInd++) {
        if (searched_values[searchInd])
          searchSucc++;
      }
      printf("Successful searches: %ld Total search queriess: %ld\n",
             searchSucc, find);
      cudaCheckErrorMacro(cudaFree(searched_values),
                          "Unable to free the allocated search result");
    }

    cudaCheckErrorMacro(cudaFree(gt), "Unable to free the hashtable memory");
    total_init_time_gpu += init_time_gpu;
    total_build_time_gpu += build_time_gpu;
    total_delete_time_gpu += delete_time_gpu;
    total_search_time_gpu += search_time_gpu;
    total_time_gpu += (build_time_gpu + delete_time_gpu + search_time_gpu);
  }

  cout << "Total time for initialization (ms): " << (total_init_time_gpu / runs)
       << "\nTotal time taken by insert kernel (ms): "
       << (total_build_time_gpu / runs)
       << "\nTotal time taken by delete kernel (ms): "
       << (total_delete_time_gpu / runs)
       << "\nTotal time taken by search kernel (ms): "
       << (total_search_time_gpu / runs)
       << "\nTotal time taken by HeteroHash kernel (ms): "
       << (total_time_gpu / runs) << "\n";

  cout << "---------- Successful execution of the data structure ----------\n";
}
