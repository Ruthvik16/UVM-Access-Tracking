/*Copyright(c) 2020, The Regents of the University of California, Davis.            */
/*                                                                                  */
/*                                                                                  */
/*Redistribution and use in source and binary forms, with or without modification,  */
/*are permitted provided that the following conditions are met :                    */
/*                                                                                  */
/*1. Redistributions of source code must retain the above copyright notice, this    */
/*list of conditions and the following disclaimer.                                  */
/*2. Redistributions in binary form must reproduce the above copyright notice,      */
/*this list of conditions and the following disclaimer in the documentation         */
/*and / or other materials provided with the distribution.                          */
/*                                                                                  */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND   */
/*ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED     */
/*WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.*/
/*IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,  */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING, BUT */
/*NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR*/
/*PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, */
/*WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) */
/*ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE        */
/*POSSIBILITY OF SUCH DAMAGE.                                                       */
/************************************************************************************/
/************************************************************************************/
/* */

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <random>
#include <vector>

#define SKEWQUERY 0
#define SHFLINS 1

#include "GpuBTree.h"

int main(int argc, char* argv[]) {

  double t_build1, t_build2, t_buildquery2, t_query2, t_validate2;
  // Input number of keys
  long numKeys = 250000000, numKeysMil = 0;

  // Input number of queries
  long numQueries = 128 * 1024 * 1024, numQueriesMil = 0;

  // RNG
  std::random_device rd;
  std::mt19937 g(rd());

  for (int i = 1; i < argc; i++) {
    GET_INT_FLAG(i, "-keys", numKeys);
    GET_INT_FLAG(i, "-keysM", numKeysMil);
    GET_INT_FLAG(i, "-queries", numQueries);
    GET_INT_FLAG(i, "-queriesM", numQueriesMil);
    if (__get_hints(i, argv)) {
      i++;
      continue;
    }
    if (strcmp(argv[i], "-h") == 0)
      return -1;
    UNRECOGNIZED_ARGUMENT(i);
  }

  if (numKeysMil)
    numKeys = numKeysMil * 1000000;
  if (numQueriesMil)
    numQueries = numQueriesMil * 1000000;
  assert(numKeys < (1UL << 31) && numQueries < (1UL << 31));

  GpuBTree::GpuBTreeMap<uint32_t, uint32_t, uint32_t> btree;

  ///////////////////////////////////
  ///		 Build the tree    	  ///
  ///////////////////////////////////

  // Prepare the keys
  t_build1 = gettime();
  std::vector<uint32_t> keys;
  std::vector<uint32_t> values;
  keys.reserve(numKeys);
  values.reserve(numKeys);
  assert(numKeys < (1UL << 31));
  for (long iKey = 0; iKey < numKeys; iKey++) {
    keys.push_back(iKey);
  }

#if SHFLINS
  // shuffle the keys
  std::shuffle(keys.begin(), keys.end(), g);
#endif

  // assign the values
  for (long iKey = 0; iKey < numKeys; iKey++) {
    values.push_back(keys[iKey]);
  }

  t_build2 = gettime();
  printf("Build keys[] values[]: %.3lf s\n", t_build2 - t_build1);

  // Move data to GPU
  // d_keys and d_values are cudaMallocHost allocations. Ideally, d_keys and
  // keys[] should be the same.
  GpuTimer build_timer;
  build_timer.timerStart();
  uint32_t *d_keys, *d_values;

  CHECK_ERROR(memoryUtil::hostAlloc(d_keys, numKeys));
  CHECK_ERROR(memoryUtil::hostAlloc(d_values, numKeys));
  CHECK_ERROR(memoryUtil::cpyToDevice(keys.data(), d_keys, numKeys));
  CHECK_ERROR(memoryUtil::cpyToDevice(values.data(), d_values, numKeys));

  // Build the tree
  t_build1 = gettime();
  btree.insertKeys(d_keys, d_values, numKeys, SourceT::DEVICE);
  build_timer.timerStop();

  CHECK_RETURN_VALUE(cudaDeviceSynchronize());
  CHECK_CUDA_ERROR();
  t_build2 = gettime();
  printf("# Built tree: %.3lf s\n", t_build2 - t_build1);

  ///////////////////////////////////
  ///		 Query the tree       ///
  ///////////////////////////////////

  btree.getAllocator()->bringToHostSide();


  // Prepare the query keys
  std::vector<uint32_t> query_keys;
  std::vector<uint32_t> query_results;

  query_keys.reserve(numQueries );
  query_results.resize(numQueries);

  for (long iKey = 0; iKey < numKeys ; iKey++) {
#if SKEWQUERY
    if(iKey % 25 == 0){
      query_keys.push_back(iKey);
    } else {
      uint32_t fifth = numKeys / 25;
      uint32_t key = iKey % fifth;
      key = 24 * fifth + key;
      query_keys.push_back(key);
    }
#else
    query_keys.push_back(iKey);
#endif
  }

  // shuffle the queries
  std::shuffle(query_keys.begin(), query_keys.end(), g);

  // Move data to GPU
  GpuTimer query_timer;
  query_timer.timerStart();
  uint32_t *d_queries, *d_results;

  UVM_ALLOC_ARR(uint32_t, d_queries, numQueries);
  UVM_ALLOC_ARR(uint32_t, d_results, numQueries);

  CHECK_ERROR(memoryUtil::cpyToDevice(query_keys.data(), d_queries, numQueries));
  memset((void*)d_results, 0, sizeof(uint32_t) * numQueries);

  t_buildquery2 = gettime();
  printf("# build query keys: %.3lf s\n", t_buildquery2 - t_build2);

  btree.searchKeys(d_queries, d_results, numQueries, SourceT::DEVICE);

  CHECK_RETURN_VALUE(cudaDeviceSynchronize());
  t_query2 = gettime();
  printf("# queried tree: %.3lf s\n", t_query2 - t_buildquery2);

  query_timer.timerStop();

  // Copy results back
  CHECK_ERROR(memoryUtil::cpyToHost(d_results, query_results.data(), numQueries));

  // Validate
  long exist_count = 0;
  for (long iKey = 0; iKey < numQueries; iKey++) {
    if (query_keys[iKey] < numKeys) {
      exist_count++;
      if (query_results[iKey] != query_keys[iKey]) {
        printf("Error validating queries (Key = %i, Value = %i) found (Value = %i)\n",
               query_keys[iKey],
               query_keys[iKey],
               query_results[iKey]);
        exit(0);
      }
    } else {
      if (query_results[iKey] != 0) {
        printf(
            "Error validating queries (Key = %i, Value = NOT_FOUND) found (Value = %i)\n",
            query_keys[iKey],
            query_results[iKey]);
        exit(0);
      }
    }
  }

  t_validate2 = gettime();
  printf("# validate operations: %.3lf s\n", t_validate2 - t_query2);

  // output
  printf("SUCCESS. ([%0.2f%%] queries exist in search.)\n",
         float(exist_count) / float(numQueries) * 100.0);

  printf("Build: %ld pairs in %f ms (%0.2f MKeys/sec)\n",
         numKeys,
         build_timer.getMsElapsed(),
         float(numKeys) * 1e-6 / build_timer.getSElapsed());

  printf("Query: %ld pairs in %f ms (%0.2f MKeys/sec)\n",
         numQueries,
         query_timer.getMsElapsed(),
         float(numQueries) * 1e-6 / query_timer.getSElapsed());
  printf("GPU.Parser.Time: %f\n", query_timer.getMsElapsed());

  printf("Tree size: %f GiBs.\n", float(btree.compute_usage()));
  // cleanup
  cudaFree(d_keys);
  cudaFree(d_values);
  cudaFree(d_queries);
  cudaFree(d_results);
  btree.free();
  return 0;
}
