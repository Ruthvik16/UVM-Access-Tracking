#!/bin/bash

# Run this script before committing any changes to dev/cuda-macros-v1.h
# Add tests here after creating any benchmark

# allocations

cd ../allocation
nvcc uvm-allocation-granularity-2M.cu
nvcc non-uvm-allocation-granularity-2M.cu
nvcc access-adjacent-cudamalloc-regions.cu

echo "Checked cuda-oversubscribed-benchmarks/allocation"
rm a.out

