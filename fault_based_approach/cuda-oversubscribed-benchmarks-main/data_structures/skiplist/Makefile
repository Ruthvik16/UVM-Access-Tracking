NVCCFLAGS = -O3 -Wno-deprecated-gpu-targets

base:
	nvcc $(NVCCFLAGS) trace_bm_gfsl_uvm.cu -o skiplist-gfsl.out  -I./include -std=c++17 -DKEY_CHECK -DUVM_PREFETCH_HINT


.PHONY: debug
debug:
	nvcc $(NVCCFLAGS) trace_bm_gfsl_uvm.cu -o skiplist-gfsl.debug.out -I./include -std=c++17 -DKEY_CHECK -DUVM_PREFETCH_HINT -g
