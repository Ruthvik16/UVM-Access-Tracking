CUDA_DIR ?= /usr/local/cuda

CUDA_LIB_DIR := $(CUDA_DIR)/lib
ifeq ($(shell uname -m), x86_64)
	ifeq ($(shell if test -d $(CUDA_DIR)/lib64; then echo T; else echo F; fi), T)
		CUDA_LIB_DIR := $(CUDA_DIR)/lib64
	endif
endif

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include
NVCCFLAGS := -Wno-deprecated-gpu-targets -O3

SRC = needle.cu

EXE = needle

release: $(SRC)
	$(CC) $(NVCCFLAGS) ${KERNEL_DIM} $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

enum: $(SRC)
	$(CC) ${KERNEL_DIM} -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debug: $(SRC)
	$(CC) ${KERNEL_DIM} -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debugenum: $(SRC)
	$(CC) ${KERNEL_DIM} -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
