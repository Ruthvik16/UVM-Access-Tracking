CUDA_INSTALL_PATH ?= /usr/local/cuda/
GCC_VER = -5

CXX := /usr/bin/g++
CC := /usr/bin/gcc
CCPATH := gcc

NVCC  := $(CUDA_INSTALL_PATH)bin/nvcc -ccbin $(CCPATH)
LINK := $(NVCC)

INCLUDES := -I. -I$(CUDA_INSTALL_PATH)include
# Libraries
LIB_CUDA := -L/usr/lib/nvidia-current -lcuda

# Options
NVCCOPTIONS := -std=c++11 -O3
CXXOPTIONS = -std=c++11 -O3

COMMONFLAGS += $(INCLUDES) 
NVCCFLAGS += $(NVCCOPTIONS)  $(COMMONFLAGS) -Wno-deprecated-gpu-targets
CXXFLAGS += $(COMMONFLAGS) $(CXXOPTIONS)
CFLAGS += $(COMMONFLAGS)
TARGET = bfs
OBJS =  graph.c.o bfsCPU.c.o bfs.cu.o bfsCUDA.cu.o
LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(LIB_CUDA) -Wno-deprecated-gpu-targets



$(TARGET): clean $(OBJS) 
	$(LINKLINE)

%.cu.o: %.cu 
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.c.o: %.cpp 
	$(CXX) $(CXXFLAGS) -c $< -o $@

debug: clean
debug: NVCCFLAGS += -g -O0
debug: CXXFLAGS += -g -O0
debug: LINKLINE += -g -O0
debug: $(TARGET)

.PHONY: clean

clean:
	rm -rf $(TARGET) *.o 
