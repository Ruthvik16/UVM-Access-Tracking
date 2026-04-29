CXXFLAGS=-O3
CUDA_CXX=nvcc
CXXFLAGS += -Wno-deprecated-gpu-targets

cuda-stream:
	$(CUDA_CXX) $(CXXFLAGS) -DCUDA main.cu $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f cuda-stream
