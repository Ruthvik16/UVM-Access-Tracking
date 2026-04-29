
NVCCFLAGS = -Wno-deprecated-gpu-targets -O3 -std=c++17 -Xptxas -O3
base:
	nvcc $(NVCCFLAGS) -o driver_hashtable_UVM.out driver_hashtable_UVM.cu

clean:
	rm -f *.out
