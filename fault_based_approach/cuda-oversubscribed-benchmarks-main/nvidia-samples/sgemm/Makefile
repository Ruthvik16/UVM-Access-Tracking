base:
	nvcc -O3 -lcublas -Wno-deprecated-gpu-targets sgemm.cu -o sgemm.out

.PHONY: clean
clean:
	rm sgemm.out
