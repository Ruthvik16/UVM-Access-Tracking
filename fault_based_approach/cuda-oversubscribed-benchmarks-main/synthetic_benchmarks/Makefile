
NVCCFLAGS= -O3 -Wno-deprecated-gpu-targets

base:
	nvcc $(NVCCFLAGS) -o int_set_uvm.out        int_set_uvm.cu &
	nvcc $(NVCCFLAGS) -o int_set_nonUvm.out     int_set_nonUvm.cu &
	nvcc $(NVCCFLAGS) -o vector_add.out         vector_add.cu &
	nvcc $(NVCCFLAGS) -o vector_mul.out         vector_mul.cu &
	nvcc $(NVCCFLAGS) -o cpu_init.out           cpu_init.cu &
	nvcc $(NVCCFLAGS) -o gpu_read.out           gpu_read.cu &
	nvcc $(NVCCFLAGS) -o random_acc.out         random_acc.cu &
	make -C stream
	sleep 2

.PHONY: debug
debug: NVCCFLAGS += -g
debug: base


NVCC = /usr/local/cuda/bin/nvcc
.PHONY: gpgpusim
gpgpusim:
	$(NVCC) --version &
	$(NVCC) $(NVCCFLAGS) -o int_set_uvm.out int_set_uvm.cu       -lcudart -ccbin=g++-9 &
	$(NVCC) $(NVCCFLAGS) -o int_set_nonUvm.out int_set_nonUvm.cu -lcudart -ccbin=g++-9 &
	$(NVCC) $(NVCCFLAGS) -o cpu_init.out cpu_init.cu             -lcudart -ccbin=g++-9 &
	sleep 2

.PHONY:clean
clean:
	rm -f *out
	make -C stream clean
