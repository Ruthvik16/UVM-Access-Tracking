
all:
	nvcc -o cudamalloc-sleep.out    cudamalloc-sleep.cu     &
	nvcc -o uvm-thrashing.out       uvm-thrashing.cu        &
	nvcc -o fault_noise.out         gpu_fault_noise.cu      &
	nvcc -o sched_noise.out         gpu_scheduling_noise.cu &
	rm -f stop.txt
	echo "Multi-threaded compilation: check for completion with ps"
	sleep 2
	ps

clean:
	rm -f stop.txt *.out

base:   all
