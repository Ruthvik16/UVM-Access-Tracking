all:
	nvcc -Wno-deprecated-gpu-targets -O3 ${CUFILES} ${DEF} -o ${EXECUTABLE} 

debug:
	nvcc -g -Wno-deprecated-gpu-targets ${CUFILES} ${DEF} -o ${EXECUTABLE} 

clean:
	rm -f *~ *.exe
