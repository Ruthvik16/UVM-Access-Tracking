//polybenchUtilFuncts.h
//Scott Grauer-Gray (sgrauerg@gmail.com)
//Functions used across hmpp codes

#ifndef POLYBENCH_UTIL_FUNCTS_H
#define POLYBENCH_UTIL_FUNCTS_H

//define a small float value
#define SMALL_FLOAT_VAL 0.00000001f

// The cuda-benchmark repo should have this header.
#include "cuda-macros-v1.h"

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


float absVal(float a)
{
	if(a < 0)
	{
		return (a * -1);
	}
   	else
	{ 
		return a;
	}
}



float percentDiff(double val1, double val2)
{
	if ((absVal(val1) < 0.01) && (absVal(val2) < 0.01))
	{
		return 0.0f;
	}

	else
	{
    		return 100.0f * (absVal(absVal(val1 - val2) / absVal(val1 + SMALL_FLOAT_VAL)));
	}
} 

#endif //POLYBENCH_UTIL_FUNCTS_H


/*
PROSPAR

    Adding some helpers to (i) change data size and (ii) skip CPU computation
    based on default params
*/

#define COMPARE_WITH_CPU_DEFAULT 0
#define COPY_BACK_CPU_DEFAULT 1
#define CHECK_CPU_MEMORY_TOUCH_TIME_DEFAULT 0

#ifndef CHECK_ARG_AND_SET_VAL
#define CHECK_ARG_AND_SET_VAL(i, flag, variable, val)   \
        if (strcmp(argv[i], flag) == 0)                 \
                variable = val;                         \
        if (strncmp(argv[i], "-h", 2) == 0) {           \
                printf(flag " sets " #variable " to "   \
                        #val "\n");                     \
        }
#endif

#ifndef TOUCH_ARRAY
// Moved to cuda-macros-v1.h
#define TOUCH_ARRAY(start_ptr, nbytes)                      \
    do {                                                    \
        DATA_TYPE touch_array_sum = 0.0;                    \
        for (unsigned long iii = 0;                         \
             iii * sizeof(DATA_TYPE) < (nbytes);            \
             iii += (4096/sizeof(DATA_TYPE))) {             \
            touch_array_sum += start_ptr[iii];              \
        }                                                   \
        if (touch_array_sum == 3.141592)                    \
            printf("Fake control dependency on" #start_ptr  \
                    "to touch pages\n");                    \
    } while (0);

#endif

/* These variables are used in our modificatons */
static int compare_with_cpu;            // Compare correctness of output. Implies copy_back_gpu_results
static int copy_back_gpu_results;       // To study D2H migration
static int check_cpu_memory_touch_time; // To find baseline CPU page touch time (for results)
