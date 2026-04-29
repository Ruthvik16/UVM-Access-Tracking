// DO NOT RENAME
// This needs to be backward compatible. Make a v2 file, or add a definition, but don't rename.
// 1 TAB -> 4 columns
/*
	This contains:

	CUDA kernels:
		increment_float_array_from_gpu()
		increment_int_array_from_gpu()
		increment_float_array_from_gpu_slowly()

		initialize_array_in_gpu()
	(todo) read kernels, access from CPU

	CHECK_RETURN_VALUE(stmt) - checks that a statement returns cudaSuccess
*/

#ifndef __CUDA_MACROS_V1__
#define __CUDA_MACROS_V1__

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/time.h>

#include <iostream>
using namespace std;

// Initializes a float array given a data size (not num_elements)
// Use initialize_array_in_gpu to avoid confusion. initialize_gpu is 
// left for backward compatibility
#define initialize_array_in_gpu initialize_array

static
__global__
void initialize_array(float *fl_array, long int array_size, float value) {
	long idx = blockIdx.x * blockDim.x + threadIdx.x;

	// This warp accesses bytes (idx*4) through (idx*4 + 3)
	if (idx * sizeof(float) + (sizeof(float) - 1) > (array_size))
		return;

	fl_array[idx] = value;
}


// Increments all elements in a float array by 1
static
__global__
void increment_float_array_from_gpu(float *fl_array, long int array_len) {
	long idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx > array_len)
		return ;

	fl_array[idx] += 1.0;
}

// Increments all elements in a int array by 1
static
__global__
void increment_int_array_from_gpu(int *int_array, long int array_len) {
	long idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx > array_len)
		return ;

	int_array[idx] += 1;
}

// Increments all elements in a float array by 1 AND takes a LOT OF TIME
static __global__ void increment_float_array_from_gpu_slowly(
		float *fl_array,
		long int array_len,
		int num_iterations		// keep this at about 2^20. 
								// And replace with long int if needed
) {
	long idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx > array_len)
		return ;

	for (int i = 0; i < num_iterations; i++) {
		fl_array[idx] += (1 / num_iterations);
	}
}


// For CUDA programs - compares with cudaSuccess
#define CHECK_RETURN_VALUE(stmt)                                    \
    do {                                                            \
        cudaError_t ret = (stmt);                                   \
        if ((stmt) != cudaSuccess) {                                \
            printf("Error in %s line %d: (%d) %s\n",                \
                    #stmt, __LINE__, ret, cudaGetErrorString(ret)); \
            exit(1);                                                \
        }                                                           \
    } while (0);

// From Tyler Allen's uvm-eval code artifact
#define CHECK_CUDA_ERROR()                                                    \
{                                                                             \
    cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess)                                                   \
    {                                                                         \
        printf("error=%d name=%s at "                                         \
                "ln: %d\n", err, cudaGetErrorString(err), __LINE__);          \
        exit(1);                                                              \
    }                                                                         \
}

#define KPROF_PROCFS "/proc/bkbuvm/"
/**
 * Squidward profiling modes.
 * Shared by SpongeBob and Mickey.
 */

#define SQUIDWARD_ENABLED 1
#define SQUIWARD_FAULTS         (1 << 4)
#define SQUIDWARD_EVICTION      (1 << 5)
#define SQUIDWARD_FAULT_LOC     (1 << 6)

#define SPONGEBOB_ACC_DIST      (1 << 12)
#define SPONGEBOB_EV_OVERLAP    (1 << 13)
#define SPONGEBOB_PREFETCH_EFF  (1 << 14)
#define SPONGEBOB_LRU_LIFETIME  (1 << 15)

#define SQUIDWARD_MICKEY_AUTOSELECT (1 << 20)

#define SQUIDWARD_PROFILE_ALL ((unsigned) 0xffffffff)
#define SQUIDWARD_DEFAULT SQUIDWARD_PROFILE_ALL

static
void squidward_report_buf_name(void *addr, long nbytes,
                               const char *ptr_n, int mode) {
    int fd = open(KPROF_PROCFS "/squidward", O_RDWR);
    char wbuf[80];
    assert((((long) addr) >> 47) == 0);
    assert(ptr_n);

    if (fd < 0) {
        return;
    }

    int ret = snprintf(wbuf, 79, "n %lx %s %u\n", (long) addr, ptr_n, mode);
    assert(ret < 76);
    assert(wbuf[ret] == '\0');

    int ret2 = write(fd, wbuf, ret + 1);
    if (ret2 != ret + 1) {
        fprintf(stderr, "Wrote char[%d] to squidward: %s\n", ret, wbuf);
        fprintf(stderr, "Retval: %d\n", ret2);
        exit(ret2);
    }
}


static int __ctr_begin = __COUNTER__;

#define UVM_ALLOC_BUF2(dtype, ptr_name, ptr_string, data_req)               \
    do {                                                                    \
        unsigned long __data_sz = (data_req >> 21? data_req : (1 << 21));   \
        int this_buf = __COUNTER__ - __ctr_begin;                           \
        const char *buf_name = &ptr_string[0];                              \
        CHECK_RETURN_VALUE(cudaMallocManaged(                               \
                    (void **) &ptr_name,                                    \
                    __data_sz                                               \
                    )                                                       \
                );                                                          \
        printf("# " #dtype " %s[%ld] = cudaMallocManaged(%ld MiB)"          \
                " at %p\t(buffer #%d)\n",                                   \
                buf_name,                                                   \
                (long int) ((data_req)/sizeof(dtype)),                      \
                (long int) (__data_sz) >> 20,                               \
                ptr_name, this_buf);                                        \
        assert(this_buf <= MAX_BUFS);                                       \
        insert_hints(ptr_name, (char *) buf_name, __data_sz, this_buf);     \
        squidward_report_buf_name(ptr_name, data_req, buf_name, -1);        \
    } while (0);

#define UVM_ALLOC_BUF(dtype, ptr_name, data_req)                            \
    UVM_ALLOC_BUF2(dtype, ptr_name, #ptr_name, data_req)

#define UVM_ALLOC_ARR(dtype, ptr_name, n_memb)                              \
    UVM_ALLOC_BUF2(dtype, ptr_name, #ptr_name, ((n_memb) * (sizeof(dtype))))

#define NONUVM_ALLOC_BUF(dtype, ptr_name, data_size)                        \
    do {                                                                    \
        CHECK_RETURN_VALUE(cudaMalloc(                                      \
                    (void **) &ptr_name,                                    \
                    data_size                                               \
                    )                                                       \
                );                                                          \
        printf("# " #dtype " " #ptr_name "[%ld] = cudaMalloc(%ld MiB)"      \
                " at %p\n",                                                 \
                (long int) ((data_size)/sizeof(dtype)),                     \
                (long int) (data_size) >> 20,                               \
                ptr_name);                                                  \
    } while (0);

#define NONUVM_ALLOC_ARR(dtype, ptr_name, n_memb)                   \
    NONUVM_ALLOC_BUF(dtype, ptr_name, ((n_memb) * (sizeof(dtype))))

// Deprecated
#define TRY_ALLOC_NON_UVM_BY_DATA_SIZE(dtype, ptr_name, data_size)          \
    dtype *ptr_name;                                                        \
    NONUVM_ALLOC_BUF(dtype, ptr_name, data_size)

// Deprecated
#define TRY_ALLOC_UVM_BY_DATA_SIZE(dtype, ptr_name, data_size)              \
    dtype *ptr_name;                                                        \
    UVM_ALLOC_BUF2(dtype, ptr_name, #ptr_name, (data_size));

// Deprecated
#define TRY_ALLOC_UVM(datatype, ptr_name, length_of_array)          \
    datatype *ptr_name;                                             \
    UVM_ALLOC_BUF2(dtype, ptr_name, #ptr_name, (data_size) * sizeof(datatype));

// Deprecated
#define TRY_ALLOC_NON_UVM(datatype, ptr_name, length_of_array)              \
    TRY_ALLOC_NON_UVM_BY_DATA_SIZE(datatype, ptr_name,                      \
            ((length_of_array)*sizeof(datatype)));

#define TRY_DEVICE_SYNCHRONIZE()                                            \
    CHECK_RETURN_VALUE(cudaDeviceSynchronize());



// Deprecated
#define FOR_EACH_ARGUMENT                                           \
    for (int iteration_counter = 1; iteration_counter < argc; iteration_counter++)

// Flags that don't expect options: -stride-4k
#define CHECK_ARG_AND_SET_VAL(str_val, flag, value)             \
    GET_BOOL_FLAG(iteration_counter, str_val, flag, value)

// Flags that expect options: -o <file>
#define CHECK_ARG_AND_SET_PARAM(str_val, param_name)            \
    GET_INT_FLAG(iteration_counter, str_val, param_name)

// These two take the loop counter as an argument. Better than mysterious hidden
// variables.
// Flags that do not expect options: --dry-run
#define GET_BOOL_FLAG(i, flag, variable, val)   \
    if (strcmp(argv[i], flag) == 0) {           \
        variable = val;                         \
        continue;                               \
    } else if (strncmp(argv[i], "-h", 2) == 0) {\
        printf(flag "\tsets " #variable "\tto " \
                #val "\n");                     \
    }

// Flags that expect options: -o <file>
#define GET_INT_FLAG(i, flag, variable)                 \
    if (strcmp(argv[i], flag) == 0) {                   \
        if (!argv[i+1] || atol(argv[i+1]) == 0)         \
            return printf("Usage: " flag " <value>\n"); \
        variable = (long) atol(argv[i+1]);              \
        i++;                                            \
        continue;                                       \
    } else if (strcmp(argv[i], "-h") == 0) {            \
        fprintf(stderr, flag " <int_val>\tsets "        \
            #variable "\t(default: %ld)\n",             \
            (long) variable);                           \
    }

#define UNRECOGNIZED_ARGUMENT(i)                        \
    if (strcmp(argv[i], "-nohint") == 0                 \
        || (i == 1 && atol(argv[i]))                    \
        || strcmp(argv[i], "-h") == 0                   \
        || i == 0) {                                    \
    } else {                                            \
        fprintf(stderr, "# Unrecognized argv[%d]: %s\n",\
                i, argv[i]);                            \
    }

#define ELAPSED_TIME(start, end, message)               \
    printf(message ": %.3f ms\n", ((float)((end - start) * 1000) / CLOCKS_PER_SEC));

static double gettime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}
/*
 * If the program logic is such that 3N^2 ints are allocated, and you want to 
 * use 2 GB of data, then do:
 * array_len = get_data_size((2UL<<30)/3, sizeof(int), 2);
 */
static inline unsigned long get_data_size(
                unsigned long target_size,
                unsigned sizeof_type,
                unsigned nth_root
                )
{
    if (nth_root < 2 || nth_root > 3)
        return printf("%s:%u Can't process arguments %lu and %u\n", __FILE__, 
                       __LINE__, target_size, nth_root);
    unsigned long target2 = target_size / sizeof_type;
    float target2_f = (float) target2;
    float s = (nth_root == 2) ? sqrt(target2_f) : cbrt(target2_f) ;
    target2 = (unsigned long) s;

    // align to 4K
    if (target2 >= (1UL << 25))
        target2 = (target2 & 0xffffffffff000);
    return target2;
}

/*
   INDENTATION PROBLEMS ??
   Sorry for this mess. We aren't changing it atm to avoid lengthy 
   git diffs.
   Set indentation to 4 spaces to view these parts.
   In ViM, it's :set ts=4 :set shiftwidth=4
*/
#define CHECK_ARGC_SHOW_USAGE(min_valid_argc, message)              \
    if (argc < (min_valid_argc)) {                                  \
        fprintf(stderr, "Usage: %s\n", message);                    \
        exit(1);                                                    \
    }

static
int retval_cuda_macros_header;
#define CHECK_RETURN_VALUE_ZERO(stmt)                               \
    retval_cuda_macros_header = ((int) stmt);                       \
    if (retval_cuda_macros_header != 0) {                           \
        printf("Error in %s line %d\n", #stmt, __LINE__);           \
        return retval_cuda_macros_header;                           \
    }


#define TOUCH_ARRAY(start_ptr, nbytes)                      \
    do {                                                    \
        typeof(start_ptr[0]) touch_array_sum =              \
            (typeof(start_ptr[0])) 0;                       \
        for (unsigned long iii = 0;                         \
             iii * sizeof(start_ptr[0]) < (nbytes);         \
             iii ++) {                                      \
            touch_array_sum += start_ptr[iii];              \
        }                                                   \
        if (touch_array_sum == (typeof(start_ptr[0]))       \
                    314159)                                 \
            printf("Fake control dependency on" #start_ptr  \
                    "to touch pages\n");                    \
    } while (0);


/* CUDA HINTS (with a print statement) */
#define CUDA_HINT(ptr, nbytes, hint, where, ptr_n)              \
    do {                                                        \
        clock_t start = clock(), end;                           \
        CHECK_RETURN_VALUE(cudaMemAdvise((void *)(ptr),         \
            (nbytes), hint, where));                            \
        end = clock();                                          \
        printf("# %s(%s(%p), %ld MB, %s): %ld ms\n",            \
            #hint, ptr_n, (ptr), (nbytes)/1000000,              \
            #where,                                             \
            (long)(1000*(float)(end - start))/CLOCKS_PER_SEC);  \
    } while (0);

#define CUDA_PIN_CPU_HINT(ptr, nbytes)      \
    CUDA_HINT(ptr, nbytes, cudaMemAdviseSetPreferredLocation,   \
               cudaCpuDeviceId, #ptr);

#define CUDA_PIN_GPU_HINT(ptr, nbytes)      \
    CUDA_HINT(ptr, nbytes, cudaMemAdviseSetPreferredLocation, 0, #ptr)

#define CUDA_ACCESSED_BY_GPU_HINT(ptr, nbytes)  \
    CUDA_HINT(ptr, nbytes, cudaMemAdviseSetAccessedBy, 0, #ptr)

#define CUDA_READ_MOSTLY_HINT(ptr, nbytes)      \
    CUDA_HINT(ptr, nbytes, cudaMemAdviseSetReadMostly, 0, #ptr)

#define CUDA_PF_ASYNC_GPU2(ptr, ptr_n, nbytes)                                      \
    do {                                                                            \
        clock_t t2, t1 = clock();                                                   \
        CHECK_RETURN_VALUE(cudaMemPrefetchAsync(ptr, (unsigned long) nbytes, 0));   \
        t2 = clock();                                                               \
        printf("# cudaMemPrefetchAsync(%s(%p), %ld MB, GPU): %ld ms\n", ptr_n, ptr, \
               (nbytes / 1000000), (long)(1000 * (float)(t2 - t1)/CLOCKS_PER_SEC)); \
    } while (0);

#define CUDA_PF_ASYNC_GPU(ptr, nbytes)  \
    CUDA_PF_ASYNC_GPU2(ptr, #ptr, nbytes)

#define CUDA_PF_ASYNC_CPU2(ptr, ptr_n, nbytes)                              \
    do {                                                                    \
        clock_t t2, t1 = clock();                                           \
        const char ptr_n[] = #ptr;                                          \
        CHECK_RETURN_VALUE(cudaMemPrefetchAsync(ptr, (unsigned long) nbytes,\
                           cudaCpuDeviceId));                               \
        t2 = clock();                                                       \
        printf("# cudaMemPrefetchAsync(%s(%p), %ld MB, cudaCpuDeviceId): "  \
               "%ld ms\n",                                                  \
               ptr_n, ptr, (nbytes / 1000000),                              \
               (long)(1000 * (float)(t2 - t1)/CLOCKS_PER_SEC));             \
    } while (0);

#define CUDA_PF_ASYNC_CPU(ptr, nbytes)                                      \
    CUDA_PF_ASYNC_CPU2(ptr, #ptr, nbytes)

static void mickey_clear() {
    int fd = open(KPROF_PROCFS "/mickey", O_RDWR);
    unsigned long zero = 0;
    if (fd >= 0)
        assert(write(fd, &zero, 8) == 8);
}

static int mickey_register_va(void *uvm_va) {
    char *disable_mickey = getenv("MICKEY_DISABLE");
    if (disable_mickey && atoi(disable_mickey) > 0)
        return 0;
    int retval, fd = open(KPROF_PROCFS "/mickey", O_RDWR);
    if (fd < 0)
        return fd;

    printf("\t# mickey_register(%p)\n", uvm_va);
    retval = write(fd, &uvm_va, 8);
    if (retval - 8)
        printf("Got retval %d\n", retval);
    assert(retval == 8);
    retval = close(fd);
    assert(retval == 0);
    return 0;
}

#define CUDA_CLI_HINTS

/* CUDA memory hints */
enum cuda_hints_enum {
    HINT_PREFLOC_CPU,
    HINT_PREFLOC_GPU,
    HINT_READ_MOSTLY,
    HINT_READ_MOSTLY_LATE,
    HINT_ACCBY_GPU,
    HINT_ACCBY_GPU_LATE,
    HINT_INIT_CPU,
    HINT_INIT_GPU,
    HINT_PF_OUTPUT,
    HINT_PREFETCH_ASYNC,
    HINT_GOHAN_DYNAMIC,
    HINT_GOHAN_DISABLE,
    HINT_GOHAN_4CYCLES,
    HINT_GOHAN_2CYCLES,
    HINT_GOHAN_0CYCLES,
    HINT_GOHAN_RDUP,
    HINT_GOHAN_STATIC,
    HINT_GOHAN_FPF_RDUP,
    HINT_GOHAN_FPF,
    HINT_COUNT
};

static const char *hint_flags[] = {
    "-prefloc-cpu",
    "-prefloc-gpu",
    "-readmostly",
    "-rm-late",
    "-accby-gpu",
    "-abg-late",
    "-pfcpu",       // PrefLoc CPU only till init.
    "-initgpu",     // Unset PrefLoc after kernel
    "-output-pf",   // prefetch to host after kernel.
    "-prefetch",    // Need to insert manually, after init.
    "-gohan-dynamic",
    "-gohan-disable",
    "-gohan4",
    "-gohan2",
    "-gohan0",
    "-gohan-rdup",
    "-gohan-static",
    "-gohan-fpf-rdup",
    "-gohan-fpf",
    ""
};

#define MAX_BUFS 10
#define HELP_MSG "%-15s <buf_idx | all | 1,3,4,.. >\n"
static bool cuda_hints[MAX_BUFS + 1][HINT_COUNT] = {0};

static void memory_hints_help() {
    printf("\nCUDA Memory hints:\n");
    for (int i = 0; i < HINT_COUNT; i++)
        printf(HELP_MSG, hint_flags[i]);
}

// Returns 1 if matched, else zero
static int __get_hints(int i, char **argv) {
    int j = 0;
    bool all_bufs = (argv[i+1] && strcmp(argv[i+1], "all") == 0);

    for (j = 0; j < HINT_COUNT; j++) {
        if (strcmp(hint_flags[j], argv[i]) != 0)
            continue;

        if (all_bufs) {
            for (int k = 0; k < MAX_BUFS + 1; k++)
                cuda_hints[k][j] = true;
            return 1;
        }

        if (!argv[i+1]) {
            printf("Usage: " HELP_MSG
                   "Get buffer indices from the console\n",
                   argv[i]);
            exit(1);
        }

        char *token = strtok(argv[i+1], ",");
        while (token) {
            int buf_num = atoi(token);
            if (!buf_num) {
                printf("Invalid argv[%d]: token %s\n", i+1, token);
                exit(1);
            }
            assert(buf_num <= MAX_BUFS);
            cuda_hints[buf_num][j] = true;
            token = strtok(NULL, ",");
        }
        return 1;
    }

    if (strcmp(argv[i], "-h") == 0) {
        memory_hints_help();
    }
    return 0;
}

struct gohan_hint {
    unsigned long addr;
    const char *buf_name;

    // Details in driver/prospar/gohan.md
    uint8_t do_custom, do_dynamic;
    uint8_t full_pf_h2d, full_pf_d2h, pin_cpu, pf_mblock_h2d, pf_mblock_d2h;
    uint8_t dont_evict_for_me;  // If an eviction is needed, do a remote mapping.
    uint8_t try_read_dup;
    unsigned eviction_cycles_allowed;
};

static struct gohan_hint gohan_zero = {0};
static struct gohan_hint gohan_simple = {
    .addr       = 0,
    .buf_name   = NULL,
    .do_custom  = 1,
    .do_dynamic = 0,
    .full_pf_h2d    = 1,
    .full_pf_d2h    = 1,
    .pf_mblock_h2d  = 1,
    .pf_mblock_d2h  = 1,
    .dont_evict_for_me = 0,
    .try_read_dup   = 1,
    .eviction_cycles_allowed = 0,
};

static int gohan_madvise_struct(struct gohan_hint *ht) {
    int ret, fd = open(KPROF_PROCFS "/gohan", O_WRONLY);
    char hint[125];
    if (fd < 0) {
        fprintf(stderr, "Gohan disabled!\n");
        return fd;
    }
    ret = sprintf(hint, "%lx %s %u %u | pf %u %u %u %u | rdup %u ev %u "
                  "pincpu %u | %u %u %u %u",
                  ht->addr, ht->buf_name, ht->do_custom, ht->do_dynamic,
                  ht->full_pf_h2d, ht->full_pf_d2h, ht->pf_mblock_h2d, ht->pf_mblock_d2h,
                  ht->try_read_dup, ht->eviction_cycles_allowed, ht->pin_cpu,
                  0, 0, 0, 0
                  );

    assert(ret < 120);
    assert(hint[ret] == '\0');

    // ret = dprintf(fd, (const char *) hint);
    ret = write(fd, hint, ret + 1);
    fprintf(stderr, "gohan_madvise ret %d | %s\n", ret, hint);

    assert(ret >= 20);
    return 0;
}

static int gohan_do_vanilla(unsigned long addr, const char *bufname) {
    struct gohan_hint gh = {
        .addr = addr,
        .buf_name = bufname,
        .do_custom = 0,
        .do_dynamic = 0,
    };

    return gohan_madvise_struct(&gh);
}

static int gohan_do_dynamic(unsigned long addr, const char *bufname) {
    struct gohan_hint gh = {
        .addr = addr,
        .buf_name = bufname,
        .do_custom = 0,
        .do_dynamic = 1,
    };
    return gohan_madvise_struct(&gh);
}

static int gohan_cycles(unsigned long addr, const char *bufname, unsigned cycles) {
    struct gohan_hint gh = {
        .addr = addr,
        .buf_name = bufname,
        .do_custom   = 1,
        .do_dynamic  = 0,
        .full_pf_h2d = 1,
        .full_pf_d2h = 1,
        .pin_cpu = 0,
        .try_read_dup = 1,
        .eviction_cycles_allowed = cycles,
    };
    return gohan_madvise_struct(&gh);
}

static int gohan_rdup_only(unsigned long addr, const char *bufname) {
    struct gohan_hint gh = gohan_zero;

    gh.do_custom = 1,
    gh.do_dynamic = 0,
    gh.addr   = addr;
    gh.buf_name     = bufname;
    gh.try_read_dup = 1;

    return gohan_madvise_struct(&gh);
}

static int gohan_static(unsigned long addr, const char *bufname) {
    struct gohan_hint gh = {
        .addr = addr,
        .buf_name = bufname,
        .do_custom = 1,
        .do_dynamic = 0,
        .full_pf_h2d = 1,
        .full_pf_d2h = 1,
        .pin_cpu = 0,
        .try_read_dup = 1,
        .eviction_cycles_allowed = 5,
    };

    return gohan_madvise_struct(&gh);
}

#define get_hints(i, argv)          \
    if (__get_hints(i, argv)) {     \
        i++;                        \
        continue;                   \
    }


static
void insert_hints(void *ptr, const char *buf_name, long nbytes,
                    int buf_idx)
{
    unsigned long addr = (unsigned long) ptr;
    struct gohan_hint gh = gohan_simple;
    gh.addr = (unsigned long) ptr;
    gh.buf_name = buf_name;

    if (cuda_hints[buf_idx][HINT_PREFLOC_CPU])
        CUDA_HINT(ptr, nbytes, cudaMemAdviseSetPreferredLocation,
                   cudaCpuDeviceId, buf_name)
    if (cuda_hints[buf_idx][HINT_PREFLOC_GPU])
        CUDA_HINT(ptr, nbytes, cudaMemAdviseSetPreferredLocation, 0, buf_name);
    if (cuda_hints[buf_idx][HINT_READ_MOSTLY])
        CUDA_HINT(ptr, nbytes, cudaMemAdviseSetReadMostly, 0, buf_name);
    if (cuda_hints[buf_idx][HINT_ACCBY_GPU])
        CUDA_HINT(ptr, nbytes, cudaMemAdviseSetAccessedBy, 0, buf_name);

    if (cuda_hints[buf_idx][HINT_INIT_CPU])
        CUDA_PF_ASYNC_CPU2(ptr, buf_name, nbytes);

    if (cuda_hints[buf_idx][HINT_GOHAN_DYNAMIC])
        gohan_do_dynamic(addr, buf_name);
    if (cuda_hints[buf_idx][HINT_GOHAN_DISABLE])
        gohan_do_vanilla(addr, buf_name);
    if (cuda_hints[buf_idx][HINT_GOHAN_4CYCLES])
        gohan_cycles(addr, buf_name, 4);
    if (cuda_hints[buf_idx][HINT_GOHAN_2CYCLES])
        gohan_cycles(addr, buf_name, 2);
    if (cuda_hints[buf_idx][HINT_GOHAN_0CYCLES])
        gohan_cycles(addr, buf_name, 0);
    if (cuda_hints[buf_idx][HINT_GOHAN_RDUP])
        gohan_rdup_only(addr, buf_name);
    if (cuda_hints[buf_idx][HINT_GOHAN_STATIC])
        gohan_static(addr, buf_name);
    if (cuda_hints[buf_idx][HINT_GOHAN_FPF_RDUP]) {
        gohan_madvise_struct(&gh);
    }
    if (cuda_hints[buf_idx][HINT_GOHAN_FPF]) {
        gh.try_read_dup = 0;
        gohan_madvise_struct(&gh);
    }
}

static
void post_init_insert_hints(void *ptr, const char *buf_name, long nbytes,
                            int buf_idx)
{
    if (cuda_hints[buf_idx][HINT_READ_MOSTLY_LATE])
        CUDA_HINT(ptr, nbytes, cudaMemAdviseSetReadMostly, 0, buf_name);
    if (cuda_hints[buf_idx][HINT_PREFETCH_ASYNC])
        CUDA_PF_ASYNC_GPU2(ptr, buf_name, nbytes);
    if (cuda_hints[buf_idx][HINT_ACCBY_GPU_LATE])
        CUDA_HINT(ptr, nbytes, cudaMemAdviseSetAccessedBy, 0, buf_name);
    if (cuda_hints[buf_idx][HINT_INIT_GPU])
        CUDA_HINT(ptr, nbytes, cudaMemAdviseSetPreferredLocation, 0, buf_name);
}

static
void post_compute_insert_hints(void *ptr, const char *buf_name, long nbytes,
                               int buf_idx)
{
    if (cuda_hints[buf_idx][HINT_INIT_GPU])
        CUDA_HINT(ptr, nbytes, cudaMemAdviseUnsetPreferredLocation, 0, buf_name);
    if (cuda_hints[buf_idx][HINT_PF_OUTPUT])
        CUDA_PF_ASYNC_CPU2(ptr, buf_name, nbytes);
}

#define HINTS_POST_INIT(ptr, nbytes, buf_idx)   \
    post_init_insert_hints(ptr, #ptr, nbytes, buf_idx)

#define HINTS_POST_COMPUTE(ptr, nbytes, buf_idx)    \
    post_compute_insert_hints(ptr, #ptr, nbytes, buf_idx)

#endif /* __CUDA_MACROS_V1__ */
