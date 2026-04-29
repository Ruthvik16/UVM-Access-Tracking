# To run for the overhead comparison and the stats. 
sanity_checks () {
    run_command_in_dir 2 polybench/2DCONV 2DConvolution.exe 1543210000 \
        -no-random
    run_command_with_count 1 UVMBench/bfs/main 0 10000000 16543000
    warmup "nvidia-samples/sgemm/sgemm.out" -data 7654321000
    echo "*** end of sanity runs ***"
}

isca_workloads () {
    increasing_data_sizes="n"
    run_command_in_dir 5 polybench/2DCONV 2DConvolution.exe -mb 30000 \
            -phases 3 -copy-back \

    run_command_in_dir 5 synthetic_benchmarks int_set_uvm.out -data 30000000000 \
            -compare -iters 4

    # warmup
    warmup "nvidia-samples/sgemm/sgemm.out" -data 1000000000
    run_command_in_dir 2 nvidia-samples/sgemm sgemm.out \
         -mb 17500 -copy-back

    export TRACE_ROOT=/data/heterods-trace
    warmup hetero-ds-hashtable/driver_hashtable_UVM.out -rns=1 \
        -ops=999900 -add=100 -tra=insert_trace-400e7-100-add-no-dup-SPARSE_UNIQUE.bin
    run_command_in_dir 2 hetero-ds-hashtable driver_hashtable_UVM.out -rns=1 \
        -ops=615000000 -add=100 -tra=insert_trace-400e7-100-add-no-dup-SPARSE_UNIQUE.bin

    run_command_in_dir 2 polybench/MVT mvt.exe -mb 10000 -copy-back
    run_command_in_dir 2 polybench/ATAX atax.exe 10000000000 -copy-back
    run_command_in_dir 2 UVMBench/bfs bfs 0 10000000 1600000000
    run_command_in_dir 2 rodinia/nw needle -mb 10240

    warmup rapids/cugraph_pagerank.py -n 10000 -m 1000000
    run_command_in_dir 2 rapids cugraph_pagerank.py -n 10000000 -m 500000000

    warmup rapids/cudf_stdev.py -rows 1000 -cols 10 -runs 2
    run_command_in_dir 5 rapids cudf_stdev.py -rows 500000000 -cols 6 -runs 10

    # For analysis
    run_command_in_dir 2 polybench/MVT mvt.exe -mb 9999 -per-kernel-array
}

profiler_analysis_v3 () {
    increasing_data_sizes="n"

    # MVT
    for data in 5500 8000 10001; do
        for hint in ""  \
                    "-thread-block 32"  \
                    "-optimal"          \
                    "-optimal-tblock"   \
                    "-prefloc-cpu 1"    \
                    "-prefloc-cpu 6"    \
                    "-prefloc-cpu 1,6"  \
                    "-accby-gpu 1"      \
                    "-readmostly 1"     \
                    "-readmostly 1,6"   \
                    ; do
            run_command_in_dir 1 polybench/MVT mvt.exe -mb $data    \
                    $hint -per-kernel-array
        done
    done

    # BFS
    for hint in \
        ""                              \
        "-accby-gpu 2,3,4,5,6,7,8"      \
        "-prefloc-gpu 2,3,4,5,6,7,8"    \
        "-accby-gpu 1"                  \
        "-thread-block 32"              \
        ; do
        run_command_in_dir 1 UVMBench/bfs bfs 0 10000000 1800000000
    done

    # SGEMM. These data sizes are chosen to fit all three, two, or one arrays.
    for data in 5800 8500 15000 17500; do
        for hint in \
            ""                  \
            "-accby-gpu 1"      \
            "-accby-gpu 1,3"    \
            "-accby-gpu all"    \
            "-prefloc-cpu 1,3"  \
            "-accby-gpu 2"      \
            "-accby-gpu 1,2"    \
            "-accby-gpu 2,3"    \
            ; do
            run_command_in_dir 1 nvidia-samples/sgemm sgemm.out -mb $data $hint
        done
    done

    # 2DCONV.
    for hint in \
        ""                  \
        "-readmostly 1"     \
        "-readmostly all"   \
        "-accby-gpu 1"      \
        "-accby-gpu 2"      \
        "-prefloc-cpu 2"    \
        "-accby-gpu all"    \
        "-optimal"          \
        ; do
        run_command_in_dir 2 polybench/2DCONV 2DConvolution.exe -mb 30000 \
                $hint -phases 3
    done

    # NW - mind the block-size-magic.
    for data in 5700 8000 10240; do
        for hint in ""  \
                "-accby-gpu 1"      \
                "-accby-gpu 2"      \
                "-accby-gpu 1,2"    \
                "-optimal"          \
                ; do
            for tblock in "" "-threads-64"; do
                run_command_in_dir 2 rodinia/nw needle -mb $data $hint $tblock
            done
        done
    done

    # INT_SET: note the thrashing, and the horrible behaviour of RM, even after
    # the first iteration.
    for data in 5650 5700 20000; do
        for hint in ""      \
            "-accby-gpu 1"      \
            "-accby-gpu 1 -cpu-init-first"  \
            "-accby-gpu 1 -compare"         \
            "-stride-2m"                    \
            "-readmostly 1"                 \
            ; do
            run_command_in_dir 2 synthetic_benchmarks \
                    int_set_uvm.out -mb $data $hint
        done
    done

    # ATAX
    for data in 5500 9999; do
        for hint in ""          \
            "-readmostly 1"     \
            "-prefloc-cpu 1"    \
            ; do
            for tblock in "" "-thread-block 32"; do
                run_command_in_dir 1 polybench/ATAX atax.exe -mb $data $hint $tblock
            done
        done
    done

    # STDEV 
    warmup rapids/cudf_stdev.py -rows 100000 -cols 3
    run_command_in_dir 2 rapids cudf_stdev.py -rows 500000000 -cols 6 -runs 10
    run_command_in_dir 2 rapids cudf_stdev.py -pool -rows 500000000 -cols 6 -runs 10

    warmup rapids/cugraph_pagerank.py -n 10000 -m 1000000
    run_command_in_dir 1 rapids cugraph_pagerank.py -n 10000000 -m 300000000
    run_command_in_dir 1 rapids cugraph_pagerank.py -n 10000000 -m 300000000 -pool
    run_command_in_dir 1 rapids cugraph_pagerank.py -n 10000000 -m 500000000
    run_command_in_dir 1 rapids cugraph_pagerank.py -n 10000000 -m 500000000 -pool

    # HT
    # SKIPLIST

}

profiler_analysis_v1 () {
    increasing_data_sizes="n"
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10000 -per-kernel-array
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 9999 -per-kernel-array
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -per-kernel-array
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -per-kernel-array -thread-block 16
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -per-kernel-array -thread-block 32
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 8888 -per-kernel-array
    run_command_in_dir 1 polybench/ATAX atax.exe 10000000000
    run_command_in_dir 1 polybench/ATAX atax.exe  8888000000
    run_command_in_dir 5 polybench/2DCONV 2DConvolution.exe -mb 30000 \
            -phases 3
    export TRACE_ROOT=/data/heterods-trace
    run_command_in_dir 1 hetero-ds-hashtable driver_hashtable_UVM.out -rns=1 \
        -ops=615000000 -add=100 -tra=insert_trace-400e7-100-add-no-dup-SPARSE_UNIQUE.bin
    run_command_in_dir 1 rodinia/nw needle -mb 10240
    run_command_in_dir 3 synthetic_benchmarks int_set_uvm.out -mb 20000 -iter 3 \
       -compare
    run_command_in_dir 1 UVMBench/bfs bfs 0 10000000 1800000000
    run_command_in_dir 1 nvidia-samples/sgemm sgemm.out -mb 17500
    run_command_in_dir 1 nvidia-samples/sgemm sgemm.out -mb 17500 \
        -acc-by-b
    run_command_in_dir 1 synthetic_benchmarks random_acc.out -mb 12000

    run_command_in_dir 2 synthetic_benchmarks int_set_uvm.out -data 30000000000 \
            -compare -iters 4

    run_command_in_dir 2 rapids cudf_stdev.py -rows 500000000 -cols 6 -runs 10

    # warmup rapids/cugraph_pagerank.py -n 10000 -m 1000000
    run_command_in_dir 1 rapids cugraph_pagerank.py -n 10000000 -m 500000000
    # run_command_in_dir 1 rapids cugraph_pagerank.py -n 10000000 -m 500000000 -pool

    warmup rapids/cugraph_sssp.py
    run_command_in_dir 1 rapids cugraph_sssp.py -n 10000000 -m 280000000

}

# Some custom optimization ONLY for the vanilla driver
profiler_analysis_v2 () {
    run_command_in_dir 1 nvidia-samples/sgemm sgemm.out -mb 17500 \
        -right-matrix-acc-by

    run_command_in_dir 1 rodinia/nw needle -mb 10240 -accessed-by-input
    run_command_in_dir 1 rodinia/nw needle -mb 10240 -accessed-by-ref
    run_command_in_dir 1 rodinia/nw needle64 -mb 10240 -optimal
    run_command_in_dir 10 polybench/MVT mvt.exe -mb 10001 -optimal
    run_command_in_dir 1 rodinia/nw needle -mb 10240 -optimal
}

driver_test1 () {
    increasing_data_sizes="n"
    run_command_in_dir 1 polybench/2DCONV 2DConvolution.exe -mb 30000 -phases 3
    run_command_in_dir 1 polybench/2DCONV 2DConvolution.exe -mb 30000 \
        -phases 2 -readmostly 1
    run_command_in_dir 1 polybench/2DCONV 2DConvolution.exe -mb 15000 \
        -readmostly all

    export TRACE_ROOT=/data/heterods-trace
    run_command_in_dir 1 hetero-ds-hashtable driver_hashtable_UVM.out -rns=1 \
        -ops=615000000 -add=100 -tra=insert_trace-400e7-100-add-no-dup-SPARSE_UNIQUE.bin
    run_command_in_dir 1 synthetic_benchmarks int_set_uvm.out -mb 20000 -iter 5 \
       -compare
    run_command_in_dir 1 nvidia-samples/sgemm sgemm.out -mb 17500 -acc-by-b
    run_command_in_dir 1 nvidia-samples/sgemm sgemm.out -mb 12500 -prefloc-cpu 3
    run_command_in_dir 1 nvidia-samples/sgemm sgemm.out -mb 12500 -prefloc-gpu 3
    run_command_in_dir 1 nvidia-samples/sgemm sgemm.out -mb 17500 -accby-gpu all

    run_command_in_dir 2 rapids cudf_stdev.py -rows 500000000 -cols 6 -runs 10
    run_command_in_dir 2 rapids cudf_stdev.py -rows 500000000 -cols 6 -runs 10 -pool

    warmup rapids/cugraph_pagerank.py -n 10000 -m 1000000
    run_command_in_dir 1 rapids cugraph_pagerank.py -n 10000000 -m 500000000
    run_command_in_dir 1 rapids cugraph_pagerank.py -n 10000000 -m 500000000 -pool
    run_command_in_dir 3 synthetic_benchmarks int_set_uvm.out -mb 20000 -iter 5 \
       -compare

    run_command_in_dir 8 UVMBench/bfs bfs 0 10000000 1800000000
    warmup rapids/cugraph_sssp.py
    run_command_in_dir 1 rapids cugraph_sssp.py -n 10000000 -m 280000000
    run_command_in_dir 1 rapids cugraph_sssp.py -n 10000000 -m 280000000 -pool
    run_command_in_dir 1 rodinia/nw needle64 -mb 10240 -optimal
    run_command_in_dir 1 rodinia/nw needle -mb 10240
    run_command_in_dir 1 rodinia/nw needle -mb 10240 -readmostly all
    run_command_in_dir 1 rodinia/nw needle -mb 10240 -prefloc-gpu 2

    run_command_in_dir 1 nvidia-samples/sgemm sgemm.out -mb 12500
    run_command_in_dir 1 synthetic_benchmarks random_acc.out -mb 12000
}

isca_workloads_single_run () {
    increasing_data_sizes="n"
    warmup_test polybench/2DCONV/2DConvolution.exe -mb 100
    run_command_in_dir 2 polybench/2DCONV 2DConvolution.exe -mb 30000
    run_command_in_dir 2 polybench/2DCONV 2DConvolution.exe -mb 30000 -phases 5

    run_command_in_dir 2 synthetic_benchmarks int_set_uvm.out -data 30000000000 \
            -compare -iter 4

    # warmup
    warmup "nvidia-samples/sgemm/sgemm.out" -data 1000000000 
    run_command_in_dir 1 nvidia-samples/sgemm sgemm.out \
         -mb 10000 -copy-back
    run_command_in_dir 1 nvidia-samples/sgemm sgemm.out \
         -mb 17500 -copy-back

    export TRACE_ROOT=/data/heterods-trace
    warmup data_structures/hetero-ds-hashtable/driver_hashtable_UVM.out -rns=1 \
        -ops=999900 -add=100 -tra=insert_trace-400e7-100-add-no-dup-SPARSE_UNIQUE.bin
    run_command_in_dir 1 data_structures/hetero-ds-hashtable driver_hashtable_UVM.out -rns=1 \
        -ops=615000000 -add=100 -tra=insert_trace-400e7-100-add-no-dup-SPARSE_UNIQUE.bin

    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -copy-back
    run_command_in_dir 1 polybench/ATAX atax.exe 10010000000
    run_command_in_dir 1 UVMBench/bfs bfs 0 10000000 1800000000

    run_command_in_dir 1 rodinia/nw needle -mb 10240

    warmup rapids/cugraph_pagerank.py -n 10000 -m 1000000
    run_command_in_dir 1 rapids cugraph_pagerank.py -n 10000000 -m 500000000

    warmup rapids/cudf_stdev.py -rows 1000 -cols 10 -runs 2
    run_command_in_dir 1 rapids cudf_stdev.py -rows 500000000 -cols 6 -runs 10
    run_command_in_dir 1 rapids cugraph_sssp.py -n 10000000 -m 280000000
}

# For the table in discussion/thrashing
thrashing_statistics () {
    assert_is_root
    load_driver_npf jerry
    # the npf, fpf flags don't do anything. Just for bookkeeping
    run_command_in_dir 1 polybench/MVT mvt.exe 10"000000000" -copy-back -nopf
    run_command_in_dir 1 polybench/2DCONV 2DConvolution.exe -mb 30000 -nopf \
            -copy-back
    run_command_in_dir 1 UVMBench/bfs bfs 0 10000000 1800000000 -nopf


    # default and pin-cpu runs
    load_driver jerry
    run_command_in_dir 1 polybench/MVT mvt.exe 9"000000000" -copy-back
    run_command_in_dir 1 polybench/MVT mvt.exe 10"000000000" -copy-back
    run_command_in_dir 1 polybench/MVT mvt.exe 10"000000000" -pin-input-cpu -copy-back

    run_command_in_dir 1 polybench/2DCONV 2DConvolution.exe -mb 30000 \
            -copy-back
    run_command_in_dir 1 polybench/2DCONV 2DConvolution.exe -mb 30000 \
            -pin-cpu -copy-back

    run_command_in_dir 1 UVMBench/bfs bfs 0 10000000 1800000000
    run_command_in_dir 1 UVMBench/bfs bfs 0 10000000 1800000000 -pin-adj-cpu

    # aggresive prefetching, but not full. :(
    load_driver_fpf jerry
    run_command_in_dir 1 polybench/2DCONV 2DConvolution.exe -mb 30000 \
            -fullpf -copy-back
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10000 -copy-back -fullpf

    unload_driver
}

try_hints_per_buffer () {
    for hint in     \
            "-accby-gpu all"    \
            "-prefloc-cpu all"  \
            "-accby-gpu all -prefetch all"  \
            "-accby-gpu all -prefetch 2"    \
            "-accby-gpu 1 -prefloc-cpu 1"   \
            "-accby-gpu 1 -prefloc-gpu 2 -prefloc-cpu 1"    \
            "-prefloc-cpu all -prefetch 2 -accby-gpu all"   \
            "-prefloc-cpu all -prefetch all -accby-gpu 1"   \
            "-prefloc-cpu all -prefetch all -accby-gpu all" \
            "-accby-gpu all -prefetch 2 -prefloc-cpu 1"     \
            "-accby-gpu 1 -prefloc-cpu 1 -prefloc-gpu 2"    \
            "-prefloc-cpu all -accby-gpu all"               \
            "-accby-gpu 1 -prefloc-gpu 1 -prefloc-gpu 2"    \
            "-accby-gpu 1 -prefloc-gpu 1 -prefetch 2"       \
            "-accby-gpu 1 -prefloc-cpu 1 -prefetch 2"       \
            "-prefloc-cpu 1 -prefetch all"                  \
            ""  \
            "-prefetch all"     \
            "-readmostly all" "-accby-gpu 1" "-prefetch 1" "-accby-gpu 2"; do

        for mb in 6000 7500 9000 12000 30000; do
            run_command_in_dir 3 polybench/2DCONV 2DConvolution.exe -mb $mb $hint
        done
        for mb in 6000 7500 9000 12000 18000; do
            run_command_in_dir 2 rodinia/nw needle -mb $mb $hint
        done
    done
}

try_hints_v2 () {
    load_driver_vanilla

    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -readmostly 1
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -thread-block 32
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -thread-block 32   \
        -readmostly 1

    # accby magic
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -thread-block 32 -kernel2-accby
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -thread-block 32 -accby-gpu 1

    # the combo
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -thread-block 32   \
            -accby-gpu all -prefetch all
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -thread-block 32   \
            -kernel2-accby -prefetch all
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001    \
            -accby-gpu all -prefetch all
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001    \
            -kernel2-accby -prefetch all

    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -optimal-tblock
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -optimal

    load_driver_vanilla uvm_perf_prefetch_enable=0
    driver_filename_prefix="_van_npf"
    driver_description="vanilla_npf"
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -thread-block 32
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -thread-block 1

    # for some reason, this driver configuration is slower.
    load_driver_vanilla uvm_perf_prefetch_threshold=100
    driver_filename_prefix="_van_npf100"
    driver_description="vanilla_npf100"
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -thread-block 32
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -thread-block 1

    load_driver_vanilla uvm_perf_prefetch_threshold=1
    driver_filename_prefix="_van_fpf"
    driver_description="vanilla_fpf"
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -thread-block 32

    # accby magic
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -thread-block 32 -kernel2-accby
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -thread-block 32 -accby-gpu 1

    # the combo
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -thread-block 32   \
            -accby-gpu all -prefetch all
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -thread-block 32   \
            -kernel2-accby -prefetch all

    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -optimal-tblock
    run_command_in_dir 1 polybench/MVT mvt.exe -mb 10001 -optimal

    load_driver_vanilla

    try_hints_per_buffer
}

try_hints_v3 () {

    for hint in ""  "-pfcpu 1 -accby-gpu 1 -prefetch 2 -output-pf 2"    \
        "-pfcpu 1 -accby-gpu 1 -initgpu 2 -output-pf 2"             \
        "-pfcpu 1 -accby-gpu all -pfcpu all"                        \
        "-pfcpu all -accby-gpu 1 -prefetch 2 -output-pf 2"          \
        "-pfcpu all -accby-gpu 1 -prefloc-gpu 2 -output-pf 2"       \
        "-pfcpu all -accby-gpu all"                                 \
        "-pfcpu all -accby-gpu 2 -rm-late 1"                        \
        ; do
        run_command_in_dir 5 polybench/2DCONV 2DConvolution.exe -mb 6000 $hint
        run_command_in_dir 5 polybench/2DCONV 2DConvolution.exe -mb 7500 $hint
        run_command_in_dir 5 polybench/2DCONV 2DConvolution.exe -mb 9000 $hint
    done

    for hint in \
        ""      \
        "-pfcpu 1 -initgpu 2 -accby-gpu 1 -output-pf 2"     \
        "-pfcpu 1 -prefetch 2 -accby-gpu 1 -output-pf 2"    \
        "-pfcpu 1,2 -accby-gpu 1,2"                         \
        "-pfcpu 1,2 -accby-gpu 2  -prefetch 1"              \
        "-pfcpu 1,2 -accby-gpu 2 -rm-late 1 -prefetch 1"    \
        "-pfcpu 1,2 -accby-gpu 2 -rm-late 1"                \
        "-pfcpu 1 -prefetch 2 -accby-gpu 1 -output-pf 2"    \
        ; do
        run_command_in_dir 5 polybench/2DCONV 2DConvolution.exe -mb 12000 $hint
    done

    for hint in \
        ""      \
        "-pfcpu 1 -initgpu 2 -output-pf 2 -accby-gpu 1"     \
        "-pfcpu 1 -initgpu 2              -accby-gpu 1"     \
        "-pfcpu 1 -initgpu 2 -output-pf 2"                  \
        "-pfcpu 1 -initgpu 2 -output-pf 2 -abg-late 1"      \
        "-pfcpu 1 -initgpu 2 -output-pf 2 -abg-late 1"      \
        "-pfcpu 1 -initgpu 2 -rm-late 1"                    \
        "-pfcpu 1 -initgpu 2 -rm-late 1"                    \
        "-pfcpu 1 -prefloc-cpu 2 -accby-gpu all"            \
        "-pfcpu 1,2  -accby-gpu all"                        \
        "-pfcpu 1,2 -accby-gpu 2 -rm-late 1"                \
        "-pfcpu 1 -accby-gpu 1,2 -prefetch 2 -output-pf 2"  \
        ; do
        run_command_in_dir 5 polybench/2DCONV 2DConvolution.exe -mb 30000 $hint
    done

    for hint in ""  "-pfcpu all -prefetch all"                      \
        "-pfcpu all -prefetch all -accby-gpu all"                   \
        "-pfcpu all -prefetch 1 -accby-gpu all"                     \
        "-pfcpu all -prefetch 2 -accby-gpu all"                     \
        "-pfcpu all -prefloc-cpu all -prefetch all -accby-gpu all"  \
        "-pfcpu all -prefetch 1 -accby-gpu 2"        \
        "-pfcpu all -prefetch 1 -accby-gpu all"      \
        "-pfcpu all -prefetch 2 -accby-gpu 1"        \
        "-pfcpu all -prefetch 2 -accby-gpu all"      \
        "-pfcpu all -prefetch 2 -accby-gpu all"                     \
        "-pfcpu all -prefetch all -accby-gpu all"                   \
        "-pfcpu all -prefetch all -accby-gpu all -prefloc-cpu all"  \
        "-pfcpu all -rm-late 1 -accby-gpu all"          \
        "-pfcpu all -prefloc-cpu all -prefetch 1"       \
        "-pfcpu all -prefloc-cpu all -prefetch 2"       \
        ; do
        run_command_in_dir 5 rodinia/nw needle -mb 6000 $hint
        run_command_in_dir 5 rodinia/nw needle -mb 7500 $hint
        run_command_in_dir 5 rodinia/nw needle -mb 9000 $hint
        run_command_in_dir 3 rodinia/nw needle -mb 11000 $hint
        run_command_in_dir 3 rodinia/nw needle -mb 12000 $hint
        run_command_in_dir 2 rodinia/nw needle -mb 18000 $hint
    done
}

try_hints_sgemm () {
    for hint in ""  \
        "-accby-gpu all"    \
        "-prefloc-cpu all"  \
        "-prefetch all"     \
        "-accby-gpu all -prefetch all"  \
        "-prefloc-cpu all -accby-gpu all"   \
        "-prefloc-cpu all -accby-gpu all -prefetch all" \
        "-readmostly all"   \
        "-initgpu 3"        \
        "-initgpu 3 -pfcpu 1,2"     \
        "-initgpu 3 -pfcpu 1,2 -accby-gpu 3"    \
        "-initgpu 3 -pfcpu 1,2 -accby-gpu 1,3"  \
        "-initgpu 3 -pfcpu 1,2 -accby-gpu 2,3"  \
        "-initgpu 3 -pfcpu 1,2 -accby-gpu 1,2"  \
        "-initgpu 3 -pfcpu 1,2 -accby-gpu all"  \
        "-prefetch 3 -pfcpu 1,2 -accby-gpu all -output-pf 3"    \
        "-prefetch 3 -pfcpu 1,2 -accby-gpu 1 -output-pf 3"      \
        "-prefetch 3 -pfcpu 1,2 -accby-gpu all -output-pf 3"    \
        "-prefetch 3 -pfcpu 1,2 -accby-gpu all -output-pf 3"    \
        ; do

        run_command_in_dir 2 nvidia-samples/sgemm sgemm.out -mb 5500 $hint
        run_command_in_dir 2 nvidia-samples/sgemm sgemm.out -mb 9000 $hint
        run_command_in_dir 2 nvidia-samples/sgemm sgemm.out -mb 15000 $hint
    done
}

try_hints_sgemm_18g () {
    for hint in ""  \
        "-accby-gpu 1,3 -prefetch 2"                    \
        "-accby-gpu 1,3 -prefetch 2"                    \
        "-accby-gpu 1,2,3 -prefetch 2"                  \
        "-accby-gpu 1,2,3 -prefetch 1 -rm-late 1,2"     \
        "-accby-gpu 1,2,3 -prefetch 2 -rm-late 1,3"     \
        "-accby-gpu 1,2,3 -prefetch 3 -rm-late 1,3"     \
        "-accby-gpu 1,2,3 -prefetch 2,3 -rm-late 1,3"     \
        "-accby-gpu 1,2,3 -prefetch 1,3 -rm-late 1,3"     \
        "-initgpu 3 -accby-gpu 1,2,3 -prefetch 2"       \
        "-initgpu 3 -prefetch 2"                        \
        "-initgpu 3 -accby-gpu 3 -prefetch 2"           \
        "-accby-gpu all"    \
        "-prefloc-cpu all"  \
        "-prefetch all"     \
        "-accby-gpu all -prefetch all"  \
        "-prefloc-cpu all -accby-gpu all"   \
        "-prefloc-cpu all -accby-gpu all -prefetch all" \
        "-readmostly all"   \
        ; do
        run_command_in_dir 1 nvidia-samples/sgemm sgemm.out -mb 18000 $hint
    done
}
