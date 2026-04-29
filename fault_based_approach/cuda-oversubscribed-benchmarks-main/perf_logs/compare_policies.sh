#!/bin/bash

# Runs do_perf_run.sh for multiple driver configurations and/or profilers.

meta_log="check_pf.$fn2_to_run.$$"


repo_root=/data/pranjal/cuda-benchmarks
logs_dir=$repo_root/perf_logs
meta_log="$logs_dir/$meta_log"
driver_description=""
driver_filename_prefix=""
timeout=200
subfn_to_run="sssp_around_300m"

fn2_to_run="all_driver_configs"
meta_log="check_pf.$fn2_to_run.$$"
child_pids=" "

profiler="kprof" # might be non-jerry or non-spongebob for some run

source helpers.sh
source workload_lists.sh
source isca_eval_workloads.sh

while getopts ":t:c:d:" o
do
    case "${o}" in
        t)
            echo "Timeout is ${OPTARG}"
            timeout=${OPTARG}
            ;;
        c)
            subfn_to_run=${OPTARG}
            ;;
        d)
            echo "Testing driver configurations: ${OPTARG}"
            fn2_to_run=${OPTARG}
            ;;
        *)
            echo bad_usage
            exit 5
            ;;
    esac
done

prologue () {
    assert_is_root
    date | tee $meta_log
    echo "Saving meta-output layer 1 to $meta_log"
    type $function_to_run | tee -a $meta_log
    echo "Running sub-function: $subfn_to_run"
}

epilogue () {
    grep -h -E elapsed\|IST perf_run*$$* >> $meta_log
    rm -rf perf_run_all.*/*$$*
    echo "TO DELETE:" >> $meta_log
    echo perf_run_all.*/*$$* >> $meta_log
    echo "Child pids: $child_pids" >> $meta_log

    output_dir=cmp_policies.$subfn_to_run.$$/
    echo "Will save output to $output_dir"
    mkdir $output_dir
    cp *$$* $output_dir/
    rm *$$*

    chown -R pranjal $output_dir

    load_driver $profiler
}

get_overhead () {
    load_driver_vanilla
    ./do_perf_run.sh -t $timeout -d $driver_description"$$" -c $subfn_to_run -l

    load_driver $profiler
    ./do_perf_run.sh -t $timeout -d $driver_description"$$" -c $subfn_to_run
}

get_overhead_v2 () {
    get_overhead

    load_driver suneo
    ./do_perf_run.sh -t $timeout -d $driver_description"$$" -c $subfn_to_run -l

    load_driver nojerry
    ./do_perf_run.sh -t $timeout -d $driver_description"$$" -c $subfn_to_run -l
}

all_driver_configs () {
    # Warning: npf is NOT the same as threshold=100. The latter still prefetches
    # the 64 kB neighbourhood.
    get_overhead

    load_driver_npf $profiler
    ./do_perf_run.sh -t $timeout -d $driver_description"$$" -c $subfn_to_run

    load_driver_fpf $profiler"_fpf"
    ./do_perf_run.sh -t $timeout -d $driver_description"$$" -c $subfn_to_run

    load_driver $profiler"_remote"
    ./do_perf_run.sh -t $timeout -d $driver_description"$$" -c $subfn_to_run

    load_driver_vanilla
    ./do_perf_run.sh -t $timeout -d $driver_description"$$" -c profiler_analysis_v2
}


check_pf_threshold () {
    all_driver_configs

    load_driver_pf_threshold $profiler 25
    ./do_perf_run.sh -t $timeout -d $driver_description"$$" -c $subfn_to_run

    load_driver_pf_threshold $profiler 75
    ./do_perf_run.sh -t $timeout -d $driver_description"$$" -c $subfn_to_run
}

prologue
$fn2_to_run
epilogue
