#!/bin/bash

# rd_dup or vanilla or ?
driver_description=rd_dup
time_limit=2000
function_to_run="run_benchmarks"

# if the last benchmark timed out, do NOT execute.
# and reset this to "y" or "n" in each new benchmark
did_last_execution_timeout="n"

# Do we want to skip suneo's logs?
save_profiler_logs="y"

# using an nvidia profiler? time? timeout?
profiler_cli_prefix=""

# increasing data sizes? Abort on timeout?
increasing_data_sizes="y"

# Is it a fpf or npf driver? Add a flag to the filename
# Not important, feel free to ignore
driver_filename_prefix=""

# Options: [ -d <vanilla|rd_dup|..>]
#          [ -t timeout ] default 2000 s
#          [ -c command ] # name of the function to run
#          [ -l] Do not save profiler logs - For vanilla runs
#          [ -p <"ncu ..."|"nsys ..."|profiler_prefix> ] when run with nsys/ncu
while getopts ":d:t:c:p:l" o
do
    case "${o}" in
        d)
            echo Driver description ${OPTARG}
            driver_description=${OPTARG}
            ;;
        t)
            echo timeout is ${OPTARG}
            time_limit=${OPTARG}
            ;;
        c)
            function_to_run=${OPTARG}
            echo "Received a function name ${OPTARG}"
            ;;
        l)
            echo "Will not save profiler logs"
            save_profiler_logs="n"
            ;;
        p)
            echo "Using Profiler command prefix: \"${OPTARG}\""
            profiler_cli_prefix=${OPTARG}
            ;;
        *)
            echo bad usage
            exit 4
            ;;
    esac
done

echo Time limit per benchmark is $time_limit
echo Driver description $driver_description

if [ -z "$time_limit" ]
then
    echo bad config
fi

repo_root=/data/pranjal/cuda-benchmarks
logs_dir=$repo_root/perf_logs
complete_log=$logs_dir/perf_run.$driver_description.$function_to_run.out.$$
echo "Saving output to $complete_log"

# sudo runs into some error
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:$LD_LIBRARY_PATH


date >> $complete_log
# the last four lines of the log have the driver config
cat /proc/bkbuvm/debug | tail -n 15 >> $complete_log
echo >> $complete_log
echo Using memory hints $MEM_HINTS | tee -a $complete_log
echo "*** *** *** ***"

source helpers.sh

clear_stats () {
    if [ ! -f "/proc/bkbuvm/debug" ]; then
        return
    fi
    echo clear  > /proc/bkbuvm/debug
}

save_stats () {
    if [[ "$save_profiler_logs" == "n" ]]
    then
        echo "Not saving profiler output"
        return
    fi
    if [ ! -f "/proc/bkbuvm/debug" ]; then
        return
    fi

    echo "** suneo stats begin **" >> $1
    cat /proc/bkbuvm/debug >> $1
    echo "** suneo stats end **" >> $1
    cat /proc/bkbuvm/squidward | head -n 1000 >> $1
    cat /proc/bkbuvm/mickey    | head -n 1000 >> $1
    echo '** all stats end **' >> $1
}

# this function actually runs the damn command
# $1 is number of runs per benchmark
# $2 is the working dir for the benchmark
# $3 onwards is benchmark and args
run_command_in_dir () {
    num_iterations=$1
    shift

    cd $repo_root
    cd $1
    shift

    sleep 2
    clear_stats
    check_pause $@

    # For now, we are not running applications with increasing data sizes
    if [[ "$increasing_data_sizes" == "n" ]]; then
        did_last_execution_timeout="n"
    fi

    if [[ "$did_last_execution_timeout" == "y" ]]
    then
        echo "TIMEOUT EARLIER | SKIP | $@" | tee -a $complete_log $about_run
        return 1
    fi

    echo
    echo "*** *** ***"
    actual_command="$@ $MEM_HINTS"
    echo Beginning $actual_command $driver_filename_prefix | tee -a $complete_log
    echo "Profiler prefix: \"$profiler_cli_prefix\""

    save_output_to=suneo_"$driver_description"_${actual_command//\ /\_}_"$driver_filename_prefix"_iter$num_iterations.out.$$
    save_output_to=${save_output_to//\//_}
    save_output_to=${save_output_to//=/_}
    save_output_to=$logs_dir/$save_output_to
    echo Saving suneo output to $save_output_to | tee -a $complete_log
    echo "# Runs: $num_iterations" | tee -a $complete_log $save_output_to

    echo "$@ $driver_filename_prefix" >> $save_output_to

    # this fails because the exit status printed $? is of tee, not timeout
    # timeout $time_limit time -ao $save_output_to -f "%C: %e" $@ | tee -a $save_output_to

    # repackage the main command, pipe and tee into a subcommand. sh is lighter than bash
    actual_command=" $profiler_cli_prefix ./$actual_command"
    rm -f temp_perf.out
    command_string="perf stat -r $num_iterations -o temp_perf.out $actual_command | tee -a $save_output_to "
    date
    timeout $(($time_limit*$num_iterations)) bash -c "$command_string"

    if [ $? -ne 0 ]
    then
        echo -ne "[ $num_iterations RUNS, $driver_description ]\t"  \
                | tee -a $complete_log $save_output_to $about_run
        echo "$actual_command $driver_filename_prefix: Timed out or killed [$time_limit] <elapsed>" | tee -a  \
                $complete_log $save_output_to $about_run
        echo ""
        did_last_execution_timeout="y"
        if [[ "$save_profiler_logs" == "y" ]]; then
            save_stats $save_output_to
        else
            rm $save_output_to
        fi

        return -1
    fi
    date

    echo -ne "[ $num_iterations RUNS, $driver_description ]\t$actual_command " >> $complete_log
    grep elapsed temp_perf.out | tee -a $save_output_to $complete_log $about_run

    echo >> $complete_log

    echo saving stats to $save_output_to
    echo "*** *** ***"
    save_stats $save_output_to
    chown pranjal $save_output_to
    if [[ "$save_profiler_logs" == "n" ]]
    then
        rm $save_output_to
    fi
}

run_command () {
    run_command_in_dir 1 $repo_root $@
}

run_command_with_count () {
    count=$1
    shift
    run_command_in_dir $count $repo_root $@
}

polybench_data_sizes_GB="4 5 6 7 8 9 10"
polybench_list_full="
polybench/2DCONV/2DConvolution.exe
polybench/2MM/2mm.exe
polybench/3DCONV/3DConvolution.exe
polybench/3MM/3mm.exe
polybench/ATAX/atax.exe
polybench/BICG/bicg.exe
polybench/CORR/correlation.exe
polybench/COVAR/covariance.exe
polybench/FDTD-2D/fdtd2d.exe
polybench/GEMM/gemm.exe
polybench/GESUMMV/gesummv.exe
polybench/GRAMSCHM/gramschmidt.exe
polybench/MVT/mvt.exe
polybench/SYR2K/syr2k.exe
polybench/SYRK/syrk.exe
"

# covar takes very long
# bicg times out at 11 GB
# syr2k times out anyways with rd
# GEMM times out at 4G
# GRAMSCHM times out at 9 GRAMSCHM times out at 9G
# SYR2K times out at 4 GB
polybench_list="
polybench/2DCONV/2DConvolution.exe
polybench/2MM/2mm.exe
polybench/ATAX/atax.exe
polybench/BICG/bicg.exe
polybench/FDTD-2D/fdtd2d.exe
polybench/MVT/mvt.exe
"

run_cugraph_pagerank () {
    cugraph_n_vertices=2000000 # 2 million 
    did_last_execution_timeout="n"

    for n_edges in 30 50 100 130 200
    do
        run_command_in_dir 1 rapids cugraph_pagerank.py -n $cugraph_n_vertices -m $n_edges"000000"
    done
    echo Finished pagerank >> $complete_log
}

# BFS: oversubscription starts at n=10M, m=1500M
run_bfs () {
    did_last_execution_timeout="n"
    for n_edges in 1400 1500 1600 1700 1800
    do
        for n_vertices_bfs in 10000000 100000000
        do
            run_command UVMBench/bfs/main 0 $n_vertices_bfs $n_edges"000000"
            run_command UVMBench/bfs/main 0 $n_vertices_bfs $n_edges"000000" -pin-adj-cpu
        done
    done
    echo Finished BFS >> $complete_log
}

fast_perf_run () {
    did_last_execution_timeout="n"
    run_command_in_dir 4 polybench/2DCONV 2DConvolution.exe 9000000000 -copy-back
    run_command_in_dir 4 polybench/2DCONV 2DConvolution.exe 12000000000 -copy-back
    run_command_in_dir 4 polybench/2DCONV 2DConvolution.exe 15000000000 -copy-back

    did_last_execution_timeout="n"
    run_command_in_dir 2 polybench/ATAX atax.exe  8000000000 -copy-back
    run_command_in_dir 2 polybench/ATAX atax.exe 10000000000 -copy-back

    did_last_execution_timeout="n"
    run_command_in_dir 2 polybench/2MM  2mm.exe  10000000000 -copy-back

    did_last_execution_timeout="n"
    run_command_with_count 2 UVMBench/bfs/main 0 10000000 1448"000000"

    did_last_execution_timeout="n"
    run_command hetero-ds-hashtable/driver_hashtable_UVM.out -ops=300"000000" \
        -rns=1 -add=60 -rem=20

    did_last_execution_timeout="n"
    run_command_in_dir 5 rapids cudf_stdev.py -rows 300000000 -cols 6
}

sanity_check () {
    run_command_in_dir 1 polybench/2DCONV 2DConvolution.exe 120000000 -copy-back -compare
    run_command        UVMBench/bfs/main 0 100000 6"000000"
}

# to avoid frequent changes to do_perf_run.sh, I'll write my functions here.
source workload_lists.sh
source isca_eval_workloads.sh
source asplos_workloads.sh


about_run="$logs_dir/about_run.$driver_description.$function_to_run.$$"
about_run="/dev/null"
type $function_to_run | tee -a $complete_log $about_run
$function_to_run


echo "%%% exec complete %%%" >> $complete_log
date >> $complete_log

chown pranjal $complete_log

cd $logs_dir
grep elapsed $complete_log > ovh_summary.$function_to_run.$$
chown pranjal ovh_summary.$function_to_run.$$ $about_run


output_dir="perf_run_all.$$"
mkdir $output_dir 2>/dev/null
if [ -d "$output_dir" ]; then
    cp *.$$ $output_dir
    chown -R pranjal $output_dir 
    echo "MOVED output to $output_dir"
    echo 'rm *.'"$$"
    rm *.$$
fi

if [[ "$save_profiler_logs" == "n" ]]; then
    rm -r $output_dir
    rm -r "suneo*$$"
fi
