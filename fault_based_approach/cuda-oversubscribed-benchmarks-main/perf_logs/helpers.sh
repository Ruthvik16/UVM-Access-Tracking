# Helpers for do_perf_run.sh and tonight.sh

unload_driver () {
    echo rmmod nvidia_drm nvidia_peermem nvidia_modeset nvidia_uvm nvidia
    rmmod nvidia_drm nvidia_peermem nvidia_modeset nvidia_uvm nvidia

    check_drivers=`lsmod | grep -c ^nvidia`
    if [ "$check_drivers" -ne "0" ]; then
        echo "Strange error - trying after 5 minutes"
        nvidia-smi > /dev/stderr
        sleep 900 # not sure what happens sometimes
        echo rmmod nvidia_drm nvidia_peermem nvidia_modeset nvidia_uvm nvidia
        rmmod nvidia_drm nvidia_peermem nvidia_modeset nvidia_uvm nvidia
    fi

    if [ "$check_drivers" -ne "0" ]; then
        echo "COULD NOT UNLOAD DRIVER"
        lsmod | grep nvidia
        exit 3
    fi
    echo "modules loaded now:"
    lsmod | grep nvi
    driver_filename_prefix=""
}

# takes an argument
load_driver () {
    unload_driver
    cur_dir=$PWD
    cd /data/pranjal/open-gpu-kernel-modules/$1/
    echo insmod nvidia.ko nvidia-uvm.ko
    insmod nvidia.ko
    insmod nvidia-uvm.ko
    echo "modules loaded now:"
    lsmod | grep nvi
    check_drivers=`lsmod | grep -c ^nvidia`
    if [ "$check_drivers" -ne "2" ]; then
        echo "COULD NOT LOAD DRIVER"
        lsmod  | grep nvidia
        exit 3
    fi
    cat /proc/bkbuvm/debug | tail > /dev/stderr
    driver_filename_prefix="_tbn"
    driver_description="$1""$driver_filename_prefix"
    cd -
}

load_driver_vanilla () {
    echo "&&&&&&&&&&&&&&& load_driver VANILLA &&&&&&&&&&&&&&"
    unload_driver
    cd /data/pranjal/artifacts/open-gpu-kernel-modules/
    chk_version=`grep 575.51.03 version.mk`
    if [ -n "$chk_version" ]; then
        make -j > /dev/null
        cd kernel-open
        echo "In PWD $PWD"
    else
        echo NOOOO WRONG version!
        exit 1
    fi

    echo insmod nvidia.ko nvidia-uvm.ko
    insmod nvidia.ko
    insmod nvidia-uvm.ko $@
    echo "modules loaded now:"
    lsmod | grep nvi

    check_two_drivers

    driver_description=vanilla
    driver_filename_prefix="_van"
    cd $logs_dir
}

check_two_drivers () {
    check_drivers=`lsmod | grep -c ^nvidia`
    if [ "$check_drivers" -ne "2" ]; then
        echo "COULD NOT LOAD DRIVER"
        lsmod  | grep nvidia
        exit 3
    fi
}

# loads profiler1
load_driver_npf() {
    unload_driver

    echo "&&&&&&&&&&&&&&& load_driver_npf &&&&&&&&&&&&&&"
    cd /data/pranjal/open-gpu-kernel-modules/$1/
    echo insmod nvidia.ko
    echo insmod nvidia-uvm.ko uvm_perf_prefetch_enable=0
    echo "In wd $PWD"
    insmod nvidia.ko
    insmod nvidia-uvm.ko uvm_perf_prefetch_enable=0
    check_two_drivers
    driver_filename_prefix="_npf"
    driver_description="$1""$driver_filename_prefix"
    cd -
}

load_driver_pf_threshold () {
    unload_driver

    cd /data/pranjal/open-gpu-kernel-modules/$1/
    echo "&&&&&&&&&&&&&&& load_driver_fpf &&&&&&&&&&&&&&"
    echo "In wd $PWD"
    echo insmod nvidia.ko
    echo insmod nvidia-uvm.ko uvm_perf_prefetch_threshold=1
    insmod nvidia.ko
    insmod nvidia-uvm.ko uvm_perf_prefetch_threshold=$2
    check_two_drivers
    driver_filename_prefix="_pf$2"
    driver_description="$1""$driver_filename_prefix"
    cd -
}

load_driver_fpf () {
    load_driver_pf_threshold $1 1
    driver_filename_prefix="_fpf"
    driver_description="$1""$driver_filename_prefix"
}

warmup () {
    warmup_binary="$repo_root/$1"
    shift
    echo "** WARMUP $warmup_binary **"
    $warmup_binary $@
    if [ "$?" -ne "0" ]; then
        echo "ERR: non-zero exit $?"
        st_cmd="strace -o error.strace $warmup_binary $@"
        echo $st_cmd
        $st_cmd
        echo ""
        echo "Do you want to: rm -rf *.$$"
        wc error.strace
        exit
    fi
    echo "** WARMUP COMPLETE **"
}

warmup_test () {
    warmup $@
}

assert_is_root () {
    if [[ "$USER" != "root" ]]; then
        echo "Running as $USER, not root!"
        exit 2
    fi
}

# It is difficult to pause and clean up the script when run as root.
check_exit () {
    exit_filename="/data/pranjal/cuda-benchmarks/perf_logs/do_perf_run_exit"
    if [ -f $exit_filename ]; then
        echo "Exiting now"
        echo "Driver is $driver_description"
        exit
    fi
    fs_usage=$(df --output=pcent /)
    is_full=`echo $fs_usage | grep -c 100`
    if [[ $is_full == "1" ]]; then
        echo "FS full! Exiting\n"
        echo \$ df /
        df /
        exit 2
    fi
    if [[ $USER == "root" ]]; then
        echo -n "DMESG IS " > /dev/stderr
        sudo dmesg | wc > /dev/stderr
        echo ============== > /dev/stderr
        nlin=$(dmesg | grep -E GRAPHICS\|kernel-open)
        if [ -n "$nlin" ]; then
            echo check dmesg
            exit 2
        fi
    fi
}

check_pause () {
    check_exit
    pause_filename="/data/pranjal/cuda-benchmarks/perf_logs/do_perf_run_pause"
    while true; do
        if [ -f "$pause_filename" ]; then
            echo "Paused with driver $driver_filename_prefix"
            echo "Paused with command $@"
            sleep 60
        else
            break
        fi
    done
}

check_exit
