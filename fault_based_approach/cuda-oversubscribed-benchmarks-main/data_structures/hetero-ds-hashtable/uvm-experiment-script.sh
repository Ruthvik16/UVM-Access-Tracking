#!/bin/bash
uvm_log_dir='execution-log/uvm-hashtable-logs'
if [ -d $uvm_log_dir ]; then
    echo $uvm_log_dir
    echo "Log directory for uvm experiments exists"
else
    echo "Creating directory for UVM experiments" 
    mkdir ${uvm_log_dir} 
fi

runs=5
# echo $5 
if [ $1 ]; then
    runs=$1
else
    echo "Default 3 runs"
    echo ${runs}
fi

# TRACE STEP IS 500000000
# htSize=(100000000 500000000 1000000000 1250000000 1500000000 2000000000 2500000000 2750000000 3000000000 3500000000 4000000000)
htSize=(500000000 1000000000 1500000000 2000000000 2500000000 3000000000 3500000000 4000000000)
# insert_trace-400e7-100-add-10-dup-DENSE_REPEAT.bin
# insert_trace-400e7-100-add-10-dup-SPARSE_REPEAT.bin
# insert_trace-400e7-100-add-no-dup-SPARSE_UNIQUE.bin
# insert_trace-400e7-100-add-no-dup-DENSE_UNIQUE.bin
# insert_trace-400e7-100-add-no-dup-MONOTONIC_INCREASE.bin

: <<comment
for size in "${htSize[@]}"
do
    ./gpu/hashtable/driver_hashtable_UVM-batch-sa-prefetch-double.out -ops=${size} -add=100 -rem=0 -tra="insert_trace-400e7-100-add-no-dup-DENSE_UNIQUE.bin" -trr="" -trf=""  -rns=${runs} -fil=1 > ${uvm_log_dir}/uvm-hashtable-${size}-double-100-add-no-dup-DENSE_UNIQUE-key-check-runs-$runs.log
    ./gpu/hashtable/driver_hashtable_UVM-batch-sa-prefetch-double.out -ops=${size} -add=100 -rem=0 -tra="insert_trace-400e7-100-add-no-dup-SPARSE_UNIQUE.bin" -trr="" -trf=""  -rns=3 -fil=1 > ${uvm_log_dir}/uvm-hashtable-${size}-double-100-add-no-dup-SPARSE_UNIQUE-new-trace.log
    ./gpu/hashtable/driver_hashtable_UVM-batch-sa-prefetch-double.out -ops=${size} -add=100 -rem=0 -tra="insert_trace-400e7-100-add-no-dup-MONOTONIC_DECREASE.bin" -trr="" -trf=""  -rns=3 -fil=1 > ${uvm_log_dir}/uvm-hashtable-${size}-double-100-add-no-dup-MONOTONIC_DECREASE.log
    ./gpu/hashtable/driver_hashtable_UVM-batch-sa-prefetch-double.out -ops=${size} -add=100 -rem=0 -tra="insert_trace-400e7-100-add-no-dup-MONOTONIC_INCREASE.bin" -trr="" -trf=""  -rns=${runs} -fil=1 > ${uvm_log_dir}/uvm-hashtable-${size}-double-100-add-no-dup-MONOTONIC_INCREASE-key-check-runs-${runs}.log
    ./gpu/hashtable/driver_hashtable_UVM-batch-sa-prefetch-double.out -ops=${size} -add=100 -rem=0 -tra="insert_trace-400e7-100-add-10-dup-DENSE_REPEAT.bin" -trr="" -trf=""  -rns=3 -fil=1 > ${uvm_log_dir}/uvm-hashtable-${size}-double-100-add-10-dup-DENSE_REPEAT-new-trace.log
    ./gpu/hashtable/driver_hashtable_UVM-batch-sa-prefetch-double.out -ops=${size} -add=100 -rem=0 -tra="insert_trace-400e7-100-add-20-dup-DENSE_REPEAT.bin" -trr="" -trf=""  -rns=3 -fil=1 > ${uvm_log_dir}/uvm-hashtable-${size}-double-100-add-20-dup-DENSE_REPEAT-new-trace.log
    ./gpu/hashtable/driver_hashtable_UVM-batch-sa-prefetch-double.out -ops=${size} -add=100 -rem=0 -tra="insert_trace-400e7-100-add-10-dup-SPARSE_REPEAT.bin" -trr="" -trf=""  -rns=3 -fil=1 > ${uvm_log_dir}/uvm-hashtable-${size}-double-100-add-10-dup-SPARSE_REPEAT-new-trace.log
    ./gpu/hashtable/driver_hashtable_UVM-batch-sa-prefetch-double.out -ops=${size} -add=100 -rem=0 -tra="insert_trace-400e7-100-add-20-dup-SPARSE_REPEAT.bin" -trr="" -trf=""  -rns=3 -fil=1 > ${uvm_log_dir}/uvm-hashtable-${size}-double-100-add-20-dup-SPARSE_REPEAT-new-trace.log
done
comment

# 2.0e9 Add 0.8e9 Rem 1.2e9 Find
#./gpu/hashtable/driver_hashtable_UVM-batch-sa-prefetch-double.out -ops=4000000000 -add=50 -rem=20 -dpa=0 -dpr=0 -dpf=0 -npd=0 -nps=0 -rns=3 -fil=1 > ${uvm_log_dir}/uvm_hahstable-400e7-50a-20r-double.log
# 2.0e9 Add 0.4e9 Rem 1.6e9 Find
#./gpu/hashtable/driver_hashtable_UVM-batch-sa-prefetch-double.out -ops=4000000000 -add=50 -rem=10 -dpa=20 -dpr=50 -dpf=40 -npd=20 -nps=10 -rns=3 -fil=1 > ${uvm_log_dir}/uvm_hahstable-400e7-50a-10r-20da-50dr-double.log
# 2.0e9 Add 0.0e9 Rem 2.0e9 Find
#./gpu/hashtable/driver_hashtable_UVM-batch-sa-prefetch-double.out -ops=5000000000 -add=50 -rem=20 -dpa=0 -dpr=0 -dpf=0 -npd=0 -nps=0 -rns=3 -fil=1 > ${uvm_log_dir}/uvm_hahstable-500e7-50a-20r-double.log
# 2.0e9 Add 0.4e9 Rem 1.6e9 Find
#./gpu/hashtable/driver_hashtable_UVM-batch-sa-prefetch-double.out -ops=5000000000 -add=50 -rem=10 -dpa=10 -dpr=40 -dpf=20 -npd=0 -nps=0 -rns=3 -fil=1 > ${uvm_log_dir}/uvm_hahstable-500e7-50a-10da-10r-40dr-20df-double.log
