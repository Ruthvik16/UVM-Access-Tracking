#!/bin/bash
# params are hardcoded atm

prepared_benchmarks="2MM 3MM 2DCONV 3DCONV ATAX BICG MVT"
mem_footprint=10000000000 #10B
n_iter=2
outfile=perf_log
repo_root="/data/pranjal/cuda-oversubscribed-benchmarks/"
run_description_suffix="   # single process, 10G, vanilla driver"

counter_file=/proc/bkbuvm/debug
if [ ! -d $repo_root ]
then
	echo change find_kexec_time.sh
	echo $repo_root not found: have you cloned the repo?
	echo "Change \"repo_root\" in the script"
	exit 1
fi

cd $repo_root
cd polybench

echo "saving output to $repo_root/polybench/$outfile"
echo "Beginning next run (not clearing old logs)" >> $outfile
for iter in `seq 1 $n_iter`
do
	for i in $prepared_benchmarks
	do
		# echo clear | sudo tee $counter_file
		echo Benchmark: $i, iter: $iter
		echo -ne "($iter)\t"	>> $outfile
		$i/*exe $mem_footprint -copy-back | tail -n 1 | tr -d '\n' >> $outfile
		echo "   $run_description_suffix" >> $outfile
		# sed -n '1,5p; 15p; 19,21p; 38p; 69,70p' $counter_file >> $outfile
	done
done

echo ===============================
echo Now printing per-benchmark-per-run time "(in order)"

for b in $prepared_benchmarks
do
	grep -i "$b" $outfile
	echo 
done

