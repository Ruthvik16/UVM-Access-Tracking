#!/bin/sh

mkdir -p Dataset/test
mkdir -p Dataset/train

cd Dataset/test
wget https://raw.githubusercontent.com/OSU-STARLAB/UVM_benchmark/refs/heads/master/UVM_benchmarks/logistic-regression/Dataset/test/dev-first10.arff
wget https://raw.githubusercontent.com/OSU-STARLAB/UVM_benchmark/refs/heads/master/UVM_benchmarks/logistic-regression/Dataset/test/dev-first1000.arff
wget https://raw.githubusercontent.com/OSU-STARLAB/UVM_benchmark/refs/heads/master/UVM_benchmarks/logistic-regression/Dataset/test/dev-first200.arff
wget https://raw.githubusercontent.com/OSU-STARLAB/UVM_benchmark/refs/heads/master/UVM_benchmarks/logistic-regression/Dataset/test/dev-first50.arff

cd ../train
wget https://raw.githubusercontent.com/OSU-STARLAB/UVM_benchmark/refs/heads/master/UVM_benchmarks/logistic-regression/Dataset/train/train-first10.arff
wget https://raw.githubusercontent.com/OSU-STARLAB/UVM_benchmark/refs/heads/master/UVM_benchmarks/logistic-regression/Dataset/train/train-first1000.arff
wget https://raw.githubusercontent.com/OSU-STARLAB/UVM_benchmark/refs/heads/master/UVM_benchmarks/logistic-regression/Dataset/train/train-first200.arff
wget https://raw.githubusercontent.com/OSU-STARLAB/UVM_benchmark/refs/heads/master/UVM_benchmarks/logistic-regression/Dataset/train/train-first50.arff
