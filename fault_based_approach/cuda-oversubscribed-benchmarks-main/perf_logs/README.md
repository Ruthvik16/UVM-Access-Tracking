# Saving Statistics

### `do_perf_run.sh`

`do_perf_run.sh` runs benchmarks and saves the console output in
`perf_run.vanilla.<PID>` or some similar file, and the individual Suneo stats in
another file with a long name.  

Inside the script, there is a bash function for each benchmark with a reasonable
data size. Instead of calling the top-level `run_benchmarks`, you can replace it
with what you want to run. Do NOT **commit** such changes, let those remain.

`grep`ing through the former file for `elapsed` should filter out the statistics
for easy analysis.

### `helpers.sh`

Helpers to load a specific driver version.

### `compare_policies.sh`

Wrapper around `do_perf_run.sh` to compare the vanilla driver, the profiler's
overhead, and driver configurations (remote mappings or a certain prefetching
threshold).

### Pausing or Exiting from Scripts run as Root

Look up `check_pause` and `check_exit` in `helpers.sh`.  
Also exits when assertions show up in `dmesg` - if run as root. `grep`s for
`kernel-open`, assuming the source code and assertion file names contain it.  

### Driver Bugs

`helpers.sh` has a function to check if there are errors from the driver in
`dmesg`, if the script is run as root.  

Check the start/end timestamps for each workload to tell if something is fishy.

### Deprecated

`get_one_line.sh` and `make_plot.py` are other helper scripts. The latter is not
supposed to be run alone. `./get_one_line.sh` prints a helpful help message if
you want to use them. At the moment, I have stopped using both.
General suggestion: maintain a backup of this directory, in `git` or elsewhere.
We will be doing a lot of `rm`s because of the number of faulty/duplicate Suneo

### Skimming through the statistics

(This is essentially for myself)

- Access distances with 10+ samples:
  `grep -h pg_ev * | grep -v  '([0-9]\s*samples' | sort -k3`

#### Misc

- Export `MEM_HINTS` before `do_perf_run.sh` to add CLI flags to workloads. Eg.
  `export MEM_HINTS="-gohan-static all"`
