#!/data/pranjal/cuda-benchmarks/rapids/rapids_venv/bin/python3]

import os
import subprocess
import matplotlib.pyplot as plt
from collections import OrderedDict

import pandas as pd
import numpy as np

verbose = 0
df = None

component_list_by_color = {
        "fetch_preprocess": "blue",
        "sfb_block_prelims": "cyan",
        "get_prefetch_hint": "green",
        "unmap_from_src": "red",
        "populate_dst": "magenta",
        "blk_copy_rw": "brown",
        "blk_copy_rdup": "grey",
        "service_finish": "purple",
        "tracker_wait": "orange",
        }

component_list = list(component_list_by_color.keys())

driver_type_formal_name = {
    "vanilla"    : "Vanilla",
    "rdup_simple": "RD",
    "rdup_tpf"   : "RD without redundant prefetches",
    "rdup_prot1" : "RD with protection promotion",
}

short_driver_names = {
    "vanilla": "V",
    "rdup_simple": "RD",
    "rdup_tpf": "RD2",
    "rdup_prot1": "RD3",
}



driver_types_global = list(driver_type_formal_name.keys())

gpu_size = 6<<30

def dump_dict(d):
    for key in d:
        print(f"{key}:\t{d[key]}")

def dump_nested_dict(d):
    for key1 in d:
        for key2 in d[key1]:
            value = d[key1][key2]
            if isinstance(value, str):
                if len(value) > 100:
                    value = "<insert long string>\n"
            elif isinstance(value, list):
                if len(value) > 10:
                    value = "<insert long list>\n"
            elif isinstance(value, np.ndarray):
                if value.shape[0] > 40:
                    value = "<insert long array>\n"
            elif '__len__' in dir(value):
                print("UNHANDLED vector type", value, type(value))
                exit()
            print(f"[{key1}][{key2}]: {value}")

def dict_list_to_np(d):
    for key in d:
        d[key] = np.array(d[key])

def get_oversub_ratio_xlabels(x):
    return [f"{(wss/gpu_size):.2f}" for wss in x]

def reorder_found_configs(df):
    configs = list(df.index.get_level_values(0).unique())
    if verbose:
        print(f"got these configs {configs}")

    configs2 = [cf for cf in ("vanilla", "rdup_simple") if cf in configs]

    configs.remove("vanilla")
    if "rdup_simple" in configs:
        configs.remove("rdup_simple")

    configs2 = configs2 + configs
    if verbose:
        print(f"returning this {configs2}")
    assert "rdup_tpf" in configs2
    return configs2

def parse_lines(lines):
    """
    works for any benchmark
    """
    if verbose:
        print(lines)

    components = {}

    for line in lines:
        if line.startswith("Breakdown"):
            continue
        if line.startswith("Total"):
            total_time = line.split()[3]
            total_time = int(total_time);
            if verbose:
                print(f"total time: {total_time} ms")
            components["h2d_total"] = total_time
            continue

        l2 = line.split(")")
        component = l2[0].split()[0]
        l2 = l2[1].split()
        if verbose:
            print(f"processing this {l2}")

        assert l2[0].isdecimal()
        assert l2[1] == "ms"
        assert l2[2].endswith("%")
        assert l2[3] == "|"
        assert l2[4].isdecimal()
        assert l2[5].endswith("M")
        assert l2[6] == "*"
        assert l2[7].isdecimal()
        assert l2[8] == "ns"
        assert l2[9] == "avg"

        call_count = int(l2[4])
        avg        = int(l2[7])
        total_time = int(l2[0])

        components[component] = total_time

    return components



def get_num_runs(fname, suneo_log):
    # use perf stat -r to do this
    that_line = [line for line in suneo_log if "perf stat" in line]
    if len(that_line) == 0:
        assert "iter" in fname
        fname_parts = fname.split("iter")
        fname_parts = fname_parts[-1].split(".")
        assert isinstance(int(fname_parts[0]), int)
        return int(fname_parts[0])

    try:
        assert len(that_line) == 1
    except:
        print(f"assert error in {fname}: perf stat occurs multiple times")
        exit(1)


    that_line = that_line[0]
    assert "perf stat -r " in that_line

    parts = that_line.split("perf stat -r")
    assert len(parts) == 2
    num_runs = int(parts[1].split()[0])
    assert 1 <= num_runs <= 20

    return num_runs

def get_total_time_bfs(fname, suneo_log):
    assert len(suneo_log)
    success_msg = "Overall Elapsed time in milliseconds"

    time_taken = [line for line in suneo_log if success_msg in line]
    try:
        assert len(time_taken) > 0
        assert ("iter" + str(len(time_taken))) in fname
    except:
        print(f"error parsing {fname}")
        # print(suneo_log)
        # exit(2)
        return 0


    total = 0
    try:
        for line in time_taken:
            time = line.split()[-2]
            time = int(time)
            total += time
    except:
        print(f"error parsing {fname}")
        print(f"error at line {line}")
        print(f"filtered lines were\n{time_taken}")
        exit(1)

    return time


def get_total_time_hashtable(fname, suneo_log):
    assert len(suneo_log)
    success_msg = "---------- Successful execution of the data structure ----------\n"
    assert success_msg in suneo_log

    time_taken = [line for line in suneo_log if "Total time taken by HeteroHash kernel (ms)" in line]
    assert len(time_taken) > 0
    assert ("iter" + str(len(time_taken))) in fname

    total = 0
    for line in time_taken:
        time = line.split()[-1]
        time = int(time)
        total += time

    return time


### this is NOT complete!!!
# return in seconds
def get_total_time(fname, suneo_log = None):
    # Return total time, NOT taking #runs into account
    if "hashtable" in fname:
        return get_total_time_hashtable(fname, suneo_log)
    if "bfs" in fname:
        return get_total_time_bfs(fname, suneo_log)

    result = subprocess.run(
        f"grep CPU+GPU {fname}".split(),
        capture_output = True,
        text = True,
        )

    lines = result.stdout.splitlines()
    if verbose:
        print(f"lines is {lines}")
    assert len(lines) != 0
    if len(lines) > 1:
        print(f"Found {len(lines)} occurences of CPU+GPU time in {fname}")

    total_time = 0
    for line in lines:
        part2 = line.split("CPU+GPU:")[1]
        part2 = part2.split()
        if verbose:
            print(part2)
        total_time += float(part2[0])

    return total_time

def get_one_suneo_line(suneo_output, search_key, get_count = False):
    grep_lines = [l for l in suneo_output if search_key in l]
    try:
        assert len(grep_lines) == 1
    except:
        print(grep_lines)
        exit()

    that_line = grep_lines[0]
    value = int(that_line.split()[3])
    if get_count:
        value = int(that_line.split()[0])
    return np.int64(value)


def get_mmap_size(suneo_output):
    mmap_lines = [l for l in suneo_output if "uvm_mmap_data_size" in l]
    assert len(mmap_lines) == 1
    that_line = mmap_lines[0]
    mmap_size = int(that_line.split()[3])
    return mmap_size

def read_one_file(fname, suneo_log = ""):
    the_text = suneo_log

    command = f"grep -A10 Breakdown {fname}"
    if verbose:
        print("running command", command)

    result = subprocess.run(
        command.split(),
        capture_output = True,
        text = True,
        )

    lines = result.stdout.splitlines()
    breakdown = parse_lines(lines)

    time = get_total_time(fname, suneo_log = suneo_log)
    mem_wss = get_mmap_size(the_text)

    return breakdown, time, mem_wss

def single_config_line_graphs(
        x,      # data sizes: need not be sorted
        y,      # dict: y = {faults = [400, 500], pf = [900, 4000], ...}
        title,
        png_prefix = "6",
        benchmark = "hello_world",
        config = "no_config",
        ylabel = "Count",
        ):

    assert isinstance(x, list) or isinstance(x, np.ndarray)
    assert isinstance(float(x[0]), float)
    assert isinstance(y, dict)

    plt.clf()
    for feature in y:
        assert isinstance(float(y[feature][0]), float)
        plt.plot(x, y[feature], '^--', label = feature)

    plt.yscale("symlog", linthresh = 1)
    plt.legend()
    plt.title(f"{benchmark}: {title}")
    plt.ylabel(ylabel)
    plt.xlabel("Oversubscription ratio (6 GiB)")

    plt.xticks(ticks = x, labels = get_oversub_ratio_xlabels(x))
    # plt.savefig(f"{png_prefix}_{config}_{benchmark}.png")
    plt.savefig(f"{benchmark.lower()}_{config}_{png_prefix}.png")
    plt.clf()

def total_exec_time_graph(
        x,
        t,
        title = "Execution time",
        png_prefix = "2",
        benchmark = "",
        ):
    """
    log graph, one bar per config
    x is per-config, not global. To be safe.
    t is obviously per-config

    I'm repurposing this to plot total fault count, and anything that is a
       per-config clustered bar plot.
    """
    assert isinstance(x, dict)
    assert isinstance(x["vanilla"], np.ndarray)
    assert isinstance(x["vanilla"][0], np.int64)
    assert isinstance(t, dict)
    assert isinstance(t["vanilla"], list) or isinstance(t["vanilla"], np.ndarray)


    wss_list = x["vanilla"] # roughly convers the whole span

    # If x has duplicates, get unique sorted values for spacing calculation
    min_gap = 1
    x_unique_sorted = np.sort(np.unique(wss_list))
    if x_unique_sorted.size > 1:
        min_gap = np.min(np.diff(x_unique_sorted))

    bar_width = 0.15 * min_gap

    driver_types = list(t.keys())

    offsets = np.linspace(-bar_width, bar_width, len(driver_types))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, label in enumerate(driver_types):
        ax.bar(
            x[label] + offsets[i],
            t[label],
            width=bar_width,
            label = driver_type_formal_name[label],
            align="center",
            )

    ax.set_xticks(wss_list)
    ax.set_xticklabels(get_oversub_ratio_xlabels(wss_list))
    # ax.set_xlabel("Data Size (x)")
    ax.set_xlabel("Oversubscription ratio (GDDR: 6 GiB)")
    ax.set_ylabel(f"{title} (s)")
    ax.set_yscale("symlog")

    ax.set_title(f"{benchmark.upper()}: {title} vs Oversubscription")
    ax.legend(title="Configuration")
    plt.tight_layout()
    plt.savefig(f"{benchmark.lower()}_{png_prefix}.png")


def stacked_bar_graph_2(x, y, benchmark = "hello"):
    """
    Different from make_stacked_bar_graph.
    This is one graph PER driver config, not all configs in one.
    configs and components can vary by orders of magnitude, so that was not a
    good idea.

    x: dict of lists: x[config] = [wss1, wss2, .. ]
    y: dict of dict of lists: y[config][component] = [t1, t2, ... ]

    Multiple graphs, one per config. In one big png
    """

    fig, axes = plt.subplots(2, 2, figsize = (20, 10))
    axes = axes.flatten()

    driver_configs = list(x.keys())
    fig.suptitle(f"{benchmark.upper()}: Per-component execution time")

    for i in range(len(driver_configs)):
        driver_type = driver_configs[i]
        ax = axes[i]

        df = pd.DataFrame(
            y[driver_type],
            index = x[driver_type],
            )

        df.sort_index(inplace = True)
        print(df)
        df.plot(
            kind = 'bar',
            ax = ax,
            xlabel = "Oversubscription",
            title = driver_type,
            xlim = (1, 10**8),
            logy = 'sym',
        )
        ax.set_xticklabels(get_oversub_ratio_xlabels(x[driver_type]))
        ax.set_xlabel("Oversubscription ratio (GDDR: 6 GiB)")

    plt.tight_layout()
    plt.savefig(f"{benchmark.lower()}_per_component_per_config_4.png")
    plt.close()


def make_stacked_bar_graph(x, y, t, benchmark = "hello"):
    """
    We hope but don't assume that all log files are listed in the "correct"
    order.
    x is a dict of list of data sizes
       x[driver_config] = [wss1, wss2, wss3, ... ]
    y is a dict of dict: y[config][component] = [c1, c2, .. ] for each x[i]
    t[driver_config] = [t1, t2, t3, ... ] for data sizes
    currently prints the percentage breakdown
    """
    assert isinstance(x, dict)
    x_base = x["vanilla"]
    assert isinstance(x_base, list) or isinstance(x_base, np.ndarray)
    assert isinstance(y, dict)
    for key in y:
        assert isinstance(y[key], dict)
    assert isinstance(t, dict)
    assert isinstance(t["vanilla"], list) or isinstance(t["vanilla"], np.array)

    num_data_sizes = len(x_base)

    # Bars per group
    bar_spacing = 0.18
    bar_width = 0.15

    # generated by perplexity
    x_list = np.arange(num_data_sizes)

    fig, ax = plt.subplots(figsize=(10,6))

    found_drivers = list(x.keys())
    for i in range(len(found_drivers)):
        driver_config = found_drivers[i]
        this_config_data = y[driver_config]
        num_data_sizes = len(this_config_data["fetch_preprocess"])

        bottom = np.zeros(num_data_sizes)
        driver_components_total = np.zeros(num_data_sizes)

        for comp in component_list:
            if comp == "h2d_total":
                continue
            driver_components_total += this_config_data[comp]

        for comp in component_list:
            if comp == "h2d_total":
                continue
            values = this_config_data[comp] / driver_components_total * 100
            assert isinstance(values, np.ndarray)
            if verbose: print(f"values is {values} ")

            ax.bar(
                x_list + i * bar_spacing,
                values,
                bar_width,
                bottom = bottom,
                label = f"{comp}" if i == 0 else "",
                alpha = 0.8,
                color = component_list_by_color[comp],
                )
            bottom += np.array(values)

        for x_loc in x_list:
            ax.text(
                x_loc + i * bar_spacing,
                100,
                short_driver_names[driver_config],
                ha = 'center',
                fontsize = 'x-small',
                )



    # Tidy up legend (show component colors just once)
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Components")

    # Set x-ticks for both bars in a group
    ax.set_xticks(x_list + bar_width/2)

    x_labels = get_oversub_ratio_xlabels(x_base)
    # ax.set_xticklabels([f"Size {i+1}" for i in data_sizes])
    ax.set_xticklabels(x_labels)


    ax.set_xlabel('Oversubscription against GPU RAM')
    ax.set_ylabel('Fraction of time spent')
    ax.set_title(f"{benchmark.upper()}: Breakdown of fault servicing")
    # ax.set_yscale("symlog")
    plt.tight_layout()

    plt_name = f"{benchmark.lower()}_percentage_3.png"
    plt.savefig(plt_name)
    plt.close()

def component_per_wss_per_benchmark(
        df,
        component_to_plot,
        denominator = None,
        benchmark = "hello world",
        png_prefix = 0,
        yscale = "log",
        ):

    """this one should be easy. make a new df and do df.plot.bar()
    new_df columns MUST be driver configs
    """
    # configs = df.index.get_level_values(0).unique()
    configs = reorder_found_configs(df)

    wss = list(df.index.get_level_values(1).unique())
    try:
        assert isinstance(wss[0], int)
    except:
        print(f"wss is {type(wss)} {wss}")

    full_idx = list(df.index)
    # an idx is ("vanilla", 4GiB) or something similar. full_idx is the list of
    # all.

    for ws in wss:
        if denominator:
            plt_name = f"{benchmark.lower()}_{ws//1000000}MB_per_{denominator}_{png_prefix}.png"
        else:
            plt_name = f"{benchmark.lower()}_{ws//1000000}MB_{png_prefix}.png"

        """ new_df structure:
              |vanilla rd1 rd2
        ------+----------------
        faults|
        prefet|
        min_ft|
        """
        new_df = pd.DataFrame(
            columns = configs,
            )


        for idx in full_idx:
            if idx[1] != ws:
                continue

            divide_by = df.loc[idx, denominator] if denominator else 1

            for comp in component_to_plot:
                new_df.loc[comp, idx[0]] = df.loc[idx, comp] / divide_by
                if yscale == "log" and new_df.loc[comp, idx[0]] < 0.0001:
                    new_df.loc[comp, idx[0]] = 1


        """
        # loop over each bar and add v/rd1/rd2 names
        for row in new_df.index:
            for column in new_df:
        """


        ax = new_df.plot.bar()

        """ does not work. the plt.legend() is enough
        for (i, patch) in enumerate(ax.patches):
            xpos = patch.get_x() + patch.get_width()/2
            ypos = patch.get_height();
            label = short_driver_names[configs[(i%len(configs))]]
            ax.text(xpos, ypos, label, ha='center',va='bottom',
                fontsize='x-small',
                )
        """


        ax.set_xlabel("Driver component")
        ax.set_ylabel("Average/total time (ns)")
        ax.set_yscale(yscale)
        ax.set_title(f"Variation of driver components by configs: {benchmark.upper()}, {ws//1000000} MB")

        plt.tight_layout()
        plt.savefig(plt_name)
        plt.close()
        print(f"Generated {plt_name}")



def clustered_stacked_bar_graph(
        df,
        components_to_plot,
        denominator = None,
        benchmark = "hello_world",
        png_prefix = "8",
        ):
    """
    this can scale well BECAUSE we use a per-batch or per-fault denominator
    """
    configs = reorder_found_configs(df)
    wss = df.index.get_level_values(1).unique()
    all_components = df.columns
    components = components_to_plot

    n_configs = len(configs)
    n_wss = len(wss)
    bar_width = 0.22

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab10.colors[:len(components)]

    # Clustered Stacked Bar Plot
    for ci, config in enumerate(configs):
        x_base = np.arange(n_wss)
        bottoms = np.zeros(n_wss)
        for comp_idx, comp in enumerate(components):
            # heights: one per wss for this config/comp
            heights = np.array([df.loc[(config, ws), comp] for ws in wss])
            if denominator:
                # print(f"dividing by {list(df.loc[(config, wss)])}")
                assert 0 not in [df.loc[(config, ws), denominator] for ws in wss]
                heights = np.array([df.loc[(config, ws), comp]/df.loc[(config, ws), denominator] for ws in wss])

            # Cluster position for each config within group
            xs = x_base + (ci - (n_configs-1)/2) * bar_width
            # Label only once for legend
            label = comp if (ci == 0) else None
            ax.bar(xs, heights, bar_width, bottom=bottoms, color=colors[comp_idx], label=label)
            bottoms += heights

            # write the config over the graph
            if comp == components[-1]:
                for wss_i in range(n_wss):
                    ax.text(
                        xs[wss_i],
                        bottoms[wss_i],
                        short_driver_names[config],
                        ha = 'center',
                        fontsize = 'small',
                        )


    # Set x-ticks at base positions; label as wss
    ax.set_xticks(np.arange(n_wss))
    # ax.set_xticklabels([str(ws) for ws in wss])
    # ax.set_xlabel("Working Set Size (wss)")
    # ax.set_title("Clustered and Stacked Bar Chart by Config and Component")
    ax.set_xticklabels(get_oversub_ratio_xlabels(wss))
    ax.set_xlabel("Oversubscription (6 GiB)")
    ax.set_ylabel("Time Taken (ns)")
    if denominator:
        ax.set_title(f"{benchmark.upper()} Breakdown: per " + denominator)
    else:
        ax.set_title(f"{benchmark.upper()} Breakdown")


    # Deduplicate legend (one entry per component)
    handles, labels = ax.get_legend_handles_labels()
    from collections import OrderedDict
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1,1))

    plt.tight_layout()
    plot_name = f"{benchmark.lower()}_breakdown"
    if denominator:
        plot_name += f"_per_{denominator}"
    plt.savefig(f"{plot_name}_{png_prefix}.png")

    plt.close()

def driver(benchmark = 'atax', make_plots = True):

    all_files = os.listdir()
    filtered_list = [file for file in all_files
            if (benchmark in file and "png" not in file)]
    filtered_list.sort()

    """
    data_sizes = [str(gb) + "000000000" for gb in (4, 5, 6, 7, 8, 9, 10)]
    sorted_list = []
    for wss in data_sizes:
        sorted_list += [k for k in filtered_list if wss in k]
    """

    final_logs_list = filtered_list
    print(f"processing these files")
    garbage = [print(t) for t in final_logs_list]
    assert final_logs_list

    wss_list = [] # working set size
    wss_per_config = {}

    time_per_config_per_component = {}
    total_time_per_config         = {}
    fault_count_per_config = {}

    """
    for config in driver_types:
        time_per_config_per_component[config] = {}
        total_time_per_config[config]         = []
        wss_per_config[config]                = []
        fault_count_per_config[config]        = []

        time_per_config_per_component[config]["h2d_total"] = []
        for c in component_list:
                time_per_config_per_component[config][c] = []
                """



    """
    Gather these stats:
    read faults, write faults, fault batches, CPU faults, H2D prefetches (real
    and already-there), eviction count, this, that, ...

    IDEA: put everything in a big dict: per_log_data[log_file] = {"rd faults":
    [], "wr faults": [], ... }
    """

    per_log_data = {}

    columns_big_df = component_list + [
        "fname"            ,
        "gpu_read_faults"  ,
        "gpu_write_faults" ,
        "h2d_prefetch_all" ,
        "num_batches"      ,
        "evicted_blks"     ,
        "evict_time"       ,
        "evict_pages_num"  ,
        "evict_rdup_page"  ,
        "h2d_pages_copied" ,
        "gpu_minor_faults" ,
        "external_time"    ,
        "h2d_total"        ,
        "populate_wo_evict",
        ]

    the_big_df = pd.DataFrame(
            index = pd.MultiIndex.from_tuples([] , names=["config", "wss"]),
            columns = columns_big_df,
            )

    for fname in final_logs_list:
        with open(fname) as file:
            the_text = file.readlines()

        breakdown, total_time, wss = read_one_file(fname, the_text)
        num_runs = get_num_runs(fname, the_text)
        assert wss % num_runs == 0
        wss = wss // num_runs

        driver_config = None
        for t in driver_types_global:
            if t in fname:
                assert driver_config == None # match only one
                driver_config = t

        if driver_config not in time_per_config_per_component:
            time_per_config_per_component[driver_config] = {}

            time_per_config_per_component[driver_config]["h2d_total"] = []
            for c in component_list:
                    time_per_config_per_component[driver_config][c] = []
        if driver_config not in total_time_per_config:
            total_time_per_config[driver_config] = []
        if driver_config not in wss_per_config:
            wss_per_config[driver_config] = []
        if driver_config not in fault_count_per_config:
            fault_count_per_config[driver_config] = []


        if None in [breakdown, total_time, wss]:
            print(f"file {fname}: got some dummy values, please check")

        the_big_df.loc[(driver_config, wss), "fetch_preprocess"] = breakdown["fetch_preprocess"]

        for c in breakdown:
            value = breakdown[c]
            assert c in time_per_config_per_component[driver_config]
            time_per_config_per_component[driver_config][c].append(value/num_runs)
            """
            if c == "h2d_total":
                continue
            try:
                assert c in the_big_df.columns
                the_big_df.loc[(driver_config, wss), c] = value
            except:
                print(c, the_big_df.columns)
                exit(0)
            """

        total_time_per_config[driver_config].append(total_time/num_runs)

        wss_per_config[driver_config].append(wss)
        if driver_config == "vanilla":
            wss_list.append(wss)

        write_faults = get_one_suneo_line(the_text, "dedup_write")
        read_faults  = get_one_suneo_line(the_text, "dedup_read")
        h2d_total_prefetch = get_one_suneo_line(the_text, "H2D_num_prefetch_pages")
        num_batches = get_one_suneo_line(the_text, "preprocess_faults", get_count = True)
        num_evict_blk = get_one_suneo_line(the_text, "evict_chunk", get_count = True)
        evict_cost    = get_one_suneo_line(the_text, "pmm_evict_GPU_alloc")
        evict_pages_real = get_one_suneo_line(the_text, "eviction_page_copy_count")
        assert evict_pages_real == get_one_suneo_line(the_text, "blk_crp_evict_pg_count")
        evict_pages_but_rd = get_one_suneo_line(the_text, "eviction_rdup_page_count")
        minor_fault_count = get_one_suneo_line(the_text, "gpu_write_to_rdup_page_count")

        # FILL UP per-log-data
        per_log_data[fname] = {}
        per_log_data[fname]["full_text"] = the_text
        per_log_data[fname]["gpu_read_faults"] = read_faults
        per_log_data[fname]["gpu_write_faults"] = write_faults
        per_log_data[fname]["h2d_prefetch_all"] = h2d_total_prefetch
        per_log_data[fname]["num_batches"]      = num_batches
        per_log_data[fname]["wss"]              = wss
        per_log_data[fname]["evicted_blks"]     = num_evict_blk
        per_log_data[fname]["evict_time"]       = evict_cost
        per_log_data[fname]["evict_pages_num"]  = evict_pages_real
        per_log_data[fname]["evict_rdup_page"]  = evict_pages_but_rd
        per_log_data[fname]["h2d_pages_copied"] = get_one_suneo_line(the_text, "copy_resd_pages_h2d_total")
        per_log_data[fname]["gpu_minor_faults"] = minor_fault_count
        per_log_data[fname]["gpu_maj_wr_faults"] = write_faults - minor_fault_count
        assert (write_faults >= minor_fault_count)
        try:
            if "prot" not in driver_config:
                assert write_faults >= minor_fault_count
        except:
            print(f"ERROR in {fname}: write_faults {write_faults} < minor_fault_count {minor_fault_count}")
        per_log_data[fname]["external_time"]    = total_time
        per_log_data[fname]["h2d_total"]  = get_one_suneo_line(the_text, "uvm_gpu_srf_overall")
        per_log_data[fname]["populate_wo_evict"]= 0

        idx = (driver_config, wss)

        per_log_data[fname]["num_runs"] = num_runs
        for key in per_log_data[fname]:
            if key in ["full_text", "wss"]:
                continue
            try:
                per_log_data[fname][key] /= num_runs
            except:
                print(f"per_log_data: cannot update key {key}")

        the_big_df.loc[idx, "fname"] = fname
        the_big_df.loc[idx, "gpu_read_faults"] = read_faults
        the_big_df.loc[idx, "gpu_write_faults"] = write_faults
        the_big_df.loc[idx, "gpu_dedup_faults"] = int(read_faults + write_faults)
        the_big_df.loc[idx, "h2d_prefetch_all"] = h2d_total_prefetch
        the_big_df.loc[idx, "num_batches"] = num_batches
        the_big_df.loc[idx, "evicted_blks"] = num_evict_blk
        the_big_df.loc[idx, "evict_time"] = evict_cost
        the_big_df.loc[idx, "evict_pages_num"] = evict_pages_real
        the_big_df.loc[idx, "evict_rdup_page"] = evict_pages_but_rd
        the_big_df.loc[idx, "h2d_pages_copied"] = get_one_suneo_line(the_text, "copy_resd_pages_h2d_total")
        the_big_df.loc[idx, "gpu_minor_faults"] = minor_fault_count
        the_big_df.loc[idx, "external_time"] = total_time
        the_big_df.loc[idx, "h2d_total"] = get_one_suneo_line(the_text, "uvm_gpu_srf_overall")
        the_big_df.loc[idx, "populate_wo_evict"] = breakdown["populate_dst"] - evict_cost
        the_big_df.loc[idx, "gpu_maj_wr_faults"] = write_faults - minor_fault_count
        assert the_big_df.loc[idx, "gpu_maj_wr_faults"] >= 0
        ## These are not in per_log_data
        the_big_df.loc[idx, "fetch_preprocess"] = get_one_suneo_line(the_text, "fetch_faults") + \
                get_one_suneo_line(the_text, "preprocess_faults")
        the_big_df.loc[idx, "sfb_block_prelims"] = get_one_suneo_line(the_text, "sfb_block_locked_prelims")
        the_big_df.loc[idx, "get_prefetch_hint"] = get_one_suneo_line(the_text, "get_prefetch_H2D")
        the_big_df.loc[idx, "unmap_from_src"] = get_one_suneo_line(the_text, "unmap_pages_from_source_H2D") + \
                get_one_suneo_line(the_text, "mk_resd_rdup:unmap")
        the_big_df.loc[idx, "populate_dst"] = get_one_suneo_line(the_text, "block_populate_pages_H2D")
        the_big_df.loc[idx, "blk_copy_rw"] = get_one_suneo_line(the_text, "mk_res_copy:blk_copy_resident_pages") # 35
        the_big_df.loc[idx, "blk_copy_rdup"] = get_one_suneo_line(the_text, "mk_resd_rdup:crp_between") # 52
        the_big_df.loc[idx, "service_finish"] = get_one_suneo_line(the_text, "service_finish_H2D") # 20
        the_big_df.loc[idx, "tracker_wait"] = get_one_suneo_line(the_text, "uvm-tracker-wait") # 4
        the_big_df.loc[idx, "h2d_pages_processed"] = the_big_df.loc[idx, "gpu_dedup_faults"] + \
                the_big_df.loc[idx, "h2d_prefetch_all"]


        fault_count_per_config[driver_config].append(read_faults + write_faults)

        per_log_data[fname]["full_text"] = 0

        the_big_df.loc[idx, "num_runs"] = num_runs
        for col in the_big_df.columns:
            if col == "fname":
                continue
            the_big_df.loc[idx, col] = np.int64(the_big_df.loc[idx, col] / num_runs)


    print(wss_list)
    dump_nested_dict(time_per_config_per_component)
    dump_dict(total_time_per_config)


    dump_nested_dict(per_log_data)

    print(time_per_config_per_component)
    dict_list_to_np(wss_per_config)

    for column in the_big_df.columns:
        if column == 'fname':
            continue
        the_big_df[column] = the_big_df[column].apply(lambda x: np.int64(x))

    if make_plots == False:
        return the_big_df
    for driver_type in time_per_config_per_component:
        dict_list_to_np(time_per_config_per_component[driver_type])

    # 4_*.png
    make_stacked_bar_graph(
        wss_per_config,
        time_per_config_per_component,
        total_time_per_config,
        benchmark = benchmark
        );

    total_exec_time_graph(
        wss_per_config,
        total_time_per_config,
        benchmark = benchmark,
        )


    """
    this is f"{benchmark}_per_component_per_config_4.png"
    we don't want it any more.
    stacked_bar_graph_2(
        wss_per_config,
        time_per_config_per_component,
        benchmark = benchmark
        )
    """

    # fault count


    found_drivers = list(wss_per_config.keys())
    for driver in list(found_drivers):
        if wss_per_config[driver].size == 0:
            wss_per_config.pop(driver)
            found_drivers.remove(driver)

    """ not needed, plot 12/13 has this
    total_exec_time_graph(
        wss_per_config,
        fault_count_per_config,
        title = "GPU faults",
        benchmark = benchmark,
        png_prefix = "5",
        )
    """

    # plot rd faults, write faults, minor faults, major faults, t+f prefetches,
    # evictions, evictions avoided, etc.

    page_stats_parameters_to_plot = [
        "gpu_read_faults",
        "gpu_write_faults",
        "gpu_minor_faults",
        "gpu_maj_wr_faults",
        "h2d_prefetch_all",
        "h2d_pages_copied",
        "evicted_blks",
        "evict_pages_num",
        "evict_rdup_page",
        "num_batches",
        ]
    print(found_drivers)
    print(wss_per_config.keys())

    for driver in found_drivers:
        this_config_logs = [t for t in final_logs_list if driver in t]
        assert this_config_logs

        x = []
        y = {}

        for key in page_stats_parameters_to_plot:
            y[key] = []

        # TODO change this to use the_big_df
        for workload in this_config_logs:
            for key in page_stats_parameters_to_plot:
                y[key].append(per_log_data[workload][key])
            x.append(per_log_data[workload]["wss"])

        print(x)
        assert isinstance(x[0], int)
        single_config_line_graphs(
            x,
            y,
            title = "Fault, prefetch and eviction statistics",
            benchmark = benchmark,
            config = driver,
            )

    # clustered_stacked_bar_graph(wss_per_config, time_per_config_per_component, benchmark = benchmark)
    clustered_stacked_bar_graph(
        the_big_df,
        component_list,
        denominator = "num_batches",
        benchmark = benchmark,
        png_prefix = 8,
        )

    clustered_stacked_bar_graph(
        the_big_df,
        component_list,
        denominator = "gpu_dedup_faults",
        benchmark = benchmark,
        png_prefix = 9,
        )

    """
    clustered_stacked_bar_graph(
        the_big_df,
        component_list,
        denominator = "h2d_pages_processed",
        benchmark = benchmark,
        png_prefix = 10,
        )
    """

    clustered_stacked_bar_graph(
        the_big_df,
        component_list,
        denominator = "h2d_pages_copied",
        benchmark = benchmark,
        png_prefix = 11,
        )

    """
    component_per_wss_per_benchmark(
        the_big_df,
        page_stats_parameters_to_plot,
        denominator = "num_batches",
        benchmark =  benchmark,
        png_prefix = 12,
        )

    component_per_wss_per_benchmark(
        the_big_df,
        page_stats_parameters_to_plot,
        denominator = "h2d_pages_processed",
        benchmark =  benchmark,
        png_prefix = 13,
        )
    """

    component_per_wss_per_benchmark(
        the_big_df,
        page_stats_parameters_to_plot,
        denominator = None,
        benchmark =  benchmark,
        png_prefix = 13,
        )

    component_per_wss_per_benchmark(
        the_big_df,
        component_list,
        denominator = None,
        benchmark = benchmark,
        png_prefix = 14,
        )


    print(the_big_df)
    global df
    df = the_big_df
    return the_big_df
    # end of driver


if __name__ == "__main__":
    # driver("mvt")
    driver("2DConv")
    driver("atax")
    # driver("2mm")
    driver("hashtable")
    driver("bfs")

# known bugs: sometimes the driver components add up to _more_ than the h2d
# total
