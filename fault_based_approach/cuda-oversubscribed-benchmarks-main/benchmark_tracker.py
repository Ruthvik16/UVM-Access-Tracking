#!/usr/bin/env python3
"""
Benchmark CUDA UVM access tracker overhead.

Methodology
-----------
1. A background CUDA process (mem_blocker.out) allocates BLOCKER_GB of GPU
   memory via cudaMalloc (non-UVM), leaving ~USABLE_MEM_MB free for workloads.
2. Each workload is sized to use (USABLE_MEM_MB * oversubscription_pct / 100) MB
   of UVM memory.  At 100% the allocation fits; at 200% half must page-fault out.
3. Phase 1: run every workload with the tracker disabled (baseline).
4. Phase 2: enable /proc/uvm_sampling_tracker, then repeat every workload.
5. Collect GPU kernel runtime from workload stdout and tracker fault/page stats
   from the procfs entry.  Compute overhead = (tracker - baseline) / baseline.

Compilation flags
-----------------
All CUDA binaries are built with:
  nvcc -O3 -Wno-deprecated-gpu-targets -gencode arch=compute_61,code=sm_61
sgemm adds -lcublas.

Usage
-----
  python3 benchmark_tracker.py                        # all workloads
  python3 benchmark_tracker.py -w rodinia_nw sgemm   # subset
  python3 benchmark_tracker.py --rebuild              # force recompile
  python3 benchmark_tracker.py --no-tracker           # baseline only
  python3 benchmark_tracker.py --list                 # list workloads
  python3 benchmark_tracker.py --runs 5 --plot        # 5 reps + PNG plots
  python3 benchmark_tracker.py --runs 5 --html        # 5 reps + HTML report
  python3 benchmark_tracker.py --plot-json results.json  # PNG plots from JSON
  python3 benchmark_tracker.py --html-json results.json  # HTML report from JSON
"""

import io
import os
import re
import sys
import json
import time
import base64
import signal
import argparse
import subprocess
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.patches import Patch
    _MATPLOTLIB = True
except ImportError:
    _MATPLOTLIB = False

# ── Configuration ─────────────────────────────────────────────────────────────

REPO_ROOT      = Path(__file__).parent.resolve()   # always absolute
TRACKER_PROCFS = "/proc/uvm_sampling_tracker"

# GPU memory we want left free for UVM workloads after blocking
USABLE_MEM_MB  = 1024      # 1 GB

# Combined CUDA context overhead for both the blocker process and the workload
# process (~200-400 MB each on Pascal/Volta; use 500 MB per process = 1 GB total)
CONTEXT_OVERHEAD_MB = 1000

# nvcc flags used for all CUDA builds
NVCC_ARCH  = "-gencode arch=compute_61,code=sm_61"
NVCC_FLAGS = f"-O3 -Wno-deprecated-gpu-targets {NVCC_ARCH}"

# Filled in at startup by _init_gpu_info()
_gpu_free_mb:   int = 0
_blocker_mb:    int = 0

# Oversubscription levels per workload (% of USABLE_MEM_MB)
OVERSUBSCRIPTION: Dict[str, List[int]] = {
    "rodinia_nw":       [150, 200],
    "polybench_mvt":    [150],
    "sgemm":            [100, 150, 200],
    "polybench_2dconv": [100],
}

# Per-workload timeout in seconds
TIMEOUTS: Dict[str, int] = {
    "rodinia_nw":       600,
    "polybench_mvt":    600,
    "sgemm":            600,
    "polybench_2dconv": 600,
}


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    name:          str
    variant:       str
    tracker_on:    bool
    oversub_pct:   int
    gpu_runtime_s: float
    wall_time_s:   float
    faults:        int
    unique_pages:  int
    error:         Optional[str] = None

    def ok(self) -> bool:
        return self.error is None


# ── Tracker ───────────────────────────────────────────────────────────────────
class StatPoller(threading.Thread):
    """
    Polls the tracker in the background to capture the peak fault 
    counts before the process exits and the kernel clears the data.
    """
    def __init__(self, interval=0.001):
        super().__init__()
        self.daemon = True
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_stats = {"accessed": 0, "rich_metadata": 0}

    def run(self):
        while not self.stop_event.is_set():
            current = Tracker.stats()
            if "accessed" in current:
                self.peak_stats["accessed"] = max(self.peak_stats["accessed"], current["accessed"])
            if "rich_metadata" in current:
                self.peak_stats["rich_metadata"] = max(self.peak_stats["rich_metadata"], current["rich_metadata"])
            
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()

class Tracker:
    @staticmethod
    def available() -> bool:
        return Path(TRACKER_PROCFS).exists()

    @staticmethod
    def _write(cmd: str) -> bool:
        try:
            Path(TRACKER_PROCFS).write_text(cmd + "\n")
            return True
        except Exception as e:
            print(f"  [tracker] write {cmd!r} failed: {e}")
            return False

    @staticmethod
    def enable()  -> bool: return Tracker._write("enable 1")

    @staticmethod
    def disable() -> bool: return Tracker._write("enable 0")

    @staticmethod
    def clear()   -> bool: return Tracker._write("clear")
        
    @staticmethod
    def stats() -> Dict:
        try:
            content = Path(TRACKER_PROCFS).read_text()
            out = {}
            for line in content.splitlines():
                for k, v in re.findall(r'(\w+)=(\d+)', line):
                    out[k.lower()] = int(v)
            return out
        except Exception:
            return {}


# ── Memory Blocker ────────────────────────────────────────────────────────────

def _get_gpu_free_mb() -> int:
    """Return free GPU memory in MB via nvidia-smi (GPU 0)."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            return int(r.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return 0


def _init_gpu_info(override_blocker_mb: int = 0):
    """
    Query free GPU memory and compute how much to block so that
    approximately USABLE_MEM_MB remains available to each workload.

    override_blocker_mb: if > 0, use this value instead of auto-detecting.
    """
    global _gpu_free_mb, _blocker_mb
    _gpu_free_mb = _get_gpu_free_mb()
    if override_blocker_mb > 0:
        _blocker_mb = override_blocker_mb
    elif _gpu_free_mb > 0:
        _blocker_mb = max(0, _gpu_free_mb - USABLE_MEM_MB - CONTEXT_OVERHEAD_MB)
    else:
        _blocker_mb = 0   # can't determine; skip blocking


class MemBlocker:
    """
    Spawn a background CUDA process that holds _blocker_mb of GPU memory
    via cudaMalloc (non-UVM), shrinking the free device memory visible to
    UVM workloads to ~USABLE_MEM_MB.

    Prints "READY" once the allocation succeeds; exits cleanly on SIGTERM.
    Use as a context manager around each workload invocation.
    """

    SRC = REPO_ROOT / "helpers" / "mem_blocker.cu"
    BIN = REPO_ROOT / "helpers" / "mem_blocker.out"

    @classmethod
    def build(cls, force: bool = False) -> bool:
        if cls.BIN.exists() and not force:
            return True
        if not cls.SRC.exists():
            print(f"  [blocker] source missing: {cls.SRC}")
            return False
        print("  [blocker] building mem_blocker.out ...")
        r = subprocess.run(
            f"nvcc {NVCC_FLAGS} {cls.SRC} -o {cls.BIN}".split(),
            capture_output=True, text=True, timeout=90,
        )
        if r.returncode != 0:
            print(f"  [blocker] build FAILED:\n{r.stderr[:600]}")
            return False
        print("  [blocker] built OK")
        return True

    def __init__(self):
        self._bytes = _blocker_mb * (1 << 20)
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> bool:
        if self._bytes == 0:
            return True   # nothing to block (GPU memory couldn't be detected)
        if not self.BIN.exists():
            if not self.build():
                return False
        self._proc = subprocess.Popen(
            [str(self.BIN), str(self._bytes)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        line = self._proc.stdout.readline()   # block until "READY\n" or EOF
        if self._proc.poll() is not None or not line.startswith("READY"):
            err = self._proc.stderr.read() if self._proc.stderr else ""
            print(f"    [blocker] FAILED to start ({_blocker_mb} MB): "
                  f"{line.strip()!r}  {err[:200]}")
            self._proc = None
            return False
        return True

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.send_signal(signal.SIGTERM)
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._proc = None

    def __enter__(self):
        if not self.start():
            print(f"  [blocker] WARNING: failed to start — workload will run without memory pressure")
        return self

    def __exit__(self, *_):
        self.stop()


# ── Runtime Parser ────────────────────────────────────────────────────────────

def parse_gpu_time(output: str) -> float:
    """
    Extract GPU kernel time in seconds from workload stdout/stderr.

    Patterns handled:
      MVT / 2DCONV : "GPU Runtime: 1.234567s"
      sgemm        : "sgemm.cu: GPU run time: 1.234 s"
      needle       : "# kernel1 took 1.234 s" + "# kernel2 took 5.678 s" (summed)
    """
    # MVT, 2DCONV: "GPU Runtime: 1.234567s"
    m = re.search(r'GPU\s+Runtime:\s*([\d.]+)s', output, re.I)
    if m:
        return float(m.group(1))

    # sgemm: "GPU run time: 1.234 s"
    m = re.search(r'GPU\s+run\s+time:\s*([\d.]+)\s*s', output, re.I)
    if m:
        return float(m.group(1))

    # needle: "# kernel1 took 1.234 s\n# kernel2 took 5.678 s" — sum both kernels
    matches = re.findall(r'kernel\d+\s+took\s+([\d.]+)\s*s', output, re.I)
    if matches:
        return sum(float(x) for x in matches)

    # fallback: "/usr/bin/time" style "12.34 seconds time elapsed"
    m = re.search(r'([\d.]+)\s+seconds\s+time\s+elapsed', output)
    if m:
        return float(m.group(1))

    return 0.0


# ── Build helpers ─────────────────────────────────────────────────────────────

def _nvcc(src: str, out: str, extra: str, cwd: Path) -> bool:
    """Run nvcc with the standard flags.  Returns True on success."""
    cmd = f"nvcc {NVCC_FLAGS} {extra} {src} -o {out}".split()
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        print(f"  [build fail]\n{r.stderr[:600]}")
        return False
    return True


def build_needle(force: bool = False) -> Optional[Path]:
    d    = REPO_ROOT / "rodinia" / "nw"
    bin_ = d / "needle"
    if bin_.exists() and not force:
        return bin_
    print("  Building rodinia/nw/needle ...")
    return bin_ if _nvcc("needle.cu", "needle", "", d) and bin_.exists() else None


def build_mvt(force: bool = False) -> Optional[Path]:
    d    = REPO_ROOT / "polybench" / "MVT"
    bin_ = d / "mvt.exe"
    if bin_.exists() and not force:
        return bin_
    print("  Building polybench/MVT/mvt.exe ...")
    return bin_ if _nvcc("mvt.cu", "mvt.exe", "", d) and bin_.exists() else None


def build_sgemm(force: bool = False) -> Optional[Path]:
    d    = REPO_ROOT / "nvidia-samples" / "sgemm"
    bin_ = d / "sgemm.out"
    if bin_.exists() and not force:
        return bin_
    print("  Building nvidia-samples/sgemm/sgemm.out ...")
    return bin_ if _nvcc("sgemm.cu", "sgemm.out", "-lcublas", d) and bin_.exists() else None


def build_2dconv(force: bool = False) -> Optional[Path]:
    d    = REPO_ROOT / "polybench" / "2DCONV"
    bin_ = d / "2DConvolution.exe"
    if bin_.exists() and not force:
        return bin_
    print("  Building polybench/2DCONV/2DConvolution.exe ...")
    return bin_ if _nvcc("2DConvolution.cu", "2DConvolution.exe", "", d) and bin_.exists() else None


# ── Workload execution ────────────────────────────────────────────────────────

def _exec(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 300, tracker_on: bool = False) -> tuple:
    """Run cmd; return (wall_s, combined_output, returncode, peak_stats)."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}

    poller = None
    if tracker_on:
        poller = StatPoller(interval=0.001)  # 1ms polling
        poller.start()

    t0 = time.time()
    try:
        r = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            timeout=timeout, env=env,
        )
        wall_time = time.time() - t0
        output = r.stdout + "\n" + r.stderr
        rc = r.returncode
        # Synchronous read immediately after the process exits, before the CUDA
        # context tears down and the driver potentially clears the tracker state.
        if tracker_on and poller:
            final = Tracker.stats()
            if "accessed" in final:
                poller.peak_stats["accessed"] = max(poller.peak_stats["accessed"], final["accessed"])
            if "rich_metadata" in final:
                poller.peak_stats["rich_metadata"] = max(poller.peak_stats["rich_metadata"], final["rich_metadata"])
    except subprocess.TimeoutExpired:
        wall_time = time.time() - t0
        output, rc = "TIMEOUT", -9
    except Exception as e:
        wall_time = time.time() - t0
        output, rc = str(e), -1
    finally:
        peak_results = {"accessed": 0, "rich_metadata": 0}
        if poller:
            poller.stop()
            poller.join()
            peak_results = poller.peak_stats

    return wall_time, output, rc, peak_results

def _make(name, variant, tracker_on, pct, gpu_s, wall_s, output, rc, peak_stats) -> BenchResult:
    if rc == 0:
        error = None
    elif rc == -9:
        error = "timeout"
    else:
        error = output[-400:].strip()
    
    return BenchResult(
        name=name, variant=variant, tracker_on=tracker_on,
        oversub_pct=pct, gpu_runtime_s=gpu_s, wall_time_s=wall_s,
        faults=peak_stats.get("accessed", 0),
        unique_pages=peak_stats.get("rich_metadata", 0), # 'rich' means unique pages in current epoch
        error=error,
    )


def run_rodinia_nw(pct: int, tracker_on: bool, force_build: bool = False) -> BenchResult:
    name, variant = "rodinia_nw", f"needle_{pct}pct"
    bin_ = build_needle(force_build)
    if not bin_:
        return BenchResult(name, variant, tracker_on, pct, 0, 0, 0, 0, "build failed")
    mem_mb = USABLE_MEM_MB * pct // 100
    print(f"    needle {pct}%  ({mem_mb} MB)  tracker={'ON' if tracker_on else 'OFF'}")
    if tracker_on:
        Tracker.clear()
    with MemBlocker():
        wall, out, rc, peak = _exec([str(bin_), "-mb", str(mem_mb)],
                               timeout=TIMEOUTS["rodinia_nw"],
                               tracker_on=tracker_on)
    return _make(name, variant, tracker_on, pct, parse_gpu_time(out), wall, out, rc, peak)


def run_polybench_mvt(pct: int, tracker_on: bool, force_build: bool = False) -> BenchResult:
    name, variant = "polybench_mvt", f"mvt_{pct}pct"
    bin_ = build_mvt(force_build)
    if not bin_:
        return BenchResult(name, variant, tracker_on, pct, 0, 0, 0, 0, "build failed")
    mem_mb = USABLE_MEM_MB * pct // 100
    print(f"    mvt    {pct}%  ({mem_mb} MB)  tracker={'ON' if tracker_on else 'OFF'}")
    if tracker_on:
        Tracker.clear()
    with MemBlocker():
        wall, out, rc, peak = _exec([str(bin_), "-mb", str(mem_mb)],
                               timeout=TIMEOUTS["polybench_mvt"],
                               tracker_on=tracker_on)
    return _make(name, variant, tracker_on, pct, parse_gpu_time(out), wall, out, rc, peak)


def run_sgemm(pct: int, tracker_on: bool, force_build: bool = False) -> BenchResult:
    name, variant = "sgemm", f"sgemm_{pct}pct"
    bin_ = build_sgemm(force_build)
    if not bin_:
        return BenchResult(name, variant, tracker_on, pct, 0, 0, 0, 0, "build failed")
    
    mem_mb = USABLE_MEM_MB * pct // 100
    print(f"    sgemm {pct}%  ({mem_mb} MB)  tracker={'ON' if tracker_on else 'OFF'}")
    if tracker_on:
        Tracker.clear() # Reset epoch and bitmaps
    
    with MemBlocker():
        wall, out, rc, peak = _exec([str(bin_), "-mb", str(mem_mb)],
                                    timeout=TIMEOUTS["sgemm"],
                                    tracker_on=tracker_on)
                                    
    return _make(name, variant, tracker_on, pct, parse_gpu_time(out), wall, out, rc, peak)


def run_polybench_2dconv(pct: int, tracker_on: bool, force_build: bool = False) -> BenchResult:
    name, variant = "polybench_2dconv", f"2dconv_{pct}pct"
    bin_ = build_2dconv(force_build)
    if not bin_:
        return BenchResult(name, variant, tracker_on, pct, 0, 0, 0, 0, "build failed")
    mem_mb = USABLE_MEM_MB * pct // 100
    print(f"    2dconv {pct}%  ({mem_mb} MB)  tracker={'ON' if tracker_on else 'OFF'}")
    if tracker_on:
        Tracker.clear()
    with MemBlocker():
        wall, out, rc, peak = _exec([str(bin_), "-mb", str(mem_mb)],
                               timeout=TIMEOUTS["polybench_2dconv"],
                               tracker_on=tracker_on)
    return _make(name, variant, tracker_on, pct, parse_gpu_time(out), wall, out, rc, peak)



# Dispatch table: workload name → runner function
_RUNNERS = {
    "rodinia_nw":       lambda pct, on, fb: run_rodinia_nw(pct, on, fb),
    "polybench_mvt":    lambda pct, on, fb: run_polybench_mvt(pct, on, fb),
    "sgemm":            lambda pct, on, fb: run_sgemm(pct, on, fb),
    "polybench_2dconv": lambda pct, on, fb: run_polybench_2dconv(pct, on, fb),
}


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_phase(workloads: Set[str], tracker_on: bool,
              force_build: bool, runs: int = 1) -> List[BenchResult]:
    results: List[BenchResult] = []
    for wl in sorted(workloads):
        for pct in OVERSUBSCRIPTION.get(wl, [100]):
            for i in range(runs):
                if runs > 1:
                    print(f"    [run {i+1}/{runs}]")
                r = _RUNNERS[wl](pct, tracker_on, force_build)
                tag = "OK" if r.ok() else f"ERR: {(r.error or '')[:60]}"
                print(f"      wall={r.wall_time_s:.1f}s  gpu={r.gpu_runtime_s:.3f}s  [{tag}]")
                results.append(r)
    return results


def _med_std(vals: List[float]) -> str:
    if not vals:
        return "n/a"
    med = statistics.median(vals)
    if len(vals) > 1:
        std = statistics.stdev(vals)
        return f"{med:.3f} ±{std:.3f}"
    return f"{med:.3f}"


def print_summary(results: List[BenchResult]):
    print("\n" + "=" * 72)
    print("SUMMARY  (overhead = (tracker_wall - baseline_wall) / baseline_wall)")
    print("=" * 72)

    by_name: Dict[str, List[BenchResult]] = defaultdict(list)
    for r in results:
        by_name[r.name].append(r)

    for name, runs in sorted(by_name.items()):
        off_runs = [r for r in runs if not r.tracker_on]
        on_runs  = [r for r in runs if r.tracker_on]
        print(f"\n  {name}")
        seen_pcts: Set[int] = set()
        for w in off_runs:
            pct = w.oversub_pct
            if pct in seen_pcts:
                continue
            seen_pcts.add(pct)
            base_group = [r for r in off_runs if r.oversub_pct == pct and r.ok()]
            base_wall_med = statistics.median(r.wall_time_s for r in base_group) if base_group else 0.0
            base_wall_str = _med_std([r.wall_time_s for r in base_group])
            base_gpu_str  = _med_std([r.gpu_runtime_s for r in base_group])
            print(f"    [{w.variant:28s}] baseline : wall={base_wall_str}s  gpu={base_gpu_str}s")
            matched = [t for t in on_runs if t.oversub_pct == pct and t.ok()]
            if matched:
                t_wall_med = statistics.median(r.wall_time_s for r in matched)
                oh = ((t_wall_med - base_wall_med) / base_wall_med * 100
                      if base_wall_med > 0 else float("nan"))
                t_wall_str   = _med_std([r.wall_time_s for r in matched])
                t_gpu_str    = _med_std([r.gpu_runtime_s for r in matched])
                faults_med   = int(statistics.median(r.faults for r in matched))
                faults_std   = statistics.stdev(r.faults for r in matched) if len(matched) > 1 else 0.0
                pages_med    = int(statistics.median(r.unique_pages for r in matched))
                faults_str   = f"{faults_med:,} ±{faults_std:,.0f}" if len(matched) > 1 else f"{faults_med:,}"
                print(f"    [{matched[0].variant:28s}] tracker : wall={t_wall_str}s  gpu={t_gpu_str}s  "
                      f"overhead={oh:+7.1f}%")
                print(f"    {'':28s}           "
                      f"faults={faults_str}  pages={pages_med:,}")


# ── Plotting ──────────────────────────────────────────────────────────────────

_WL_COLORS = {
    "polybench_2dconv": "#4C72B0",
    "polybench_mvt":    "#DD8452",
    "rodinia_nw":       "#55A868",
    "sgemm":            "#C44E52",
}
_BASE_CLR    = "#5B9BD5"
_TRACKER_CLR = "#ED7D31"


def _agg_results(results: List[dict]) -> Dict[Tuple, dict]:
    """Aggregate raw result dicts by (name, oversub_pct, tracker_on) → median/stdev."""
    groups: Dict[Tuple, List[dict]] = defaultdict(list)
    for r in results:
        if r.get("error") is None:
            groups[(r["name"], r["oversub_pct"], r["tracker_on"])].append(r)

    out: Dict[Tuple, dict] = {}
    for key, grp in groups.items():
        def med(f):
            return statistics.median(g[f] for g in grp)
        def std(f):
            return statistics.stdev(g[f] for g in grp) if len(grp) > 1 else 0.0
        out[key] = {
            "variant":    grp[0]["variant"],
            "wall_med":   med("wall_time_s"),
            "wall_std":   std("wall_time_s"),
            "gpu_med":    med("gpu_runtime_s"),
            "gpu_std":    std("gpu_runtime_s"),
            "faults_med": med("faults"),
            "faults_std": std("faults"),
            "pages_med":  med("unique_pages"),
            "n":          len(grp),
        }
    return out


def _plot_args(agg: Dict) -> Tuple:
    """Compute shared arguments used by all figure builders."""
    base_keys: List[Tuple] = sorted(
        [k for k in agg if not k[2]], key=lambda k: (k[0], k[1])
    )
    variants    = [agg[k]["variant"] for k in base_keys]
    workloads   = [k[0]              for k in base_keys]
    xi          = list(range(len(base_keys)))
    bar_colors  = [_WL_COLORS.get(w, "#888888") for w in workloads]
    has_tracker = any((k[0], k[1], True) in agg for k in base_keys)
    return base_keys, variants, workloads, xi, bar_colors, has_tracker


def _build_overhead_fig(agg, base_keys, xi, variants, workloads, bar_colors):
    """Return the wall-time overhead % figure, or None if no tracker data."""
    oh_vals, oh_errs = [], []
    for bk in base_keys:
        tk = (bk[0], bk[1], True)
        b  = agg[bk]
        if tk in agg and b["wall_med"] > 0:
            t   = agg[tk]
            oh  = (t["wall_med"] - b["wall_med"]) / b["wall_med"] * 100
            rel = ((b["wall_std"] / b["wall_med"])**2 +
                   (t["wall_std"] / t["wall_med"])**2)**0.5 if t["wall_med"] > 0 else 0.0
            oh_vals.append(oh)
            oh_errs.append(abs(oh) * rel)
        else:
            oh_vals.append(0.0)
            oh_errs.append(0.0)

    if not any(v != 0 for v in oh_vals):
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(xi) * 1.2), 5))
    ax.bar(xi, oh_vals,
           yerr=oh_errs if any(e > 0 for e in oh_errs) else None,
           color=bar_colors, edgecolor="white", linewidth=0.6,
           error_kw={"elinewidth": 1.2, "capsize": 4, "ecolor": "#444"})
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xi)
    ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Wall-time overhead (%)")
    ax.set_title("UVM Tracker Wall-Time Overhead by Benchmark Variant")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    seen: Dict[str, str] = {}
    for w, c in zip(workloads, bar_colors):
        seen[w] = c
    ax.legend(handles=[Patch(facecolor=c, label=w) for w, c in seen.items()],
              fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def _build_times_fig(agg, base_keys, xi, variants, has_tracker):
    """Return the wall time & GPU time comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(max(9, len(xi) * 1.8), 5))
    for ax, field, title, ylabel in zip(
            axes,
            ["wall", "gpu"],
            ["Wall Time: Baseline vs Tracker", "GPU Kernel Time: Baseline vs Tracker"],
            ["Wall time (s)", "GPU kernel time (s)"],
    ):
        w = 0.35
        b_meds = [agg[bk][f"{field}_med"] for bk in base_keys]
        b_errs = [agg[bk][f"{field}_std"] for bk in base_keys]
        ax.bar([i - w/2 for i in xi], b_meds, width=w, label="Baseline",
               color=_BASE_CLR, edgecolor="white", linewidth=0.5,
               yerr=b_errs if any(e > 0 for e in b_errs) else None,
               error_kw={"elinewidth": 1.0, "capsize": 3, "ecolor": "#333"})
        if has_tracker:
            t_meds = [agg.get((k[0], k[1], True), {}).get(f"{field}_med", 0) for k in base_keys]
            t_errs = [agg.get((k[0], k[1], True), {}).get(f"{field}_std", 0) for k in base_keys]
            ax.bar([i + w/2 for i in xi], t_meds, width=w, label="Tracker",
                   color=_TRACKER_CLR, edgecolor="white", linewidth=0.5,
                   yerr=t_errs if any(e > 0 for e in t_errs) else None,
                   error_kw={"elinewidth": 1.0, "capsize": 3, "ecolor": "#333"})
        ax.set_xticks(xi)
        ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("UVM Tracker Runtime Comparison", fontsize=11)
    fig.tight_layout()
    return fig


def _fig_to_b64(fig) -> str:
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def generate_plots(json_path: Path, plot_dir: Path) -> List[Path]:
    """Load a benchmark results JSON and write PNG plots into plot_dir."""
    if not _MATPLOTLIB:
        print("  [plot] matplotlib not installed — skipping  (pip install matplotlib)")
        return []

    with open(json_path) as f:
        data = json.load(f)
    results = data.get("results", [])
    if not results:
        print("  [plot] JSON contains no results — nothing to plot")
        return []

    agg = _agg_results(results)
    plot_dir.mkdir(parents=True, exist_ok=True)
    base_keys, variants, workloads, xi, bar_colors, has_tracker = _plot_args(agg)
    saved: List[Path] = []

    fig = _build_overhead_fig(agg, base_keys, xi, variants, workloads, bar_colors)
    if fig:
        p = plot_dir / "overhead.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

    fig = _build_times_fig(agg, base_keys, xi, variants, has_tracker)
    p = plot_dir / "times.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    return saved


def generate_html_report(json_path: Path, html_path: Path) -> Path:
    """Generate a self-contained HTML report with embedded plots and a data table."""
    if not _MATPLOTLIB:
        print("  [html] matplotlib not installed — skipping  (pip install matplotlib)")
        return html_path

    with open(json_path) as f:
        data = json.load(f)
    results = data.get("results", [])
    if not results:
        print("  [html] JSON contains no results — nothing to report")
        return html_path

    agg = _agg_results(results)
    base_keys, variants, workloads, xi, bar_colors, has_tracker = _plot_args(agg)

    # Build embedded plot images
    plots_html = ""
    fig = _build_overhead_fig(agg, base_keys, xi, variants, workloads, bar_colors)
    if fig:
        plots_html += (
            '<div class="plot-block">'
            '<p class="plot-title">Wall-Time Overhead (%)</p>'
            f'<img src="data:image/png;base64,{_fig_to_b64(fig)}">'
            '</div>'
        )
        plt.close(fig)

    fig = _build_times_fig(agg, base_keys, xi, variants, has_tracker)
    plots_html += (
        '<div class="plot-block">'
        '<p class="plot-title">Runtime Comparison — Baseline vs Tracker</p>'
        f'<img src="data:image/png;base64,{_fig_to_b64(fig)}">'
        '</div>'
    )
    plt.close(fig)

    # Build results table rows
    def fmt(val, std, decimals=2):
        s = f"{val:.{decimals}f}"
        if std > 0:
            s += f'<span class="std"> ±{std:.{decimals}f}</span>'
        return s

    def oh_badge(oh):
        if   oh < 100: cls = "oh-low"
        elif oh < 200: cls = "oh-mid"
        elif oh < 300: cls = "oh-high"
        else:          cls = "oh-extreme"
        return f'<span class="badge {cls}">{oh:+.1f}%</span>'

    rows = ""
    prev_wl = None
    for bk in base_keys:
        name, pct, _ = bk
        tk = (name, pct, True)
        b  = agg[bk]
        t  = agg.get(tk)

        wl_cell = f'<td class="wl">{name}</td>' if name != prev_wl else '<td class="wl cont"></td>'
        prev_wl = name

        if t:
            oh = (t["wall_med"] - b["wall_med"]) / b["wall_med"] * 100 if b["wall_med"] > 0 else 0.0
            tracker_cells = (
                f'<td class="num">{fmt(t["wall_med"], t["wall_std"])}</td>'
                f'<td class="num">{fmt(t["gpu_med"],  t["gpu_std"], 3)}</td>'
                f'<td class="center">{oh_badge(oh)}</td>'
                f'<td class="num">{fmt(t["faults_med"], t["faults_std"], 0)}</td>'
                f'<td class="num">{int(t["pages_med"]):,}</td>'
            )
        else:
            tracker_cells = '<td colspan="5" class="na">—</td>'

        rows += (
            f'<tr>'
            f'{wl_cell}'
            f'<td class="mono">{b["variant"]}</td>'
            f'<td class="center">{pct}%</td>'
            f'<td class="num">{fmt(b["wall_med"], b["wall_std"])}</td>'
            f'<td class="num">{fmt(b["gpu_med"],  b["gpu_std"], 3)}</td>'
            f'{tracker_cells}'
            f'<td class="center">{b["n"]}</td>'
            f'</tr>\n'
        )

    # Metadata pills
    meta_items = [
        ("Run",      data.get("timestamp", "—")),
        ("Blocker",  f'{data.get("blocker_mb", "?")} MB'),
        ("Usable",   f'{data.get("usable_mem_mb", "?")} MB'),
        ("NVCC",     data.get("nvcc_flags", "—")),
        ("Tracker",  "available" if data.get("tracker_available") else "not available"),
    ]
    meta_html = "".join(
        f'<span class="pill"><span class="pill-key">{k}</span> {v}</span>'
        for k, v in meta_items
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CUDA UVM Benchmark Report</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
     background:#f1f5f9;color:#1e293b;line-height:1.5;font-size:14px}}
.page{{max-width:1280px;margin:0 auto;padding:28px 24px}}

/* header */
header{{background:linear-gradient(135deg,#1e293b 0%,#334155 100%);
        color:#f8fafc;border-radius:10px;padding:24px 28px;margin-bottom:22px;
        box-shadow:0 2px 8px rgba(0,0,0,.18)}}
header h1{{font-size:1.35rem;font-weight:700;letter-spacing:-.01em;margin-bottom:12px}}
.pills{{display:flex;flex-wrap:wrap;gap:8px}}
.pill{{background:rgba(255,255,255,.12);border-radius:20px;padding:3px 12px;
       font-size:.78rem;white-space:nowrap}}
.pill-key{{opacity:.65;margin-right:4px}}

/* cards */
.card{{background:#fff;border-radius:10px;padding:22px 24px;
       margin-bottom:22px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.card h2{{font-size:.95rem;font-weight:700;color:#475569;text-transform:uppercase;
          letter-spacing:.05em;margin-bottom:16px;padding-bottom:10px;
          border-bottom:1px solid #e2e8f0}}

/* table */
.tbl-wrap{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:.84rem}}
th{{background:#f8fafc;text-align:left;padding:9px 12px;font-weight:600;
    color:#64748b;border-bottom:2px solid #e2e8f0;white-space:nowrap}}
td{{padding:8px 12px;border-bottom:1px solid #f1f5f9;vertical-align:middle}}
tr:last-child td{{border-bottom:none}}
tbody tr:hover td{{background:#f8fafc}}
.wl{{font-family:monospace;font-size:.8rem;color:#334155;font-weight:600;
     border-left:3px solid #cbd5e1;padding-left:9px}}
.wl.cont{{border-left-color:transparent}}
.mono{{font-family:monospace;font-size:.82rem}}
.num{{text-align:right;font-variant-numeric:tabular-nums;font-family:monospace}}
.center{{text-align:center}}
.std{{color:#94a3b8;font-size:.85em}}
.na{{text-align:center;color:#94a3b8}}

/* overhead badges */
.badge{{display:inline-block;padding:2px 9px;border-radius:12px;
        font-weight:700;font-size:.78rem;letter-spacing:.01em}}
.oh-low    {{background:#dcfce7;color:#15803d}}
.oh-mid    {{background:#fef9c3;color:#a16207}}
.oh-high   {{background:#ffedd5;color:#c2410c}}
.oh-extreme{{background:#fee2e2;color:#b91c1c}}

/* plots */
.plot-block{{margin-bottom:18px}}
.plot-block:last-child{{margin-bottom:0}}
.plot-title{{font-size:.82rem;font-weight:600;color:#64748b;
             text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px}}
.plot-block img{{width:100%;height:auto;border-radius:6px;
                 border:1px solid #e2e8f0}}
</style>
</head>
<body>
<div class="page">

  <header>
    <h1>CUDA UVM Tracker &mdash; Benchmark Report</h1>
    <div class="pills">{meta_html}</div>
  </header>

  <div class="card">
    <h2>Results Summary</h2>
    <div class="tbl-wrap">
    <table>
      <thead>
        <tr>
          <th>Workload</th>
          <th>Variant</th>
          <th class="center">Oversub</th>
          <th class="num">Base Wall (s)</th>
          <th class="num">Base GPU (s)</th>
          <th class="num">Track Wall (s)</th>
          <th class="num">Track GPU (s)</th>
          <th class="center">Overhead</th>
          <th class="num">Faults</th>
          <th class="num">Unique Pages</th>
          <th class="center">N</th>
        </tr>
      </thead>
      <tbody>
{rows}      </tbody>
    </table>
    </div>
  </div>

  <div class="card">
    <h2>Plots</h2>
    {plots_html}
  </div>

</div>
</body>
</html>"""

    html_path.write_text(html)
    return html_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark CUDA UVM access tracker overhead",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workloads and oversubscription levels (% of 1 GB usable GPU memory):
  rodinia_nw       150%, 200%
  polybench_mvt    150%
  sgemm            100%, 150%, 200%
  polybench_2dconv 100%

Examples:
  python3 benchmark_tracker.py
  python3 benchmark_tracker.py -w rodinia_nw sgemm
  python3 benchmark_tracker.py --rebuild --no-tracker
        """,
    )
    ap.add_argument("-w", "--workloads", nargs="+",
                    choices=list(OVERSUBSCRIPTION), metavar="WL",
                    help="Workloads to run (default: all)")
    ap.add_argument("-o", "--output", default="benchmark_results.json",
                    help="JSON output file (default: benchmark_results.json)")
    ap.add_argument("--rebuild", action="store_true",
                    help="Force recompile all CUDA binaries")
    ap.add_argument("--no-tracker", action="store_true",
                    help="Run baseline phase only (skip tracker phase)")
    ap.add_argument("--blocker-mb", type=int, default=0, metavar="MB",
                    help="MB to block with cudaMalloc (default: auto from nvidia-smi)")
    ap.add_argument("--list", action="store_true",
                    help="List workloads and exit")
    ap.add_argument("--runs", type=int, default=1, metavar="N",
                    help="Repeat each benchmark N times and report median ± stddev (default: 1)")
    ap.add_argument("--plot", action="store_true",
                    help="Generate PNG plots after the run (saved to plots/ next to the JSON)")
    ap.add_argument("--html", action="store_true",
                    help="Generate a self-contained HTML report after the run")
    ap.add_argument("--plot-json", metavar="FILE",
                    help="Load an existing results JSON and generate PNG plots")
    ap.add_argument("--html-json", metavar="FILE",
                    help="Load an existing results JSON and generate an HTML report")
    args = ap.parse_args()

    if args.list:
        print("Workload             Oversubscription levels")
        print("-" * 45)
        for wl, pcts in OVERSUBSCRIPTION.items():
            print(f"  {wl:20s} {pcts}")
        return 0

    if args.plot_json:
        json_path = Path(args.plot_json)
        if not json_path.exists():
            print(f"ERROR: {json_path} not found")
            return 1
        plot_dir = json_path.parent / "plots"
        print(f"Generating plots from {json_path} → {plot_dir}/")
        for p in generate_plots(json_path, plot_dir):
            print(f"  Saved: {p}")
        return 0

    if args.html_json:
        json_path = Path(args.html_json)
        if not json_path.exists():
            print(f"ERROR: {json_path} not found")
            return 1
        html_path = json_path.with_suffix(".html")
        print(f"Generating HTML report from {json_path} → {html_path}")
        generate_html_report(json_path, html_path)
        print(f"  Saved: {html_path}")
        return 0

    # Determine how much GPU memory to block
    _init_gpu_info(override_blocker_mb=args.blocker_mb)

    workloads     = set(args.workloads) if args.workloads else set(OVERSUBSCRIPTION)
    tracker_avail = Tracker.available()

    print("=" * 72)
    print("CUDA ACCESS TRACKER BENCHMARK SUITE")
    print("=" * 72)
    print(f"  Repo root  : {REPO_ROOT}")
    print(f"  NVCC flags : {NVCC_FLAGS}")
    print(f"  GPU free   : {_gpu_free_mb} MB")
    print(f"  Blocker    : {_blocker_mb} MB  "
          f"(leaves ~{_gpu_free_mb - _blocker_mb} MB free = "
          f"~{_gpu_free_mb - _blocker_mb - CONTEXT_OVERHEAD_MB} MB for workloads)")
    print(f"  Usable MB  : {USABLE_MEM_MB} MB  (baseline for oversubscription %)")
    print(f"  Tracker    : {'available at ' + TRACKER_PROCFS if tracker_avail else 'NOT available'}")
    print(f"  Workloads  : {', '.join(sorted(workloads))}")
    print()

    # Build mem_blocker.out once up front
    MemBlocker.build(force=args.rebuild)

    all_results: List[BenchResult] = []

    # ── Phase 1: baseline (tracker OFF) ────────────────────────────────────
    print("-" * 72)
    print(f"PHASE 1 — tracker OFF  (baseline, {args.runs} run(s) each)")
    print("-" * 72)
    try:
        all_results += run_phase(workloads, tracker_on=False,
                                 force_build=args.rebuild, runs=args.runs)
    except KeyboardInterrupt:
        print("\n  Interrupted during Phase 1")

    # ── Phase 2: with tracker ───────────────────────────────────────────────
    if not args.no_tracker:
        if not tracker_avail:
            print(f"\nWARNING: {TRACKER_PROCFS} not found — skipping tracker phase")
            print("         Check that your modified UVM driver is loaded.")
        else:
            print()
            print("-" * 72)
            print(f"PHASE 2 — tracker ON  ({args.runs} run(s) each)")
            print("-" * 72)
            Tracker.enable()
            try:
                all_results += run_phase(workloads, tracker_on=True,
                                         force_build=False, runs=args.runs)
            except KeyboardInterrupt:
                print("\n  Interrupted during Phase 2")
            finally:
                Tracker.disable()

    # ── Save results ────────────────────────────────────────────────────────
    out_path = REPO_ROOT / args.output
    with open(out_path, "w") as f:
        json.dump({
            "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
            "tracker_available": tracker_avail,
            "usable_mem_mb":   USABLE_MEM_MB,
            "blocker_mb":      _blocker_mb,
            "nvcc_flags":      NVCC_FLAGS,
            "results":         [asdict(r) for r in all_results],
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if args.plot:
        plot_dir = out_path.parent / "plots"
        print(f"Generating plots → {plot_dir}/")
        for p in generate_plots(out_path, plot_dir):
            print(f"  Saved: {p}")

    if args.html:
        html_path = out_path.with_suffix(".html")
        print(f"Generating HTML report → {html_path}")
        generate_html_report(out_path, html_path)
        print(f"  Saved: {html_path}")

    print_summary(all_results)

    failed = [r for r in all_results if not r.ok()]
    print(f"\nTotal: {len(all_results)}  Failed: {len(failed)}")
    if failed:
        print("Failed workloads:")
        for r in failed:
            print(f"  {r.variant:35s} {r.error}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
