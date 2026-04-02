#!/usr/bin/env python3
"""
audit_human_execve.py

Reads a Linux audit log file, groups lines into "bundles" (events) by msg=audit(ts:serial),
then filters to keep only bundles that:
  1) contain an EXECVE record
  2) have tty != "(none)" in the SYSCALL record (interactive)

Then:
  - clusters bundles by time (collapse bursts)
  - computes inter-cluster inter-event times
  - plots histogram (log bins + density)
  - additionally: finds clusters that contain a cmd execution and collects
    the time-to-next-cluster (cmd->next) deltas, prints stats, and plots histogram.

Usage:
  python audit_human_execve.py /path/to/audit.log

Optional env vars:
  CLUSTER_WINDOW=0.5   # seconds
"""

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Literal

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# Global
# -----------------------------

Metric = Literal["median", "average"]
Series = Literal["all", "cmd"]

### Regex helpers
# Example: msg=audit(1765984141.131:2375):
AUDIT_ID_RE = re.compile(r"msg=audit\((?P<ts>\d+(?:\.\d+)?):(?P<serial>\d+)\)")

# field pattern: key=value where value is either "quoted string" or a bare token
FIELD_RE_TEMPLATE = r"{key}=(?P<val>\"[^\"]*\"|\S+)"


# -----------------------------
# Functions
# -----------------------------

def extract_audit_id(line: str) -> Optional[Tuple[float, int]]:
    """Return (timestamp, serial) if line contains msg=audit(...:serial), else None."""
    m = AUDIT_ID_RE.search(line)
    if not m:
        return None
    return float(m.group("ts")), int(m.group("serial"))


def extract_field_from_line(line: str, key: str) -> Optional[str]:
    """Extract key=value from a single audit line (handles quoted strings)."""
    pattern = re.compile(FIELD_RE_TEMPLATE.format(key=re.escape(key)))
    m = pattern.search(line)
    if not m:
        return None
    val = m.group("val")
    if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
        val = val[1:-1]
    return val


# -----------------------------
# Bundle model
# -----------------------------

@dataclass
class Bundle:
    ts: float
    serial: int
    lines: List[str] = field(default_factory=list)


def read_bundles(path: str) -> List[Bundle]:
    bundles: Dict[Tuple[float, int], Bundle] = {}

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            info = extract_audit_id(line)
            if info is None:
                continue
            ts, serial = info
            key = (ts, serial)
            if key not in bundles:
                bundles[key] = Bundle(ts=ts, serial=serial, lines=[])
            bundles[key].lines.append(line)

    return sorted(bundles.values(), key=lambda b: (b.ts, b.serial))


# -----------------------------
# Bundle inspection
# -----------------------------

def has_execve(bundle: Bundle) -> bool:
    """True if bundle contains an EXECVE record."""
    return any("type=EXECVE" in line for line in bundle.lines)


def syscall_line(bundle: Bundle) -> Optional[str]:
    """
    Return the first SYSCALL line in the bundle, if any.
    If your bundles sometimes contain multiple SYSCALL lines and you want the execve one,
    switch this to check syscall=59 or SYSCALL=execve.
    """
    for line in bundle.lines:
        if "type=SYSCALL" in line:
            return line
    return None


def get_tty_exe_comm(bundle: Bundle) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract tty, exe, comm from the bundle's SYSCALL line (if present)."""
    sl = syscall_line(bundle)
    if sl is None:
        return None, None, None
    tty = extract_field_from_line(sl, "tty")
    exe = extract_field_from_line(sl, "exe")
    comm = extract_field_from_line(sl, "comm")
    return tty, exe, comm


def bundle_mentions_cmd(bundle: Bundle, cmd: str) -> bool:
    """
    Detect whether this bundle corresponds to running `cmd` (e.g. "grep", "tail").

    Checks:
      - SYSCALL comm
      - SYSCALL exe basename
      - EXECVE a0
    """
    cmd = os.path.basename(cmd)

    _, exe, comm = get_tty_exe_comm(bundle)

    if comm == cmd:
        return True
    if exe and os.path.basename(exe) == cmd:
        return True

    for line in bundle.lines:
        if line.startswith("type=EXECVE"):
            a0 = extract_field_from_line(line, "a0")
            if a0 == cmd or (a0 and os.path.basename(a0) == cmd):
                return True

    return False



# -----------------------------
# Filtering logic
# -----------------------------

def filter_bundles(bundles: List[Bundle]) -> List[Bundle]:
    """
    Keep only bundles that:
      1) have EXECVE
      2) have tty != "(none)" (interactive)
    """
    kept: List[Bundle] = []
    for b in bundles:
        if not has_execve(b):
            continue

        tty, _, _ = get_tty_exe_comm(b)

        if tty is None:
            continue
        if tty == "(none)":
            continue

        kept.append(b)

    return kept


# -----------------------------
# Clustering
# -----------------------------

def cluster_bundles(bundles: List[Bundle], cluster_window: float = 0.5) -> List[List[Bundle]]:
    """
    Cluster bundles by time proximity. Each cluster is a list of bundles.
    """
    if not bundles:
        return []
    bundles = sorted(bundles, key=lambda b: (b.ts, b.serial))

    clusters: List[List[Bundle]] = []
    current: List[Bundle] = [bundles[0]]
    last_ts = bundles[0].ts

    for b in bundles[1:]:
        if (b.ts - last_ts) <= cluster_window:
            current.append(b)
            last_ts = b.ts
        else:
            clusters.append(current)
            current = [b]
            last_ts = b.ts

    clusters.append(current)
    return clusters


def inter_event_deltas(timestamps: List[float]) -> List[float]:
    """Compute successive differences between sorted timestamps."""
    return [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]


def cmd_to_next_cluster_deltas(clusters: List[List[Bundle]], cmd: str) -> List[float]:
    """
    For each cluster that contains `cmd`, return delta to the NEXT cluster start time.
    Delta is (next_cluster_start - this_cluster_start).
    """
    if len(clusters) < 2:
        return []

    starts = [c[0].ts for c in clusters]

    out: List[float] = []
    for i, c in enumerate(clusters[:-1]):
        if any(bundle_mentions_cmd(b, cmd) for b in c):
            out.append(starts[i + 1] - starts[i])

    return out



# -----------------------------
# Plot helpers
# -----------------------------

def plot_log_hist(values: List[float], title: str, xlabel: str, bins_n: int = 50) -> None:
    values = [v for v in values if np.isfinite(v) and v > 0]
    if not values:
        print(f"[WARN] No positive finite values to plot for: {title}")
        return

    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        # avoid logspace issues
        vmin = max(vmin * 0.9, 1e-12)
        vmax = vmax * 1.1

    bins = np.logspace(np.log10(vmin), np.log10(vmax), bins_n)

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, density=True)
    plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def basic_stat(values: List[float]) -> Dict[str, float]:
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return {}

    v = np.array(sorted(values), dtype=float)

    stats = {
        "min": float(v[0]),
        "median": float(np.median(v)),
        "max": float(v[-1]),
        "average": float(np.mean(v)),
        # deviation/dispersion:
        "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,   # sample std dev
        "count": float(len(v)),
        # optional robust spread:
        "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)) if len(v) > 1 else 0.0,
    }
    return stats


def analyze_file(path: str, cluster_window: float = 0.5, print_plot_flag: bool = False, cmd: Optional[str] = None) -> Dict[str, Any]:
    bundles = read_bundles(path)
    kept = filter_bundles(bundles)
    clusters = cluster_bundles(kept, cluster_window=cluster_window)

    if print_plot_flag:
        print(f"Total bundles parsed: {len(bundles)}")
        print(f"Bundles after filters (EXECVE + tty != (none)): {len(kept)}")
        print(f"After clustering (window={cluster_window:.3f}s): {len(clusters)}")

    # Overall inter-event times (cluster start -> next cluster start)
    cluster_starts = [c[0].ts for c in clusters]
    deltas = inter_event_deltas(cluster_starts)
    all_stats = basic_stat(deltas)

    if print_plot_flag:
        # print stats if available
        if all_stats:
            print("\nInter-event deltas (cluster->cluster)")
            print(f"  n     = {len([v for v in deltas if np.isfinite(v)])}")
            print(f"  min   = {all_stats['min']:.6f} s")
            print(f"  median= {all_stats['median']:.6f} s")
            print(f"  max   = {all_stats['max']:.6f} s")
            print(f"  avg   = {all_stats['average']:.6f} s")

        plot_log_hist(
            deltas,
            title="Histogram of inter-event times (cluster->cluster)",
            xlabel="Inter-event time (seconds, log scale)",
            bins_n=50,
        )

    # Grep -> next cluster deltas
    # cmd -> next cluster deltas (optional)
    cmd_deltas: List[float] = []
    cmd_stats: Dict[str, float] = {}

    if cmd is not None:
        cmd_deltas = cmd_to_next_cluster_deltas(clusters, cmd)
        cmd_stats = basic_stat(cmd_deltas)

        if print_plot_flag:
            if cmd_stats:
                print(f"\n{cmd}->next-cluster deltas")
                print(f"  n     = {len([v for v in cmd_deltas if np.isfinite(v)])}")
                print(f"  min   = {cmd_stats['min']:.6f} s")
                print(f"  median= {cmd_stats['median']:.6f} s")
                print(f"  max   = {cmd_stats['max']:.6f} s")
                print(f"  avg   = {cmd_stats['average']:.6f} s")

            if cmd_deltas:
                plot_log_hist(
                    cmd_deltas,
                    title=f"Histogram: {cmd} cluster -> next cluster time",
                    xlabel=f"Time after {cmd} to next action (seconds, log scale)",
                    bins_n=40,
                )

                print(f"\nFirst 20 {cmd}->next deltas:")
                for i, d in enumerate(cmd_deltas[:20]):
                    print(f"  {i:02d}: {d:.6f} s")
    return {
        "meta": {
            "path": path,
            "cluster_window": cluster_window,
            "total_bundles": len(bundles),
            "kept_bundles": len(kept),
            "num_clusters": len(clusters),
        },
        "all": {
            "n": len([v for v in deltas if np.isfinite(v)]),
            "stats": all_stats,
        },
        "cmd": {
            "name": cmd,
            "n": len([v for v in cmd_deltas if np.isfinite(v)]),
            "stats": cmd_stats,
        } if cmd is not None else None,
    }


def compare_files_plot(
    paths: List[str],
    labels: List[str],
    *,
    cluster_window: float = 0.5,
    metric: Metric = "median",
    series: Series = "all",
    cmd: Optional[str] = None,
    title: Optional[str] = None,
    dev_metric: str = "std",   # NEW: "std" or "iqr" or ...
) -> List[Dict[str, Any]]:
    ...
    results: List[Dict[str, Any]] = []
    y: List[float] = []
    yerr: List[float] = []     # NEW

    for p in paths:
        res = analyze_file(p, cluster_window=cluster_window, print_plot_flag=False, cmd=cmd)
        results.append(res)

        if series == "all":
            stats = res.get("all", {}).get("stats", {})
        elif series == "cmd":
            cmd_block = res.get("cmd") or {}
            stats = cmd_block.get("stats", {})
        else:
            stats = {}

        y.append(stats.get(metric, float("nan")))
        yerr.append(stats.get(dev_metric, float("nan")))   # NEW

    plt.figure(figsize=(max(6, 0.8 * len(labels)), 4))
    plt.bar(labels, y, yerr=yerr, capsize=4)  # NEW: yerr + capsize
    ylabel_series = "all" if series == "all" else cmd
    plt.ylabel(f"{ylabel_series}.{metric} (seconds)")
    plt.xlabel("File")
    series_label = "all" if series == "all" else (cmd or "cmd")
    plt.title(title or f"{series_label} {metric} by file (cluster_window={cluster_window}s)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    return results



# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    cluster_window = 2

    paths = [
        "/home/lorenz/Documents/llm-admin-shell/ExperimentResult/combine/ExperimentAggregated/Armin/audit.log",
        "/home/lorenz/Documents/llm-admin-shell/ExperimentResult/combine/ExperimentAggregated/Benni/audit.log",
        "/home/lorenz/Documents/llm-admin-shell/ExperimentResult/combine/ExperimentAggregated/Hotti/audit.log",
        "/home/lorenz/Documents/llm-admin-shell/ExperimentResult/combine/ExperimentAggregated/Marvin/audit.log",
        "/home/lorenz/Documents/llm-admin-shell/ExperimentResult/combine/ExperimentAggregated/Nico/audit.log",
        "/home/lorenz/Documents/llm-admin-shell/ExperimentResult/combine/ExperimentAggregated/Torina/audit.log",
        "/home/lorenz/Documents/llm-admin-shell/ExperimentResult/combine/ExperimentAggregated/GPT4.1/audit.log",
        "/home/lorenz/Documents/llm-admin-shell/ExperimentResult/combine/ExperimentAggregated/GPT4.1_V2/audit.log",
        "/home/lorenz/Documents/llm-admin-shell/ExperimentResult/combine/ExperimentAggregated/GPT4o/audit.log",
        "/home/lorenz/Documents/llm-admin-shell/ExperimentResult/combine/ExperimentAggregated/GPT5/audit.log",
    ]
    labels = ["Armin", "Benni", "Hotti", "Marvin", "Nico", "Torina", "GPT4.1", "GPT4.1_V2", "GPT4o", "GPT5"]

    # interesting commands: grep, rg, ripgrep, sed, awk, less, more, most, ls, tree, stat, file, vim, cd, rm, mv, cp, chmod, chown, curl, wget, scp

    cmd = "grep"
    cmd = "tail"
    cmd = "vim"
    cmd = "chmod"
    cmd = "ls"

    compare_files_plot(
        paths,
        labels,
        cluster_window=cluster_window,
        metric="median",
        series="cmd",
        cmd=cmd,
        dev_metric="iqr",  # <- standard deviation error bars
        title=f"command latency comparison ({cmd})",
    )
