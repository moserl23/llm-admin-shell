#!/usr/bin/env python3
"""Analyze timing structure in audit logs and compare actor-specific distributions.

The script groups audit records into execution bundles and short time-window
clusters, then visualizes inter-cluster delays either globally or conditioned
on a command.

Examples
--------
All inter-event delays for 3 humans and 2 AIs from Nextcloud:

    python -m src.runners.stats.audit_timing_plot \
        --dataset Nextcloud \
        --series all \
        --n_humans 3 \
        --n_ais 2

Command-conditioned delays after `tail` for 2 humans and 2 AIs from WordPress:

    python -m src.runners.stats.audit_timing_plot \
        --dataset WordPress \
        --series cmd \
        --cmd tail \
        --n_humans 2 \
        --n_ais 2 \
        --cluster_window 0.5 \
        --bins_n 50 \
        --save_dir results
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence, Any

import matplotlib.pyplot as plt
import numpy as np

from src.core.stats.data_catalog import get_log_path, analysis_actors


AUDIT_ID_RE = re.compile(r"msg=audit\((?P<ts>\d+(?:\.\d+)?):(?P<serial>\d+)\)")
FIELD_RE_TEMPLATE = r"{key}=(?P<val>\"[^\"]*\"|\S+)"


# ---- CLI configuration ----

def parse_args() -> argparse.Namespace:
    """Parse and validate CLI arguments for actor-wise timing analysis."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot actor-wise timing distributions from audit logs. "
            "You can analyze either all inter-cluster delays or only delays "
            "following clusters that contain a specific command."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=["Nextcloud", "WordPress"],
        required=True,
        help=(
            "Dataset to use. "
            "'Nextcloud' uses the Nextcloud dataset, while 'WordPress' uses the WordPress dataset."
        ),
    )

    parser.add_argument(
        "--series",
        choices=["all", "cmd"],
        required=True,
        help=(
            "Which timing series to plot. "
            "'all' plots all inter-cluster delays. "
            "'cmd' plots only delays from clusters containing --cmd to the next cluster."
        ),
    )

    parser.add_argument(
        "--cmd",
        type=str,
        default=None,
        help=(
            "Command to condition on when --series cmd is selected. "
            "Examples: tail, grep, vim. "
            "Ignored when --series all is used."
        ),
    )

    parser.add_argument(
        "--cluster_window",
        type=float,
        default=0.5,
        help=(
            "Maximum time gap in seconds for merging nearby bundles into one cluster. "
            "Default: 0.5"
        ),
    )

    parser.add_argument(
        "--bins_n",
        type=int,
        default=50,
        help=(
            "Number of logarithmic histogram bins to use. "
            "Default: 50"
        ),
    )

    parser.add_argument(
        "--n_humans",
        type=int,
        default=None,
        help=(
            "How many human actors to include in the plots. "
            "Actors are taken in the default dataset actor order returned by analysis_actors(dataset). "
            "If omitted, no human actors are included unless you explicitly pass a value."
        ),
    )

    parser.add_argument(
        "--n_ais",
        type=int,
        default=None,
        help=(
            "How many AI actors to include in the plots. "
            "Actors are taken in the default dataset actor order returned by analysis_actors(dataset). "
            "If omitted, no AI actors are included unless you explicitly pass a value."
        ),
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="results",
        help=(
            "Directory where the generated plots should be saved. "
            "Default: results"
        ),
    )

    args = parser.parse_args()

    if args.series == "cmd" and not args.cmd:
        parser.error("When --series cmd is used, --cmd is required")

    if args.cluster_window <= 0:
        parser.error("--cluster_window must be > 0")

    if args.bins_n <= 0:
        parser.error("--bins_n must be a positive integer")

    if args.n_humans is not None and args.n_humans < 0:
        parser.error("--n_humans must be >= 0")

    if args.n_ais is not None and args.n_ais < 0:
        parser.error("--n_ais must be >= 0")

    if args.n_humans is None and args.n_ais is None:
        parser.error("At least one of --n_humans or --n_ais must be provided")

    return args


def is_ai_actor(label: str) -> bool:
    """Return whether an actor label should be treated as an AI actor."""
    return "gpt" in label.lower()


def anonymize_actor_labels(actor_names: Sequence[str]) -> Dict[str, str]:
    """Map actor names to display labels for publication-friendly plots.

    Human participants are anonymized sequentially, while GPT-labelled actors
    remain identifiable to preserve the comparison of interest.
    """
    mapping: Dict[str, str] = {}
    human_idx = 1

    for name in actor_names:
        if is_ai_actor(name):
            mapping[name] = name
        else:
            mapping[name] = f"Human {human_idx}"
            human_idx += 1

    return mapping


def select_actors(
    file_pairs: Sequence[Tuple[str, str]],
    *,
    include_humans: Optional[int] = None,
    include_ais: Optional[int] = None,
    specific_actors: Optional[Sequence[str]] = None,
) -> List[Tuple[str, str]]:
    """Select actors either explicitly or by human/AI quotas.

    When counts are used, actors are taken in the input order so selection stays
    aligned with the dataset's canonical actor ordering.
    """
    if specific_actors is not None:
        wanted = set(specific_actors)
        return [(label, path) for label, path in file_pairs if label in wanted]

    humans = [(label, path) for label, path in file_pairs if not is_ai_actor(label)]
    ais = [(label, path) for label, path in file_pairs if is_ai_actor(label)]

    selected: List[Tuple[str, str]] = []

    if include_humans is not None:
        selected.extend(humans[:include_humans])

    if include_ais is not None:
        selected.extend(ais[:include_ais])

    return selected


# ---- Audit parsing and bundle construction ----

def extract_audit_id(line: str) -> Optional[Tuple[float, int]]:
    """Extract the audit timestamp and serial number from a raw log line.

    These two fields uniquely identify one audit event bundle. Returns `None`
    when the line does not contain a parseable audit identifier.
    """
    m = AUDIT_ID_RE.search(line)
    if not m:
        return None
    return float(m.group("ts")), int(m.group("serial"))


def extract_field_from_line(line: str, key: str) -> Optional[str]:
    """Read a single key-value field from an audit log line.

    Quoted values are unwrapped so downstream comparisons can treat quoted and
    unquoted fields uniformly. Returns `None` if the field is absent.
    """
    pattern = re.compile(FIELD_RE_TEMPLATE.format(key=re.escape(key)))
    m = pattern.search(line)
    if not m:
        return None

    val = m.group("val")
    if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
        val = val[1:-1]
    return val


@dataclass
class Bundle:
    """One reconstructed audit event bundle."""
    ts: float
    serial: int
    lines: List[str] = field(default_factory=list)


def read_bundles(path: str) -> List[Bundle]:
    """Group raw audit lines into bundles keyed by audit timestamp and serial.

    Audit events are emitted across multiple lines, so analysis operates on the
    reconstructed bundle rather than on individual records.
    """
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


def has_execve(bundle: Bundle) -> bool:
    """Return whether the bundle contains an EXECVE record."""
    return any("type=EXECVE" in line for line in bundle.lines)


def syscall_line(bundle: Bundle) -> Optional[str]:
    """Return the SYSCALL line for a bundle when present."""
    for line in bundle.lines:
        if "type=SYSCALL" in line:
            return line
    return None


def get_tty_exe_comm(bundle: Bundle) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract basic execution context from the bundle's SYSCALL record.

    The tuple contains `(tty, exe, comm)` and is used to distinguish
    interactive command execution from background or system activity.
    """
    sl = syscall_line(bundle)
    if sl is None:
        return None, None, None

    tty = extract_field_from_line(sl, "tty")
    exe = extract_field_from_line(sl, "exe")
    comm = extract_field_from_line(sl, "comm")
    return tty, exe, comm


def bundle_mentions_cmd(bundle: Bundle, cmd: str) -> bool:
    """Check whether a bundle corresponds to the requested command.

    Matching is intentionally permissive across `comm`, `exe`, and `EXECVE a0`
    because audit records are not fully consistent across environments.
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


def filter_bundles(bundles: List[Bundle]) -> List[Bundle]:
    """Keep only interactive command bundles relevant for timing analysis.

    Bundles without `EXECVE` or without an attached TTY are excluded to avoid
    mixing user-driven activity with background audit noise.
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


def cluster_bundles(
    bundles: List[Bundle],
    cluster_window: float = 0.5,
) -> List[List[Bundle]]:
    """Merge nearby bundles into temporal clusters.

    The fixed window treats short bursts of related audit activity as one event,
    which stabilizes the timing distribution against audit line granularity.
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
    """Compute consecutive delays between ordered timestamps."""
    return [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]


def cmd_to_next_cluster_deltas(clusters: List[List[Bundle]], cmd: str) -> List[float]:
    """Measure time from command-containing clusters to the next cluster start.

    This isolates the delay immediately following one command of interest rather
    than the full inter-event distribution.
    """
    if len(clusters) < 2:
        return []

    starts = [c[0].ts for c in clusters]

    out: List[float] = []
    for i, c in enumerate(clusters[:-1]):
        if any(bundle_mentions_cmd(b, cmd) for b in c):
            out.append(starts[i + 1] - starts[i])

    return out


def analyze_file(
    path: str,
    cluster_window: float = 0.5,
    cmd: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full timing analysis pipeline for one audit log.

    The returned structure contains metadata, all inter-cluster delays, and an
    optional command-conditioned delay series for downstream plotting.
    """
    bundles = read_bundles(path)
    kept = filter_bundles(bundles)
    clusters = cluster_bundles(kept, cluster_window=cluster_window)

    cluster_starts = [c[0].ts for c in clusters]
    deltas = inter_event_deltas(cluster_starts)

    cmd_deltas: List[float] = []
    if cmd is not None:
        cmd_deltas = cmd_to_next_cluster_deltas(clusters, cmd)

    return {
        "meta": {
            "path": path,
            "cluster_window": cluster_window,
            "total_bundles": len(bundles),
            "kept_bundles": len(kept),
            "num_clusters": len(clusters),
        },
        "all": {
            "values": deltas,
        },
        "cmd": {
            "name": cmd,
            "values": cmd_deltas,
        } if cmd is not None else None,
    }


# ---- Plotting ----

def compute_global_log_xlim(
    results: List[Dict[str, Any]],
    *,
    series: str = "all",
) -> Optional[Tuple[float, float]]:
    """Compute shared log-scale x-limits across multiple result sets.

    A common axis range makes cross-actor histograms visually comparable.
    Returns `None` when no positive finite values are available.
    """
    vals: List[float] = []

    for res in results:
        if series == "all":
            values = res.get("all", {}).get("values", [])
        elif series == "cmd":
            cmd_block = res.get("cmd")
            values = cmd_block.get("values", []) if cmd_block is not None else []
        else:
            raise ValueError(f"Unknown series: {series}")

        vals.extend(v for v in values if np.isfinite(v) and v > 0)

    if not vals:
        return None

    vmin = min(vals)
    vmax = max(vals)

    if vmin == vmax:
        vmin = max(vmin * 0.9, 1e-12)
        vmax = vmax * 1.1

    return (vmin, vmax)


def plot_log_hist(
    values: List[float],
    *,
    label: str,
    xlabel: str,
    bins_n: int = 50,
    xlim: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot a density histogram on a logarithmic time axis.

    Non-positive and non-finite values are discarded because the visualization
    is defined on log-scaled delays only.
    """
    values = [v for v in values if np.isfinite(v) and v > 0]
    if not values:
        print(f"[WARN] No positive finite values to plot for: {label}")
        return

    if xlim is not None:
        vmin, vmax = xlim
    else:
        vmin = min(values)
        vmax = max(values)
        if vmin == vmax:
            vmin = max(vmin * 0.9, 1e-12)
            vmax = vmax * 1.1

    bins = np.logspace(np.log10(vmin), np.log10(vmax), bins_n)

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, density=True)
    plt.xscale("log")
    plt.xlim(vmin, vmax)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(label)
    plt.tight_layout()

    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_obj)

    plt.show()


def sanitize_filename_component(s: str) -> str:
    """Convert a free-form label into a filesystem-friendly filename component."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def plot_selected_actors(
    selected_file_pairs: List[Tuple[str, str]],
    *,
    cluster_window: float,
    cmd: Optional[str],
    series: str = "all",
    bins_n: int = 50,
    save_dir: Optional[str] = None,
) -> None:
    """Plot per-actor timing histograms with shared scaling.

    Each selected log is analyzed independently, but the plots reuse a common
    x-range so actor-specific distributions can be compared directly.
    """
    if not selected_file_pairs:
        print("No actor files selected for plotting.")
        return

    selected_results = [
        analyze_file(path, cluster_window=cluster_window, cmd=cmd)
        for _, path in selected_file_pairs
    ]

    shared_xlim = compute_global_log_xlim(selected_results, series=series)

    actor_names = [label for label, _ in selected_file_pairs]
    display_names = anonymize_actor_labels(actor_names)

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for (label, _), result in zip(selected_file_pairs, selected_results):
        display_label = display_names[label]
        safe_actor = sanitize_filename_component(display_label)

        if series == "all":
            values = result["all"]["values"]
            xlabel = "Inter-event time (seconds, log scale)"
            title = f"{display_label}"
            save_path = (
                str(Path(save_dir) / f"{safe_actor}_all.pdf")
                if save_dir else None
            )
        elif series == "cmd":
            cmd_block = result.get("cmd")
            values = cmd_block["values"] if cmd_block is not None else []
            xlabel = f"Time after {cmd} to next cluster (seconds, log scale)"
            title = f"{display_label} - {cmd} to next cluster"
            safe_cmd = sanitize_filename_component(cmd or "cmd")
            save_path = (
                str(Path(save_dir) / f"{safe_actor}_{safe_cmd}.pdf")
                if save_dir else None
            )
        else:
            raise ValueError(f"Unknown series: {series}")

        plot_log_hist(
            values,
            label=title,
            xlabel=xlabel,
            bins_n=bins_n,
            xlim=shared_xlim,
            save_path=save_path,
        )


# ---- Script entry point ----

if __name__ == "__main__":
    args = parse_args()

    file_pairs = [
        (actor, str(get_log_path(actor, "audit", dataset=args.dataset)))
        for actor in analysis_actors(args.dataset)
    ]

    selected_file_pairs = select_actors(
        file_pairs,
        specific_actors=None,
        include_humans=args.n_humans,
        include_ais=args.n_ais,
    )

    plot_selected_actors(
        selected_file_pairs,
        cluster_window=args.cluster_window,
        cmd=args.cmd,
        series=args.series,
        bins_n=args.bins_n,
        save_dir=args.save_dir,
    )