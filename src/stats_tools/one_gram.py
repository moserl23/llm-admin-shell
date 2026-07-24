"""Utilities for 1-gram distribution analysis across actor log files.

The module compares unigram frequencies at the word or character level and
reports pairwise divergence scores for downstream statistical analysis.
"""

from pathlib import Path
from itertools import combinations
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

from src.core.stats.data_catalog import analysis_actors, get_log_path


# ---- Tokenization and counting ----

def load_lines(file_path: str) -> list[str]:
    """Load a text file as lines with permissive UTF-8 decoding.

    Decoding errors are replaced rather than raised so analysis can proceed on
    imperfect logs. Returns the file contents split into lines.
    """
    return Path(file_path).read_text(encoding="utf-8", errors="replace").splitlines()


def tokenize(lines: list[str], mode: str = "word") -> list[str]:
    """Convert lines into word- or character-level tokens.

    Character mode preserves line boundaries via newline insertion, while word
    mode uses simple whitespace splitting. Returns a flat token sequence.
    """
    if mode == "char":
        return list("\n".join(lines))
    elif mode == "word":
        tokens = []
        for line in lines:
            tokens.extend(line.split())
        return tokens
    else:
        raise ValueError("mode must be 'word' or 'char'")


def get_token_counts(file_path: str, mode: str = "word") -> Counter:
    """Count 1-gram occurrences in a file for the requested tokenization mode.

    The returned counter is the basic representation used by all downstream
    comparison and plotting utilities.
    """
    lines = load_lines(file_path)
    toks = tokenize(lines, mode=mode)
    return Counter(toks)


def counter_to_probs(cnt: Counter, vocab: list[str]) -> np.ndarray:
    """Project counts onto a shared vocabulary and normalize to probabilities.

    Returns a probability vector aligned with `vocab`; empty counters yield a
    zero vector so downstream callers can handle degenerate cases explicitly.
    """
    arr = np.array([cnt[tok] for tok in vocab], dtype=float)
    total = arr.sum()
    if total == 0:
        return np.zeros_like(arr, dtype=float)
    return arr / total


def js_score_from_counters(cnt1: Counter, cnt2: Counter, vocab: list[str]) -> float:
    """Compute Jensen-Shannon distance between two counters on a shared vocabulary.

    If either projected distribution is empty, the score is reported as `nan`
    rather than forcing a misleading finite value.
    """
    p = counter_to_probs(cnt1, vocab)
    q = counter_to_probs(cnt2, vocab)

    if p.sum() == 0 or q.sum() == 0:
        return float("nan")

    return float(jensenshannon(p, q))


def compute_diff_from_counters(
    cnt1: Counter,
    cnt2: Counter,
    *,
    top_k: int = 20,
    min_count: int = 1,
):
    """Summarize the largest unigram frequency differences between two counters.

    Tokens are filtered on pooled count so rare events can be suppressed
    consistently across both sources. Returns the top rows plus L1 and JS scores.
    """
    total1 = sum(cnt1.values())
    total2 = sum(cnt2.values())

    if total1 == 0 or total2 == 0:
        raise RuntimeError("One file has no tokens.")

    # Filtering on pooled frequency avoids keeping tokens that appear only once
    # in one file and would otherwise dominate the ranking by noise.
    pooled = cnt1 + cnt2
    vocab = [tok for tok, c in pooled.items() if c >= min_count]

    p = counter_to_probs(cnt1, vocab)
    q = counter_to_probs(cnt2, vocab)
    diff = p - q

    rows = []
    for tok, p1, p2, delta in zip(vocab, p, q, diff):
        rows.append((tok, cnt1[tok], cnt2[tok], p1, p2, delta))

    l1_score = float(np.sum(np.abs(diff)))
    js_score = js_score_from_counters(cnt1, cnt2, vocab)

    rows.sort(key=lambda x: abs(x[5]), reverse=True)
    return rows[:top_k], l1_score, js_score


def compute_diff(
    file1: str,
    file2: str,
    mode: str = "word",
    top_k: int = 20,
    min_count: int = 1,
):
    """Load two files and compute their top unigram differences.

    This is a convenience wrapper around the counter-based implementation and
    returns the same `(rows, l1_score, js_score)` tuple.
    """
    cnt1 = get_token_counts(file1, mode=mode)
    cnt2 = get_token_counts(file2, mode=mode)
    return compute_diff_from_counters(cnt1, cnt2, top_k=top_k, min_count=min_count)


# ---- Visualization ----

def plot_diff(rows, title: str = "Top 1-gram frequency differences"):
    """Plot signed relative-frequency differences for the selected tokens.

    Positive values indicate higher relative frequency in the first file and
    negative values indicate enrichment in the second file.
    """
    if not rows:
        print("No differences to plot.")
        return

    labels = [r[0] for r in rows][::-1]
    values = [r[5] for r in rows][::-1]

    plt.figure(figsize=(10, max(4, 0.4 * len(labels))))
    bars = plt.barh(labels, values)
    plt.axvline(0)

    plt.xlabel("Relative frequency difference (file1 - file2)")
    plt.title(title)

    max_abs = max(abs(v) for v in values) if values else 0.001
    offset = max(0.001, 0.02 * max_abs)

    for bar, v in zip(bars, values):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2

        if x >= 0:
            plt.text(x + offset, y, f"{v:+.4f}", va="center", ha="left")
        else:
            plt.text(x - offset, y, f"{v:+.4f}", va="center", ha="right")

    plt.tight_layout()
    plt.show()


def plot_two_histograms(
    cnt1: Counter,
    cnt2: Counter,
    *,
    label1: str = "file1",
    label2: str = "file2",
    top_k: int = 20,
    min_count: int = 1,
    sort_by: str = "absdiff",
    title: str = "Top 1-gram distributions",
):
    """Plot side-by-side relative frequencies for a shared token subset.

    The plotting subset is derived from the pooled vocabulary and can be sorted
    either by contrast or by prominence in one or both files.
    """
    pooled = cnt1 + cnt2
    vocab = [tok for tok, c in pooled.items() if c >= min_count]

    if not vocab:
        print("No tokens satisfy min_count; nothing to plot.")
        return

    p = counter_to_probs(cnt1, vocab)
    q = counter_to_probs(cnt2, vocab)
    diff = p - q

    rows = list(zip(vocab, p, q, diff))

    if sort_by == "absdiff":
        rows.sort(key=lambda x: abs(x[3]), reverse=True)
    elif sort_by == "file1":
        rows.sort(key=lambda x: x[1], reverse=True)
    elif sort_by == "file2":
        rows.sort(key=lambda x: x[2], reverse=True)
    elif sort_by == "sum":
        rows.sort(key=lambda x: x[1] + x[2], reverse=True)
    else:
        raise ValueError("sort_by must be one of: 'absdiff', 'file1', 'file2', 'sum'")

    rows = rows[:top_k]

    labels = [r[0] for r in rows][::-1]
    vals1 = [r[1] for r in rows][::-1]
    vals2 = [r[2] for r in rows][::-1]

    y = np.arange(len(labels))
    height = 0.38

    plt.figure(figsize=(12, max(4, 0.45 * len(labels))))
    plt.barh(y - height / 2, vals1, height=height, label=label1)
    plt.barh(y + height / 2, vals2, height=height, label=label2)

    plt.yticks(y, labels)
    plt.xlabel("Relative frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pair_comparison(
    file1: str,
    file2: str,
    *,
    mode: str = "word",
    top_k: int = 20,
    min_count: int = 1,
    label1: str | None = None,
    label2: str | None = None,
    show_diff: bool = True,
    show_histograms: bool = True,
    histogram_sort_by: str = "absdiff",
):
    """Compute and visualize unigram differences for a single file pair.

    The function prints summary distances and optionally shows the difference
    plot and paired histograms. Returns the same scores used for plotting.
    """
    cnt1 = get_token_counts(file1, mode=mode)
    cnt2 = get_token_counts(file2, mode=mode)

    rows, l1_score, js_score = compute_diff_from_counters(
        cnt1,
        cnt2,
        top_k=top_k,
        min_count=min_count,
    )

    if label1 is None:
        label1 = Path(file1).name
    if label2 is None:
        label2 = Path(file2).name

    print(f"{label1} vs {label2} | L1={l1_score:.6f} | JS={js_score:.6f}")

    if show_diff:
        plot_diff(
            rows,
            title=f"Top 1-gram differences\n{label1} vs {label2} | L1={l1_score:.4f}, JS={js_score:.4f}",
        )

    if show_histograms:
        plot_two_histograms(
            cnt1,
            cnt2,
            label1=label1,
            label2=label2,
            top_k=top_k,
            min_count=min_count,
            sort_by=histogram_sort_by,
            title=f"Top 1-gram distributions\n{label1} vs {label2}",
        )

    return rows, l1_score, js_score


# ---- Pairwise scoring ----

def score_file_pairs(
    files: list[str],
    *,
    mode: str = "word",
    min_count: int = 1,
):
    """Score all unordered file pairs using cached unigram counts.

    Counts are computed once per file to keep larger sweeps inexpensive.
    Returns `(file1, file2, l1_score, js_score)` tuples for each pair.
    """
    cached = {f: get_token_counts(f, mode=mode) for f in files}
    results = []

    for f1, f2 in combinations(files, 2):
        cnt1 = cached[f1]
        cnt2 = cached[f2]

        total1 = sum(cnt1.values())
        total2 = sum(cnt2.values())

        if total1 == 0 or total2 == 0:
            results.append((f1, f2, float("nan"), float("nan")))
            continue

        _, l1_score, js_score = compute_diff_from_counters(
            cnt1,
            cnt2,
            top_k=20,  # Only the aggregate scores matter here.
            min_count=min_count,
        )
        results.append((f1, f2, l1_score, js_score))

    return results


def run(config: dict) -> dict:
    """Run the configured pairwise 1-gram analysis across dataset actors.

    Actor labels are resolved from the data catalog so downstream code can work
    with named comparisons instead of raw file paths.
    """
    dataset = config["dataset"]
    log_type = config["log_type"]
    mode = config["mode"]
    min_count = config.get("min_count", 1)

    # Log paths are assembled centrally so actor ordering stays consistent
    # across repeated sweeps and visualizations.
    file_pairs = [
        (actor, str(get_log_path(actor, log_type, dataset=dataset)))
        for actor in analysis_actors(dataset)
    ]

    labels = [label for label, _ in file_pairs]
    files = [path for _, path in file_pairs]
    label_by_path = {path: label for label, path in file_pairs}

    raw_results = score_file_pairs(
        files,
        mode=mode,
        min_count=min_count,
    )

    pairwise_results = [
        {
            "label_1": label_by_path[f1],
            "label_2": label_by_path[f2],
            "l1": l1,
            "js": js,
        }
        for f1, f2, l1, js in raw_results
    ]

    return {
        "labels": labels,
        "pairwise_results": pairwise_results,
    }


def build_sweep_configs(
    *,
    dataset: str,
    log_types: list[str],
    modes: list[str],
    metrics: list[str],
    min_count: int = 1,
) -> list[dict]:
    """Generate configuration dictionaries for a parameter sweep.

    The metric field is carried through unchanged so callers can align these
    analysis settings with a broader evaluation pipeline.
    """
    configs = []

    for log_type in log_types:
        for mode in modes:
            for metric in metrics:
                configs.append(
                    {
                        "dataset": dataset,
                        "log_type": log_type,
                        "mode": mode,
                        "metric": metric,
                        "min_count": min_count,
                    }
                )

    return configs
