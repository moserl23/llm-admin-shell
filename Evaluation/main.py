import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Iterable, Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

# custom
from evaluation_class import Evaluation
import all_file_paths

import textwrap
from typing import Any, Dict, Optional

from collections import Counter
from matplotlib.patches import Patch


# -----------------------------
# Labels / small utilities
# -----------------------------

def make_labels(files: List[str]) -> List[str]:
    """
    Create short, unique labels like 'Armin', 'Benni', ... from paths such as:
    .../LOGS_Result_Armin/exp7/audit.log
    Falls back to basename if pattern not found.
    """
    labels: List[str] = []
    pat = re.compile(r"LOGS_Result_([^/]+)")
    for f in files:
        f_str = str(f)
        m = pat.search(f_str)
        labels.append(m.group(1) if m else os.path.basename(f_str))
    return labels


def _to_nonneg_int(x) -> int:
    """
    Convert evaluator output to a non-negative int.
    - bool -> 0/1
    - int  -> int
    Raises if it's something unexpected.
    """
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    raise TypeError(f"Evaluator returned {type(x)}; expected bool or int.")


def _annotate_heatmap(ax, M, fmt="{:g}", fontsize=8):
    nrows, ncols = M.shape
    for i in range(nrows):
        for j in range(ncols):
            v = M[i, j]
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                s = "nan"
            else:
                try:
                    s = fmt.format(v)
                except Exception:
                    s = str(v)
            ax.text(j, i, s, ha="center", va="center", fontsize=fontsize)


def _heatmap(ax, M: np.ndarray, labels: List[str], title: str, integer_scale: bool = False):
    """
    Generic heatmap helper.
    integer_scale=True -> discrete bins (nice for small integer counts).
    """
    if integer_scale:
        max_val = int(np.nanmax(M)) if M.size else 0
        max_val = max(max_val, 0)
        cmap = plt.get_cmap("viridis", max_val + 1 if max_val >= 0 else 1)
        bounds = np.arange(-0.5, max_val + 1.5, 1)
        norm = BoundaryNorm(bounds, cmap.N)
        im = ax.imshow(M, cmap=cmap, norm=norm, interpolation="nearest")
        cbar = plt.colorbar(im, ax=ax, ticks=range(max_val + 1))
    else:
        im = ax.imshow(M, interpolation="nearest")
        cbar = plt.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Files")
    ax.set_ylabel("Files")
    return im, cbar


def slice_paths(experiment_number: int, log_type: str) -> list:
    return [
        name_dict[experiment_number][log_type]
        for name_dict in all_file_paths.files["singular"].values()
    ]


# -----------------------------
# Your existing "evaluate" heatmap (sum of hits)
# -----------------------------

def make_eval_fn(method_name: str, **kwargs) -> Callable[[str, str], int]:
    def run(file1: str, file2: str) -> int:
        ev = Evaluation()
        ev.set_files(file1, file2)

        needs_templates = method_name in {
            "n_gram_evaluate",
            "one_gram_evaluate",
            "complexity_index_evaluate",
        }
        if needs_templates:
            ev.build_templates()

        result = getattr(ev, method_name)(**kwargs)
        return max(0, _to_nonneg_int(result))

    run.__name__ = method_name
    return run


def plot_pairwise_differences(
    files: List[str],
    difference_functions: List[Callable[[str, str], int]],
    labels: List[str] | None = None,
    title: str = "Pairwise file differences",
):
    """
    Produces an NxN heatmap where each cell is the SUM of integer "hits"
    returned by all evaluation functions for that file pair.
    (Upper triangle computed and mirrored; assumes symmetry.)
    """
    if not files:
        raise ValueError("files is empty")
    if not difference_functions:
        raise ValueError("difference_functions is empty")

    n_files = len(files)

    if labels is None:
        labels = make_labels(files)
    if len(labels) != n_files:
        raise ValueError("labels must have same length as files")

    M = np.zeros((n_files, n_files), dtype=int)

    def safe_call(fn: Callable[[str, str], int], f1: str, f2: str) -> int:
        try:
            return max(0, _to_nonneg_int(fn(f1, f2)))
        except Exception as e:
            print(f"[WARN] {getattr(fn, '__name__', 'diff_fn')} failed for:\n  {f1}\n  {f2}\n  error: {e}")
            return 0

    # Compute only upper triangle and mirror (assumes symmetry)
    for i in range(n_files):
        for j in range(i + 1, n_files):
            val = sum(safe_call(fn, files[i], files[j]) for fn in difference_functions)
            M[i, j] = val
            M[j, i] = val

    # Adapt color scale to observed max
    max_val = int(M.max())
    cmap = plt.get_cmap("viridis", max_val + 1 if max_val >= 0 else 1)
    bounds = np.arange(-0.5, max_val + 1.5, 1) if max_val >= 0 else np.array([-0.5, 0.5])
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(M, cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_title(f"{title} (0–{max_val})")
    ax.set_xlabel("Files")
    ax.set_ylabel("Files")

    ax.set_xticks(range(n_files))
    ax.set_yticks(range(n_files))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Write values in each cell
    for i in range(n_files):
        for j in range(n_files):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax, ticks=range(max_val + 1))
    cbar.set_label("Total # of hits across all evaluation functions")

    plt.tight_layout()
    plt.show()

    return M


# -----------------------------
# (1) n_gram_report plots
# -----------------------------


def plot_n_gram_report_top_features(
    report: Dict[str, Any],
    *,
    top_k: Optional[int] = None,
    suptitle: str = "Top n-gram features per model",
    shared_color: str = "tab:orange",
    unique_color: str = "tab:blue",
):
    """
    report = evaluator.n_gram_report(...)
    Creates one horizontal bar plot per model, using signed weights.

    Tokens that appear in top_k for >=2 models are highlighted.
    Uses ONE global legend, cleanly separated from the title.
    """
    models = report.get("models", {})
    if not models:
        raise ValueError("Report has no 'models'.")

    if top_k is None:
        top_k = int(report.get("settings", {}).get("top_k_features", 30))

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    # --- 1) Determine shared tokens ---
    token_counter = Counter()
    for info in models.values():
        feats: List[Tuple[str, float]] = info.get("top_features", [])[:top_k]
        token_counter.update(t for t, _ in feats)

    shared_tokens = {tok for tok, c in token_counter.items() if c >= 2}

    # --- 2) Consistent x-axis scaling ---
    all_weights: List[float] = []
    for info in models.values():
        feats = info.get("top_features", [])[:top_k]
        all_weights.extend(w for _, w in feats)

    max_abs = max(abs(w) for w in all_weights) if all_weights else 1.0
    pad = 0.10 * max_abs
    xlim = (-(max_abs + pad), max_abs + pad)

    # --- 3) Plot ---
    for ax, (model_name, info) in zip(axes, models.items()):
        feats: List[Tuple[str, float]] = info.get("top_features", [])[:top_k]

        if not feats:
            ax.set_title(f"{model_name} (no features)")
            ax.axis("off")
            continue

        tokens = [t for t, _ in feats]
        weights = np.array([w for _, w in feats], dtype=float)
        y = np.arange(len(tokens))

        colors = [shared_color if t in shared_tokens else unique_color for t in tokens]

        ax.barh(y, weights, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(tokens)
        ax.invert_yaxis()
        ax.axvline(0.0, color="black", linewidth=1, zorder=0)
        ax.set_xlim(xlim)

        bac = info.get("balanced_accuracy", None)
        if bac is not None:
            ax.set_title(f"{model_name}  |  balanced_accuracy={bac:.3f}")
        else:
            ax.set_title(model_name)

        ax.set_xlabel("Feature weight (signed)")

    # --- Title (top) ---
    fig.suptitle(suptitle, y=0.98)

    # --- Legend (below title) ---
    legend_handles = [
        Patch(facecolor=unique_color, label="Unique (top-k in 1 model)"),
        Patch(facecolor=shared_color, label="Shared (top-k in ≥2 models)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=True,
        fontsize=9,
    )

    # Leave room at the top for title + legend
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()
    return fig





def run_and_plot_n_gram_report(
    file1: str,
    file2: str,
    *,
    max_lines: int = 5000,
    test_size: float = 0.30,
    random_state: int = 42,
    preprocessing_flag: bool = True,
    template_flag: bool = False,
    use_char_tfidf: bool = False,
    top_k_features: int = 30,
    window_mode: Literal["none", "raw", "cid"] = "none",
    window_size: int = 5,
):
    ev = Evaluation()
    ev.set_files(file1, file2)
    if template_flag:
        ev.build_templates()

    report = ev.n_gram_report(
        max_lines=max_lines,
        test_size=test_size,
        random_state=random_state,
        preprocessing_flag=preprocessing_flag,
        template_flag=template_flag,
        use_char_tfidf=use_char_tfidf,
        top_k_features=top_k_features,
        window_mode=window_mode,
        window_size=window_size,
    )
    plot_n_gram_report_top_features(report)
    return report


# -----------------------------
# (2) combo_detector_anomaly_count heatmap (COUNT, directed)
# -----------------------------

def anomaly_count_matrix(
    files: Sequence[str],
    *,
    labels: Optional[List[str]] = None,
    zero_diagonal: bool = True,
    max_lines: int = 5000,
) -> Tuple[np.ndarray, List[str]]:
    """
    M[i, j] = anomaly_count when training on files[i] and testing on files[j].
    This is DIRECTED (not symmetrized).
    """
    files = list(files)
    n = len(files)
    if labels is None:
        labels = make_labels(files)

    M = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            if zero_diagonal and i == j:
                M[i, j] = 0
                continue

            ev = Evaluation()
            ev.set_files(files[i], files[j])

            try:
                M[i, j] = int(ev.combo_detector_anomaly_count(max_lines=max_lines))
            except Exception as e:
                print(f"[WARN] combo_detector_anomaly_count failed for {files[i]} -> {files[j]}: {e}")
                M[i, j] = 0

    return M, labels


def plot_anomaly_count_heatmap(
    files: Sequence[str],
    *,
    labels: Optional[List[str]] = None,
    title: str = "Combo detector anomaly COUNT (train row -> test col)",
    max_lines: int = 5000,
    zero_diagonal: bool = True,
):
    M, labels = anomaly_count_matrix(
        files,
        labels=labels,
        max_lines=max_lines,
        zero_diagonal=zero_diagonal,
    )
    fig, ax = plt.subplots(figsize=(9, 8))
    _heatmap(ax, M, labels, title, integer_scale=True)
    _annotate_heatmap(ax, M, fmt="{:d}", fontsize=8)
    plt.tight_layout()
    plt.show()
    return M


def missing_combo_count_matrix(
    files: Sequence[str],
    *,
    labels: Optional[List[str]] = None,
    zero_diagonal: bool = True,
    max_lines: int = 5000,
) -> Tuple[np.ndarray, List[str]]:
    """
    M[i, j] = total_missing_combo_count when training on files[i]
    and testing on files[j].
    This is DIRECTED (not symmetrized).
    """
    files = list(files)
    n = len(files)
    if labels is None:
        labels = make_labels(files)

    M = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            if zero_diagonal and i == j:
                M[i, j] = 0
                continue

            ev = Evaluation()
            ev.set_files(files[i], files[j])

            try:
                M[i, j] = int(ev.combo_detector_total_missing_combos(max_lines=max_lines))
            except Exception as e:
                print(f"[WARN] combo_detector_total_missing_combos failed for {files[i]} -> {files[j]}: {e}")
                M[i, j] = 0

    return M, labels


def plot_missing_combo_count_heatmap(
    files: Sequence[str],
    *,
    labels: Optional[List[str]] = None,
    title: str = "Combo detector TOTAL MISSING COMBOS (train row -> test col)",
    max_lines: int = 5000,
    zero_diagonal: bool = True,
):
    M, labels = missing_combo_count_matrix(
        files,
        labels=labels,
        max_lines=max_lines,
        zero_diagonal=zero_diagonal,
    )
    fig, ax = plt.subplots(figsize=(9, 8))
    _heatmap(ax, M, labels, title, integer_scale=True)
    _annotate_heatmap(ax, M, fmt="{:d}", fontsize=8)
    plt.tight_layout()
    plt.show()
    return M


# -----------------------------
# (3) complexity_indices_result heatmaps per metric
# -----------------------------

def complexity_delta_matrices(
    files: Sequence[str],
    *,
    max_lines: int = 5000,
    window_size: int = 10,
    stride: int = 10,
    labels: Optional[List[str]] = None,
    zero_diagonal: bool = True,   # NEW
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    For each metric key in result["delta"], build an NxN matrix.
    Uses Evaluation.complexity_indices_result() which returns absolute deltas already.
    """
    files = list(files)
    n = len(files)
    if labels is None:
        labels = make_labels(files)

    # First pass: discover metric keys from one pair
    metric_keys = None
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ev = Evaluation()
            ev.set_files(files[i], files[j])
            try:
                res = ev.complexity_indices_result(
                    max_lines=max_lines, window_size=window_size, stride=stride
                )
                metric_keys = sorted(list(res.get("delta", {}).keys()))
                break
            except Exception:
                continue
        if metric_keys is not None:
            break

    if not metric_keys:
        raise RuntimeError("Could not determine complexity metric keys (all pairs failed?).")

    mats = {k: np.full((n, n), np.nan, dtype=float) for k in metric_keys}

    for i in range(n):
        for j in range(n):
            if zero_diagonal and i == j:
                for k in metric_keys:
                    mats[k][i, j] = 0.0
                continue

            ev = Evaluation()
            ev.set_files(files[i], files[j])
            try:
                res = ev.complexity_indices_result(
                    max_lines=max_lines, window_size=window_size, stride=stride
                )
                delta = res.get("delta", {})
                for k in metric_keys:
                    v = delta.get(k, np.nan)
                    mats[k][i, j] = float(v) if v is not None else np.nan
            except Exception as e:
                print(f"[WARN] complexity_indices_result failed for {files[i]} vs {files[j]}: {e}")
                # leave as nan

    return mats, labels


def normalize_matrix(M: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize a matrix ignoring NaNs.
    method:
      - "minmax": scales to [0,1] per-matrix
      - "zscore": (x-mean)/std per-matrix
    If the matrix has zero range / zero std (or all-NaN), returns zeros (preserving NaNs).
    """
    M = np.asarray(M, dtype=float)
    out = M.copy()

    finite = np.isfinite(out)
    if not np.any(finite):
        # all NaN/inf -> return all NaN (or zeros; but NaN is safer)
        return out

    vals = out[finite]

    method = method.lower().strip()
    if method in ("minmax", "min-max", "min_max"):
        mn = float(np.min(vals))
        mx = float(np.max(vals))
        denom = mx - mn
        if denom <= 0:
            out[finite] = 0.0
        else:
            out[finite] = (out[finite] - mn) / denom
        return out

    if method in ("zscore", "z", "standard", "standardize", "z_score"):
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        if sd <= 0:
            out[finite] = 0.0
        else:
            out[finite] = (out[finite] - mu) / sd
        return out

    raise ValueError(f"Unknown normalization method: {method}")


def combine_heatmaps(
    mats: Dict[str, np.ndarray],
    keys: List[str],
    *,
    normalization: str = "minmax",
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Normalize each matrix (per-key), then sum them up.
    - weights: optional dict {key: weight}. Missing keys default to 1.0.
    NaNs are ignored in the sum (i.e., treated as missing).
    """
    if not keys:
        raise ValueError("combine_heatmaps: keys is empty.")

    # Start with zeros, but use NaN-aware accumulation
    combined = np.zeros_like(mats[keys[0]], dtype=float)

    for k in keys:
        M = mats[k]
        w = float(weights.get(k, 1.0)) if weights else 1.0

        Mn = normalize_matrix(M, method=normalization)

        # NaNs: treat as missing -> add 0 for those entries
        Mn_safe = np.where(np.isfinite(Mn), Mn, 0.0)

        combined += w * Mn_safe

    return combined


def plot_complexity_delta_heatmaps(
    files: Sequence[str],
    *,
    labels: Optional[List[str]] = None,
    max_lines: int = 5000,
    window_size: int = 10,
    stride: int = 10,
    metrics: Optional[List[str]] = None,
    annotate: bool = True,
    zero_diagonal: bool = True,
    # NEW:
    plot_combined: bool = True,
    combined_normalization: str = "minmax",  # "minmax" or "zscore"
    combined_annotate: bool = True,
    combined_weights: Optional[Dict[str, float]] = None,
):
    mats, labels = complexity_delta_matrices(
        files,
        labels=labels,
        max_lines=max_lines,
        window_size=window_size,
        stride=stride,
        zero_diagonal=zero_diagonal,
    )

    keys = list(mats.keys())
    if metrics is not None:
        keys = [k for k in keys if k in set(metrics)]
        if not keys:
            raise ValueError("Requested metrics not found in computed matrices.")

    for k in keys:
        M = mats[k]
        fig, ax = plt.subplots(figsize=(9, 8))
        _heatmap(ax, M, labels, title=f"Complexity |Δ| heatmap: {k}", integer_scale=False)
        if annotate:
            _annotate_heatmap(ax, M, fmt="{:.3f}", fontsize=7)
        plt.tight_layout()
        plt.show()

    # --- Combined heatmap ---
    if plot_combined:
        combined = combine_heatmaps(
            mats,
            keys,
            normalization=combined_normalization,
            weights=combined_weights,
        )

        fig, ax = plt.subplots(figsize=(9, 8))
        _heatmap(
            ax,
            combined,
            labels,
            title=f"COMBINED complexity heatmap (sum of {len(keys)} metrics, norm={combined_normalization})",
            integer_scale=False,
        )
        if combined_annotate:
            # For minmax sum, values are typically 0..len(keys)
            # For zscore sum, values can be negative/positive
            _annotate_heatmap(ax, combined, fmt="{:.2f}", fontsize=7)

        plt.tight_layout()
        plt.show()

    return mats


# -----------------------------
# (4) one_gram_diff_report plot
# -----------------------------

def plot_one_gram_diff_report(
    diff_report: Dict[str, Any],
    *,
    top_k: Optional[int] = None,
    title: str = "Top 1-gram distribution differences (Δ = A - B)",
):
    """
    diff_report = evaluator.one_gram_diff_report(...)
    Plots top tokens by |delta| as a horizontal bar chart.
    """
    rows = diff_report.get("top_differences", [])
    if not rows:
        raise ValueError("diff_report has no 'top_differences' rows.")

    if top_k is None:
        top_k = int(diff_report.get("settings", {}).get("top_k", 20))

    rows = rows[:top_k]
    tokens = [r["token"] for r in rows]
    delta = np.array([r["delta"] for r in rows], dtype=float)

    y = np.arange(len(tokens))
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(tokens) + 2)))
    ax.barh(y, delta)
    ax.set_yticks(y)
    ax.set_yticklabels(tokens)
    ax.invert_yaxis()
    ax.axvline(0.0)
    ax.set_title(title)
    ax.set_xlabel("Δ (probability or count, per report settings)")
    plt.tight_layout()
    plt.show()
    return fig



def plot_one_gram_diff_report_pretty(
    diff_report: Dict[str, Any],
    *,
    top_k: Optional[int] = None,
    title: str = "Top 1-gram distribution differences (Δ = A - B)",
    wrap_width: int = 28,
    show_values: bool = True,
    value_fmt: str = "{:+.4f}",
):
    rows = diff_report.get("top_differences", [])
    if not rows:
        raise ValueError("diff_report has no 'top_differences' rows.")

    if top_k is None:
        top_k = int(diff_report.get("settings", {}).get("top_k", 20))

    rows = rows[:top_k]

    tokens = [r["token"] for r in rows]
    delta = np.array([r["delta"] for r in rows], dtype=float)

    # Wrap / shorten labels (helps a LOT)
    def nice_label(s: str) -> str:
        s = s.replace("\t", " ").replace("\n", " ")
        if len(s) <= wrap_width:
            return s
        return "\n".join(textwrap.wrap(s, width=wrap_width))

    labels = [nice_label(t) for t in tokens]

    # Dynamic figure sizing
    min_h = 5.0
    max_h = 10.0
    h = min(max_h, max(min_h, 0.42 * len(labels) + 1.5))
    fig, ax = plt.subplots(figsize=(11, h))

    y = np.arange(len(labels))
    ax.barh(y, delta)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    # Symmetric x-limits around 0
    max_abs = float(np.max(np.abs(delta))) if len(delta) else 1.0
    pad = 0.25 * max_abs
    ax.set_xlim(-(max_abs + pad), (max_abs + pad))

    # Zero line + subtle grid
    ax.axvline(0.0, linewidth=1.2)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.set_axisbelow(True)

    # Title + xlabel
    settings = diff_report.get("settings", {})
    mode = settings.get("mode", "")
    use_prob = settings.get("use_prob", True)
    xlabel = "Δ probability (p_A - p_B)" if use_prob else "Δ count (count_A - count_B)"
    ax.set_title(f"{title}\n(mode={mode})", pad=10)
    ax.set_xlabel(xlabel)

    # Optional numeric annotations on bars
    if show_values:
        for yi, d in zip(y, delta):
            x = d
            # place text slightly outside bar end
            offset = 0.02 * (max_abs + pad)
            ax.text(
                x + (offset if d >= 0 else -offset),
                yi,
                value_fmt.format(d),
                va="center",
                ha="left" if d >= 0 else "right",
                fontsize=9,
            )

    plt.tight_layout()
    plt.show()
    return fig


def run_and_plot_one_gram_diff_report(
    file1: str,
    file2: str,
    *,
    max_lines: int = 5000,
    preprocessing_flag: bool = True,
    template_flag: bool = False,
    mode: str = "word",
    top_k: int = 20,
    min_count: int = 2,
    use_prob: bool = True,
):
    ev = Evaluation()
    ev.set_files(file1, file2)
    if template_flag:
        ev.build_templates()

    rep = ev.one_gram_diff_report(
        max_lines=max_lines,
        preprocessing_flag=preprocessing_flag,
        template_flag=template_flag,
        mode=mode,
        top_k=top_k,
        min_count=min_count,
        use_prob=use_prob,
    )
    plot_one_gram_diff_report_pretty(rep, top_k=top_k)
    return rep




def plot_timestamp_distributions(
    list_a,
    list_b,
    labels=("A", "B"),
    *,
    title_prefix: str = "",
    bins: int | str = "sqrt",     # "sqrt", "fd", "auto", or an int
    log_x: bool = False,          # log-scale x axis (good for heavy tails)
    clip_quantile: float | None = 0.999,  # clip extreme tail for nicer plots; None disables
    show_grid: bool = True,
):
    """
    Create Histogram, smooth density (KDE-like), and ECDF plots for comparing two distributions.

    Parameters
    ----------
    list_a, list_b:
        Sequences of numeric values (e.g., inter-event time differences in seconds)

    labels:
        Tuple[str, str] for legend labeling

    title_prefix:
        Optional prefix added to each plot title

    bins:
        - "sqrt": sqrt rule
        - "fd": Freedman–Diaconis rule
        - "auto": numpy heuristic
        - int: explicit number of bins

    log_x:
        If True, x-axis is log scale. Values <= 0 are filtered out (since log undefined).

    clip_quantile:
        If not None, clip both distributions to [0, Q] (or [min_positive, Q] in log mode)
        for nicer visuals when there are huge outliers.

    Notes
    -----
    - The "KDE-like" plot uses a Gaussian-smoothed histogram, so you don't need SciPy.
    """

    a = np.asarray(list_a, dtype=float)
    b = np.asarray(list_b, dtype=float)

    # Remove non-finite
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    label_a, label_b = labels

    if log_x:
        # Remove non-positive values for log plots
        a = a[a > 0]
        b = b[b > 0]

    if a.size == 0 or b.size == 0:
        print("[WARN] plot_timestamp_distributions: one or both inputs are empty after filtering.")
        print(f"  {label_a}: n={a.size}")
        print(f"  {label_b}: n={b.size}")
        return

    # Optional tail clipping for nicer plots
    def _clip(x: np.ndarray) -> np.ndarray:
        if clip_quantile is None:
            return x
        q = float(np.quantile(x, clip_quantile))
        if log_x:
            # keep strictly positive; lower bound is min positive in x
            lo = float(np.min(x[x > 0])) if np.any(x > 0) else 1e-12
            return x[(x >= lo) & (x <= q)]
        else:
            return x[(x >= 0) & (x <= q)]

    a_plot = _clip(a)
    b_plot = _clip(b)

    if a_plot.size == 0 or b_plot.size == 0:
        # if clipping nuked everything, fall back to unclipped
        a_plot, b_plot = a, b

    # ---------- choose shared bins ----------
    def _num_bins_fd(x: np.ndarray) -> int:
        # Freedman–Diaconis rule
        x = np.asarray(x, dtype=float)
        if x.size < 2:
            return 10
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25
        if iqr <= 0:
            return max(10, int(np.sqrt(x.size)))
        bin_width = 2 * iqr / (x.size ** (1 / 3))
        if bin_width <= 0:
            return max(10, int(np.sqrt(x.size)))
        nb = int(np.ceil((x.max() - x.min()) / bin_width))
        return max(10, nb)

    combined = np.concatenate([a_plot, b_plot])

    if isinstance(bins, int):
        nbins = max(5, bins)
    else:
        bins = str(bins).lower()
        if bins == "fd":
            nbins = _num_bins_fd(combined)
        elif bins == "auto":
            # numpy's heuristic (we still convert to an int count)
            # We'll estimate by making histogram with 'auto' and using its bin count.
            _, edges = np.histogram(combined, bins="auto")
            nbins = max(5, len(edges) - 1)
        else:
            # default: sqrt rule
            nbins = max(10, int(np.sqrt(combined.size)))

    # Bin edges: linear or log
    xmin = float(np.min(combined))
    xmax = float(np.max(combined))

    if xmin == xmax:
        # add a tiny range so plotting doesn't crash
        eps = 1e-9 if not log_x else xmin * 1e-6
        xmin = max(xmin - eps, 1e-12 if log_x else 0.0)
        xmax = xmax + eps

    if log_x:
        xmin = max(xmin, 1e-12)
        edges = np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)
    else:
        edges = np.linspace(xmin, xmax, nbins + 1)

    # ---------- 1) Histogram ----------
    plt.figure(figsize=(10, 5))
    plt.hist(a_plot, bins=edges, alpha=0.5, density=True, label=f"{label_a} (n={a_plot.size})")
    plt.hist(b_plot, bins=edges, alpha=0.5, density=True, label=f"{label_b} (n={b_plot.size})")

    t = "Histogram Comparison"
    if title_prefix:
        t = f"{title_prefix} - {t}"
    plt.title(t)
    plt.xlabel("Inter-event time difference (s)")
    plt.ylabel("Density")
    if log_x:
        plt.xscale("log")
    plt.legend()
    if show_grid:
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---------- 2) Smooth density (KDE-like via Gaussian-smoothed histogram) ----------
    # We'll compute density histograms and smooth them with a small Gaussian kernel.
    def _gaussian_kernel(sigma_bins: float, radius: int) -> np.ndarray:
        x = np.arange(-radius, radius + 1)
        k = np.exp(-(x**2) / (2.0 * sigma_bins**2))
        k /= np.sum(k)
        return k

    def _smooth_density(x: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # density histogram
        h, _ = np.histogram(x, bins=edges, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        # choose smoothing strength relative to bin count
        # (you can tweak sigma_bins if you want smoother/less smooth)
        sigma_bins = max(1.0, 0.02 * len(h))
        radius = int(max(3, 3 * sigma_bins))
        kernel = _gaussian_kernel(sigma_bins=sigma_bins, radius=radius)
        h_smooth = np.convolve(h, kernel, mode="same")
        return centers, h_smooth

    plt.figure(figsize=(10, 5))

    xa, da = _smooth_density(a_plot, edges)
    xb, db = _smooth_density(b_plot, edges)

    plt.plot(xa, da, label=f"{label_a} smooth density")
    plt.plot(xb, db, label=f"{label_b} smooth density")

    t = "Smoothed density (KDE-like) Comparison"
    if title_prefix:
        t = f"{title_prefix} - {t}"
    plt.title(t)
    plt.xlabel("Inter-event time difference (s)")
    plt.ylabel("Density")
    if log_x:
        plt.xscale("log")
    plt.legend()
    if show_grid:
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---------- 3) ECDF ----------
    def ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.sort(x)
        y = np.arange(1, x.size + 1) / x.size
        return x, y

    plt.figure(figsize=(10, 5))
    x_a, y_a = ecdf(a_plot)
    x_b, y_b = ecdf(b_plot)

    plt.plot(x_a, y_a, label=f"{label_a} ECDF", drawstyle="steps-post")
    plt.plot(x_b, y_b, label=f"{label_b} ECDF", drawstyle="steps-post")

    t = "ECDF Comparison"
    if title_prefix:
        t = f"{title_prefix} - {t}"
    plt.title(t)
    plt.xlabel("Inter-event time difference (s)")
    plt.ylabel("Cumulative probability")
    if log_x:
        plt.xscale("log")
    plt.legend()
    if show_grid:
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_inter_event_distributions(
    file1: str,
    file2: str,
    *,
    max_lines: int = 5000,
    min_events: int = 20,
    labels: tuple[str, str] = ("A", "B"),
):
    ev = Evaluation()
    ev.set_files(file1, file2)

    res = ev.inter_event_result(max_lines=max_lines, min_events=min_events)
    diffs1 = res["diffs_1"]
    diffs2 = res["diffs_2"]

    # Nice auto labels if not provided
    label_a, label_b = labels
    label_a = f"{label_a} ({res.get('dominant_type_1','?')}, n_ts={res.get('n_timestamps_1',0)})"
    label_b = f"{label_b} ({res.get('dominant_type_2','?')}, n_ts={res.get('n_timestamps_2',0)})"

    if len(diffs1) == 0 or len(diffs2) == 0:
        print("[INFO] Not enough timestamps to compute inter-event diffs for one or both files.")
        print(f"  file1: {file1}")
        print(f"  file2: {file2}")
        return res

    # changed here
    # remove noise
    diffs1 = [elem for elem in diffs1 if elem > 0.2 and elem < 15]
    diffs2 = [elem for elem in diffs2 if elem > 0.2 and elem < 15]


    plot_timestamp_distributions(diffs1, diffs2, labels=(label_a, label_b), log_x=False) # changed here
    return res


# -----------------------------
# (NEW) n_gram_report metric heatmaps (pairwise)
# -----------------------------

def n_gram_report_metric_matrices(
    files: Sequence[str],
    *,
    labels: Optional[List[str]] = None,
    max_lines: int = 5000,
    test_size: float = 0.30,
    random_state: int = 42,
    preprocessing_flag: bool = True,
    template_flag: bool = False,
    use_char_tfidf: bool = False,
    top_k_features: int = 30,
    train_idx: Optional[Sequence[int]] = None,
    test_idx: Optional[Sequence[int]] = None,
    model_agg: str = "mean",
    metrics: Optional[List[str]] = None,
    zero_diagonal: bool = False,
    # NEW:
    pair: Optional[Tuple[int, int]] = None,   # e.g. (0,1)
    window_mode: Literal["none", "raw", "cid"] = "none",
    window_size: int = 5,
) -> Tuple[Dict[str, np.ndarray], List[str]]:

    files = list(files)
    n = len(files)
    if labels is None:
        labels = make_labels(files)

    def _agg(values: List[float]) -> float:
        if not values:
            return float("nan")
        if model_agg == "max":
            return float(np.max(values))
        if model_agg == "min":
            return float(np.min(values))
        return float(np.mean(values))

    # If fixed split is provided and no pair is specified, default to (0,1)
    if (train_idx is not None or test_idx is not None) and pair is None:
        pair = (0, 1)

    # --- discover metrics automatically if not provided ---
    metric_keys: Optional[List[str]] = metrics
    if metric_keys is None:
        # discover using the pair if we’re in single-pair mode; else scan pairs
        scan_pairs = [pair] if pair is not None else [(i, j) for i in range(n) for j in range(i + 1, n)]
        for (i, j) in scan_pairs:
            ev = Evaluation()
            ev.set_files(files[i], files[j])
            if template_flag:
                ev.build_templates()
            try:
                rep = ev.n_gram_report(
                    max_lines=max_lines,
                    test_size=test_size,
                    random_state=random_state,
                    preprocessing_flag=preprocessing_flag,
                    template_flag=template_flag,
                    use_char_tfidf=use_char_tfidf,
                    top_k_features=top_k_features,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    window_size=window_size,
                    window_mode=window_mode,
                )
                models = rep.get("models", {})
                if models:
                    first_model = next(iter(models.values()))
                    metric_keys = sorted([k for k, v in first_model.items() if isinstance(v, (int, float))])
                    break
            except Exception:
                continue

    if not metric_keys:
        raise RuntimeError("Could not determine n_gram_report metric keys (all pairs failed?).")

    # --- allocate matrices ---
    if pair is not None:
        # 1x1 matrices in single-pair mode
        mats: Dict[str, np.ndarray] = {k: np.full((1, 1), np.nan, dtype=float) for k in metric_keys}
        i, j = pair

        ev = Evaluation()
        ev.set_files(files[i], files[j])
        if template_flag:
            ev.build_templates()

        rep = ev.n_gram_report(
            max_lines=max_lines,
            test_size=test_size,
            random_state=random_state,
            preprocessing_flag=preprocessing_flag,
            template_flag=template_flag,
            use_char_tfidf=use_char_tfidf,
            top_k_features=top_k_features,
            train_idx=train_idx,
            test_idx=test_idx,
            window_size=window_size,
            window_mode=window_mode,
        )
        models = rep.get("models", {})
        for k in metric_keys:
            vals = []
            for info in models.values():
                v = info.get(k, None)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    vals.append(float(v))
            mats[k][0, 0] = _agg(vals)

        # labels become the pair label
        pair_labels = [f"{labels[i]} vs {labels[j]}"]
        return mats, pair_labels

    # --- original NxN behavior ---
    mats: Dict[str, np.ndarray] = {k: np.full((n, n), np.nan, dtype=float) for k in metric_keys}

    for i in range(n):
        for j in range(n):
            if zero_diagonal and i == j:
                for k in metric_keys:
                    mats[k][i, j] = 0.0
                continue

            ev = Evaluation()
            ev.set_files(files[i], files[j])
            if template_flag:
                ev.build_templates()

            try:
                rep = ev.n_gram_report(
                    max_lines=max_lines,
                    test_size=test_size,
                    random_state=random_state,
                    preprocessing_flag=preprocessing_flag,
                    template_flag=template_flag,
                    use_char_tfidf=use_char_tfidf,
                    top_k_features=top_k_features,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    window_size=window_size,
                    window_mode=window_mode,
                )
                models = rep.get("models", {})
                for k in metric_keys:
                    vals = []
                    for info in models.values():
                        v = info.get(k, None)
                        if isinstance(v, (int, float)) and np.isfinite(v):
                            vals.append(float(v))
                    mats[k][i, j] = _agg(vals)

            except Exception as e:
                print(f"[WARN] n_gram_report failed for {files[i]} vs {files[j]}: {e}")

    return mats, labels


def plot_n_gram_report_metric_heatmaps(
    files: Sequence[str],
    *,
    labels: Optional[List[str]] = None,
    title_prefix: str = "n_gram_report metrics",
    max_lines: int = 1000000,
    test_size: float = 0.30,
    random_state: int = 42,
    preprocessing_flag: bool = True,
    template_flag: bool = False,
    use_char_tfidf: bool = False,
    top_k_features: int = 30,
    train_idx: Optional[Sequence[int]] = None,
    test_idx: Optional[Sequence[int]] = None,
    model_agg: str = "mean",
    metrics: Optional[List[str]] = None,
    annotate: bool = True,
    zero_diagonal: bool = False,
    pair: Optional[Tuple[int, int]] = None,
    window_mode: Literal["none", "raw", "cid"] = "none",
    window_size: int = 5,
):
    """
    Behavior:
      - Normal mode (NxN): plots one heatmap per metric key.
      - Single-pair mode (1x1): plots ONE bar chart with all metric values.

    Returns:
      mats: dict metric -> matrix (NxN or 1x1 depending on mode)
    """
    mats, out_labels = n_gram_report_metric_matrices(
        files,
        labels=labels,
        max_lines=max_lines,
        test_size=test_size,
        random_state=random_state,
        preprocessing_flag=preprocessing_flag,
        template_flag=template_flag,
        use_char_tfidf=use_char_tfidf,
        top_k_features=top_k_features,
        model_agg=model_agg,
        metrics=metrics,
        zero_diagonal=zero_diagonal,
        train_idx=train_idx,
        test_idx=test_idx,
        pair=pair,
        window_size=window_size,
        window_mode=window_mode,
    )

    # Decide metric keys to use
    keys = list(mats.keys())
    if metrics is not None:
        keys = [k for k in keys if k in set(metrics)]
        if not keys:
            raise ValueError("Requested metrics not found in computed matrices.")

    # Detect 1x1 mode
    any_M = next(iter(mats.values()))
    is_single = (any_M.shape == (1, 1))

    # -------------------------
    # 1x1 mode -> BAR CHART
    # -------------------------
    if is_single:
        # Extract scalar metric values
        values: List[float] = []
        for k in keys:
            v = float(mats[k][0, 0])
            values.append(v)

        # Sort metrics by value (optional; makes bar chart easier to read)
        order = np.argsort(np.nan_to_num(values, nan=-np.inf))[::-1]
        keys_sorted = [keys[i] for i in order]
        vals_sorted = [values[i] for i in order]

        # Plot
        fig_h = max(4.0, 0.35 * len(keys_sorted) + 1.5)
        fig, ax = plt.subplots(figsize=(10, fig_h))

        y = np.arange(len(keys_sorted))
        ax.barh(y, vals_sorted)
        ax.set_yticks(y)
        ax.set_yticklabels(keys_sorted)
        ax.invert_yaxis()

        # Nice title with pair label
        pair_label = out_labels[0] if out_labels else "pair"
        ax.set_title(f"{title_prefix}: {pair_label} (model_agg={model_agg})", pad=12)
        ax.set_xlabel("Metric value")

        # Optional numeric labels on bars
        if annotate:
            finite_vals = [v for v in vals_sorted if np.isfinite(v)]
            max_abs = max(abs(v) for v in finite_vals) if finite_vals else 1.0
            pad = 0.02 * max_abs if max_abs > 0 else 0.02

            for yi, v in enumerate(vals_sorted):
                if not np.isfinite(v):
                    txt = "nan"
                    x = 0.0
                else:
                    txt = f"{v:.3f}"
                    x = v

                ax.text(
                    x + (pad if (np.isfinite(v) and v >= 0) else -pad),
                    yi,
                    txt,
                    va="center",
                    ha="left" if (np.isfinite(v) and v >= 0) else "right",
                    fontsize=9,
                )

        # Optional: if most metrics are probabilities, you can clamp axis to [0,1]
        # Comment this in/out depending on your metric set.
        # ax.set_xlim(0, 1)

        ax.grid(True, axis="x", alpha=0.25)
        plt.tight_layout()
        plt.show()
        return mats

    # -------------------------
    # NxN mode -> HEATMAPS
    # -------------------------
    for k in keys:
        M = mats[k]
        fig, ax = plt.subplots(figsize=(9, 8))
        _heatmap(
            ax,
            M,
            out_labels,
            title=f"{title_prefix} heatmap: {k} (model_agg={model_agg})",
            integer_scale=False,
        )
        if annotate:
            _annotate_heatmap(ax, M, fmt="{:.3f}", fontsize=7)
        plt.tight_layout()
        plt.show()

    return mats

# -----------------------------
# deep learning method
# -----------------------------

def run_deep_learning_report_two_files(
    file1: str,
    file2: str,
    *,
    max_lines: int = 5000,
    preprocessing_flag: bool = True,
    template_flag: bool = False,
    window_mode: Literal["none", "raw", "cid"] = "none",
    window_size: int = 5,
    train_idx: Optional[Sequence[int]] = None,
    test_idx: Optional[Sequence[int]] = None,
    # training knobs (keep defaults unless you want to tweak)
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    ev = Evaluation()
    ev.set_files(file1, file2)
    if template_flag:
        ev.build_templates()

    return ev.deep_learning_report(
        max_lines=max_lines,
        preprocessing_flag=preprocessing_flag,
        template_flag=template_flag,
        window_mode=window_mode,
        window_size=window_size,
        train_idx=train_idx,
        test_idx=test_idx,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

def run_inter_event_classifier_report_two_files(
    file1: str,
    file2: str,
    *,
    max_lines: int = 5000,
    min_events: int = 20,
    test_size: float = 0.30,
    random_state: int = 42,
    model: Literal["logreg", "svm", "rf"] = "logreg",
    use_log_transform: bool = True,
    use_scaling: bool = True,
    # --- NEW: window classification ---
    window_mode: Literal["single", "window"] = "single",
    window_size: int = 5,
    window_stride: int = 1,
    drop_last: bool = True,
    split_within_class: bool = True,
) -> Dict[str, Any]:

    ev = Evaluation()
    ev.set_files(file1, file2)

    return ev.inter_event_classifier_report(
        max_lines=max_lines,
        min_events=min_events,
        test_size=test_size,
        random_state=random_state,
        model=model,
        use_log_transform=use_log_transform,
        use_scaling=use_scaling,
        window_mode=window_mode,
        window_size=window_size,
        window_stride=window_stride,
        drop_last=drop_last,
        split_within_class=split_within_class,
    )



def get_holdout_indices(
    *,
    indices: dict[str, dict[str, tuple[int, int]]],
    log_type: str,
    humans: Iterable[str],
    ais: Iterable[str],
) -> tuple[list[int], list[int]]:
    """
    Build (train_idx, test_idx) for n_gram_report manual splitting.

    Parameters
    ----------
    indices:
        Output of calc_indices().
        indices[log_type][name] = (start, end)

    log_type:
        e.g. "audit", "syslog", "nextcloud", ...

    humans:
        Iterable of human names to hold out (e.g. ["Benni", "Armin"])

    ais:
        Iterable of AI names to hold out (e.g. ["GPT4o", "GPT5"])

    Returns
    -------
    train_idx, test_idx:
        Lists of indices into X_text = human + ai
    """

    if log_type not in indices:
        raise KeyError(f"Unknown log_type: {log_type}")

    idx_map = indices[log_type]

    # --- build test indices ---
    test_idx: list[int] = []

    for name in list(humans) + list(ais):
        if name not in idx_map:
            raise KeyError(f"Name '{name}' not found for log_type '{log_type}'")

        start, end = idx_map[name]
        test_idx.extend(range(start, end))

    if not test_idx:
        raise ValueError("Empty test_idx: no humans or AIs specified for holdout.")

    # --- build train indices as complement ---
    total_len = max(end for (_, end) in idx_map.values())
    test_set = set(test_idx)

    train_idx = [i for i in range(total_len) if i not in test_set]

    if not train_idx:
        raise ValueError("Empty train_idx: everything was held out.")

    return train_idx, test_idx

from typing import Sequence, Tuple, List
import numpy as np

def _idx_to_ranges(idxs: Sequence[int]) -> list[tuple[int, int]]:
    """[3,4,5, 10,11] -> [(3,6),(10,12)] (half-open)."""
    if not idxs:
        return []
    idxs = sorted(set(int(i) for i in idxs))
    out = []
    start = prev = idxs[0]
    for i in idxs[1:]:
        if i == prev + 1:
            prev = i
        else:
            out.append((start, prev + 1))
            start = prev = i
    out.append((start, prev + 1))
    return out


def adjust_split_indices_for_windows(
    *,
    n_human_lines: int,
    n_ai_lines: int,
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    window_size: int,
    stride: int | None = None,
    drop_last: bool = True,
) -> tuple[list[int], list[int]]:
    """
    Convert line-level train/test indices (into X_text = human_lines + ai_lines)
    into window-level train/test indices (into X_text = human_windows + ai_windows),
    with NO leakage: any window that overlaps any test line becomes test.

    IMPORTANT: stride/drop_last must match your window builder.
      - your code uses stride=None => stride=window_size
      - drop_last=True
    """
    if stride is None:
        stride = window_size

    n_total = n_human_lines + n_ai_lines

    train_idx = np.asarray(list(train_idx), dtype=int)
    test_idx  = np.asarray(list(test_idx), dtype=int)

    if np.any(train_idx < 0) or np.any(train_idx >= n_total) or np.any(test_idx < 0) or np.any(test_idx >= n_total):
        raise ValueError(f"Indices out of range for n_total={n_total}")

    # --- split test indices into per-side (LOCAL) indices ---
    test_h = [int(i) for i in test_idx if i < n_human_lines]
    test_a = [int(i - n_human_lines) for i in test_idx if i >= n_human_lines]

    # compress to ranges (optional but efficient)
    test_h_ranges = _idx_to_ranges(test_h)
    test_a_ranges = _idx_to_ranges(test_a)

    # helper (your overlap logic)
    def line_ranges_to_window_mask(n_lines: int, test_ranges: Sequence[tuple[int, int]]) -> np.ndarray:
        starts = list(range(0, n_lines, stride))
        if drop_last:
            starts = [s for s in starts if s + window_size <= n_lines]
        else:
            starts = [s for s in starts if s < n_lines]

        mask = np.zeros(len(starts), dtype=bool)
        for k, s in enumerate(starts):
            e = min(s + window_size, n_lines)
            for (t0, t1) in test_ranges:
                # overlap => test
                if (s < t1) and (e > t0):
                    mask[k] = True
                    break
        return mask

    mask_h = line_ranges_to_window_mask(n_human_lines, test_h_ranges)
    mask_a = line_ranges_to_window_mask(n_ai_lines, test_a_ranges)

    n_h_windows = int(mask_h.size)
    n_a_windows = int(mask_a.size)

    test_w = np.where(mask_h)[0].tolist() + (np.where(mask_a)[0] + n_h_windows).tolist()
    train_w = np.where(~mask_h)[0].tolist() + (np.where(~mask_a)[0] + n_h_windows).tolist()

    return train_w, test_w




# -----------------------------
# Main / examples
# -----------------------------
if __name__ == "__main__":
    singular_flag = False
    experimentAgg_flag = False
    totalAgg_flag = True

    if singular_flag:
        files = slice_paths(7, "audit")
        labels = make_labels(files)

    elif experimentAgg_flag:
        files = [all_file_paths.files["experimentAgg"][person]["audit"] for person in all_file_paths.names]
        labels = all_file_paths.names

    elif totalAgg_flag:
        file1 = all_file_paths.files["totalAgg"]["Human"]["audit"]
        file2 = all_file_paths.files["totalAgg"]["AI"]["audit"]
        files = [file1, file2]
        labels = ["Human", "AI"]

    else:
        raise ValueError("No dataset selection flag is enabled.")

    # combo detector anomaly count heatmap
    M_anom = plot_anomaly_count_heatmap(
        files,
        labels=labels,
        title="Combo detector anomaly COUNT (train row -> test col)",
        max_lines=1_000_000,
        zero_diagonal=True,
    )
    print("Anomaly count matrix:")
    print(M_anom)

    # combo detector anomaly count heatmap
    M_anom = plot_missing_combo_count_heatmap(
        files,
        labels=labels,
        title="Combo detector missing-combo COUNT (train row -> test col)",
        max_lines=1_000_000,
        zero_diagonal=True,
    )
    print("Anomaly count matrix:")
    print(M_anom)

    # complexity delta heatmaps
    mats = plot_complexity_delta_heatmaps(
        files,
        labels=labels,
        max_lines=1_000_000,
        window_size=10,
        stride=10,
        annotate=True,
        zero_diagonal=True,
        plot_combined=True,
        combined_normalization="minmax",
        combined_annotate=True,
    )


