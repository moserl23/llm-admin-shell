import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evaluation_class import Evaluation

import all_file_paths


# -----------------------------
# Helpers
# -----------------------------

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


def _make_labels_from_paths(files: Sequence[str]) -> List[str]:
    # if you already have make_labels(), just use that instead
    import os, re
    labels = []
    pat = re.compile(r"LOGS_Result_([^/]+)")
    for f in files:
        m = pat.search(str(f))
        labels.append(m.group(1) if m else os.path.basename(str(f)))
    return labels


def _heatmap(ax, M: np.ndarray, labels: List[str], title: str, integer_scale: bool = False):
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


# -----------------------------
# (1) n_gram_report plots
# -----------------------------

def plot_n_gram_report_top_features(
    report: Dict[str, Any],
    *,
    top_k: Optional[int] = None,
    suptitle: str = "Top n-gram features per model",
):
    """
    report = evaluator.n_gram_report(...)
    Creates one horizontal bar plot per model, using signed weights.
    """
    models = report.get("models", {})
    if not models:
        raise ValueError("Report has no 'models'.")

    # Determine top_k from report settings if not given
    if top_k is None:
        top_k = int(report.get("settings", {}).get("top_k_features", 30))

    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1, figsize=(10, max(3, 3 * n_models)))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, info) in zip(axes, models.items()):
        feats: List[Tuple[str, float]] = info.get("top_features", [])
        feats = feats[:top_k]

        if not feats:
            ax.set_title(f"{model_name} (no features)")
            ax.axis("off")
            continue

        tokens = [t for t, _ in feats]
        weights = np.array([w for _, w in feats], dtype=float)

        y = np.arange(len(tokens))
        ax.barh(y, weights)
        ax.set_yticks(y)
        ax.set_yticklabels(tokens)
        ax.invert_yaxis()
        ax.axvline(0.0)  # sign separator
        bac = info.get("balanced_accuracy", None)
        if bac is not None:
            ax.set_title(f"{model_name}  |  balanced_accuracy={bac:.3f}")
        else:
            ax.set_title(model_name)
        ax.set_xlabel("Feature weight (signed)")

    fig.suptitle(suptitle)
    plt.tight_layout()
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
    )
    plot_n_gram_report_top_features(report)
    return report


# -----------------------------
# (2) combo_detector_anomaly_count heatmap (COUNT, directed)
# -----------------------------

def anomaly_count_matrix(
    files: Sequence[str],
    *,
    max_lines: int = 5000,
    labels: Optional[List[str]] = None,
    zero_diagonal: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    M[i, j] = anomaly_count when training on files[i] and testing on files[j].
    This is DIRECTED (not symmetrized).
    """
    files = list(files)
    n = len(files)
    if labels is None:
        labels = _make_labels_from_paths(files)

    M = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            if zero_diagonal and i == j:
                M[i, j] = 0
                continue

            ev = Evaluation()
            ev.set_files(files[i], files[j])
            # combo_detector_anomaly_count uses lines_file_1 as train, lines_file_2 as test
            try:
                M[i, j] = int(ev.combo_detector_anomaly_count())
            except Exception as e:
                print(f"[WARN] combo_detector_anomaly_count failed for {files[i]} -> {files[j]}: {e}")
                M[i, j] = 0

    return M, labels


def plot_anomaly_count_heatmap(
    files: Sequence[str],
    *,
    labels: Optional[List[str]] = None,
    title: str = "Combo detector anomaly COUNT (train row -> test col)",
):
    M, labels = anomaly_count_matrix(files, labels=labels)
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
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    For each metric key in result["delta"], build an NxN matrix.
    Uses Evaluation.complexity_indices_result() which returns absolute deltas already.
    """
    files = list(files)
    n = len(files)
    if labels is None:
        labels = _make_labels_from_paths(files)

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
            if i == j:
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


def plot_complexity_delta_heatmaps(
    files: Sequence[str],
    *,
    labels: Optional[List[str]] = None,
    max_lines: int = 5000,
    window_size: int = 10,
    stride: int = 10,
    metrics: Optional[List[str]] = None,
    annotate: bool = True,
):
    mats, labels = complexity_delta_matrices(
        files,
        labels=labels,
        max_lines=max_lines,
        window_size=window_size,
        stride=stride,
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
    plot_one_gram_diff_report(rep, top_k=top_k)
    return rep




def slice_paths(experiment_number: int, log_type: str)-> list:
    return [
        name_dict[experiment_number][log_type]
        for name_dict in all_file_paths.files.values()
    ]



if __name__ == "__main__":

    files = slice_paths(7, "audit")

    #run_and_plot_n_gram_report(files[0], files[1], top_k_features=25)

    plot_anomaly_count_heatmap(files, labels=make_labels(files))