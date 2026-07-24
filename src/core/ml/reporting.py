"""Utility printers for split diagnostics and model evaluation summaries.

This module keeps reporting lightweight: it provides quick split sanity checks
and console summaries for comparing models across validation or test runs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from src.core.ml.data import Example
from src.core.ml.splits import Split
from src.core.ml.eval import EvalResult


def _count_values(arr: np.ndarray) -> Dict[str, int]:
    """Return value counts for a 1D array of labels or other hashable entries."""
    out: Dict[str, int] = {}
    for v in arr.tolist():
        out[v] = out.get(v, 0) + 1
    return out


def print_split_stats(examples: List[Example], split: Split) -> None:
    """
    Print a compact sanity check for a dataset split.

    Reports sample counts, label distributions, and the number of distinct
    groups in each partition when group metadata is available.
    """
    y = np.array([ex.label for ex in examples], dtype=object)

    def unique_groups(idxs: np.ndarray) -> Optional[int]:
        # Ignore missing group identifiers so grouped and ungrouped datasets can
        # use the same reporting path without special handling upstream.
        g = [examples[i].group for i in idxs if examples[i].group is not None]
        if not g:
            return None
        return len(set(g))

    # ---- Split overview ----
    for name, idxs in [("train", split.train_idx), ("val", split.val_idx), ("test", split.test_idx)]:
        counts = _count_values(y[idxs])
        ug = unique_groups(idxs)
        ug_str = "n/a" if ug is None else str(ug)
        print(f"{name:5s}: n={len(idxs):4d} | label_counts={counts} | unique_groups={ug_str}")


def print_leaderboard(
    results: Dict[str, Dict[str, EvalResult]],
    split_name: str = "val",
    *,
    sort_by: str = "f1_macro",
    top_k: Optional[int] = None,
) -> None:
    """
    Print a leaderboard-style summary for one evaluation split.

    Expects ``results[model_name][split_name]`` to contain ``EvalResult``
    objects and sorts models by the requested aggregate metric.
    """
    if sort_by not in {"f1_macro", "f1_weighted", "accuracy"}:
        raise ValueError("sort_by must be one of: f1_macro, f1_weighted, accuracy")

    rows: List[Tuple[str, float, float, float]] = []
    for model_name, res in results.items():
        if split_name not in res:
            raise KeyError(f"split_name='{split_name}' not found for model '{model_name}'")
        r = res[split_name]
        rows.append((model_name, r.f1_macro, r.f1_weighted, r.accuracy))

    sort_idx = {"f1_macro": 1, "f1_weighted": 2, "accuracy": 3}[sort_by]
    rows.sort(key=lambda x: x[sort_idx], reverse=True)

    if top_k is not None:
        rows = rows[:top_k]

    print(f"\nLeaderboard ({split_name}) — sorted by {sort_by}")
    print("model\t\tf1_macro\tf1_weighted\taccuracy")
    for m, f1m, f1w, acc in rows:
        print(f"{m:12s}\t{f1m:.4f}\t\t{f1w:.4f}\t\t{acc:.4f}")


def print_model_report(
    results: Dict[str, Dict[str, EvalResult]],
    model_name: str,
    split_name: str = "val",
) -> None:
    """
    Print the detailed report for a single model on one split.

    This exposes the stored per-class metrics and confusion matrix for
    inspection after model selection or debugging.
    """
    if model_name not in results:
        raise KeyError(f"Unknown model '{model_name}'. Available: {list(results.keys())}")

    if split_name not in results[model_name]:
        raise KeyError(f"split_name='{split_name}' not found for model '{model_name}'")

    r = results[model_name][split_name]
    print(f"\n=== {model_name} ({split_name}) classification report ===\n{r.per_class_report}")
    print(f"Confusion matrix ({split_name}):\n{r.confusion}")
