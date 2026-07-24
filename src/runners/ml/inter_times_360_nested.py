"""Nested evaluation for inter-event time features across predefined group splits.

For each outer (validation, test) split, the script searches over windowing choices
and hyperparameters, selects by validation performance only, and records held-out
test metrics for the selected configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.shared.loader import load_examples, LoadConfig
from src.core.ml.splits import make_splits
from src.core.ml.val_test_combs import make_val_test_splits
from src.core.ml.benchmark import bench

from src.ml_pipelines.inter_times_pipeline import Candidate, search

def resolve_log_files(dataset: str, log_type: str) -> Tuple[str, ...]:
    """Map a dataset/log-type pair to the concrete log file(s) to load.

    The mapping is intentionally explicit because not every dataset exposes the
    same logical log sources. Returns a tuple for direct use in `LoadConfig`.
    """
    allowed = {
        "Nextcloud": {"audit": "audit.log", "syslog": "syslog.log", "nextcloud": "nextcloud.log"},
        "WordPress": {"audit": "audit.log", "syslog": "syslog.log"},
        "Data": {"audit": "audit.log", "syslog": "syslog.log", "nextcloud": "nextcloud.log"},
        "Data_WP": {"audit": "audit.log", "syslog": "syslog.log"},
    }

    if dataset not in allowed:
        raise ValueError(f"Unknown dataset: {dataset}")

    if log_type not in allowed[dataset]:
        valid = ", ".join(sorted(allowed[dataset].keys()))
        raise ValueError(
            f"log_type={log_type!r} is not valid for dataset={dataset!r}. "
            f"Allowed values: {valid}"
        )

    return (allowed[dataset][log_type],)

# ---- CLI ----
def parse_args():
    """Parse command-line options for the nested evaluation run.

    The model family is fixed per run, while preprocessing variants and model
    hyperparameters are tuned inside the nested search.
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=str,
        default="Nextcloud",
        choices=["Nextcloud", "WordPress", "Data", "Data_WP"],
        help="Which aggregated dataset root to use. Preferred names: Nextcloud, WordPress. Legacy aliases: Data, Data_WP.",
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "dummy_most_frequent",
            "dummy_stratified",
            "gnb",
            "logreg",
            "svm",
            "sgd_hinge",
            "sgd_log",
            "ridge",
            "knn",
        ],
        help="Which model family to run (model choice is NOT tuned).",
    )
    p.add_argument(
        "--log_type",
        type=str,
        required=True,
        choices=["audit", "syslog", "nextcloud"],
        help="Logical log source to evaluate. "
             "For dataset=Nextcloud: audit, syslog, nextcloud. "
             "For dataset=WordPress: audit, syslog.",
    )

    p.add_argument(
        "--metric",
        type=str,
        default="f1_macro",
        choices=["f1_macro", "f1_weighted", "accuracy", "balanced_accuracy"],
        help="Metric used to select best config on VAL (nested-CV correct).",
    )
    p.add_argument("--out_csv", type=str, default="results/inter_times_360_nested_results.csv")
    p.add_argument("--limit_outer", type=int, default=0, help="If >0, run only first N outer splits (debug).")
    p.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of outer splits to run in parallel. Use 1 to keep serial behavior.",
    )
    p.add_argument("--clip_max", type=float, default=3600.0, help="Clip inter-event diffs at this many seconds.")
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Print timing for critical sections (load_examples, search).",
    )
    return p.parse_args()


def _safe_float(x: object) -> float:
    """Convert a metric-like value to float and fall back to `nan` on failure.

    This keeps CSV writing and summary aggregation robust to missing or
    unavailable metrics.
    """
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def _resolve_out_csv(path: str) -> str:
    """Normalize the output path and ensure its parent directory exists.

    Bare filenames are written under `results/` to keep experiment outputs in a
    predictable location. Returns the resolved path as a string.
    """
    p = Path(path)
    if str(p.parent) == ".":
        p = Path("results") / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


# ---- Load configuration grid ----
@dataclass(frozen=True)
class NamedLoad:
    """Pair a human-readable configuration name with a `LoadConfig` instance."""
    name: str
    cfg: LoadConfig


def make_load_configs(*, clip_max: float, dataset: str, log_type: str) -> List[NamedLoad]:
    """Build the preprocessing grid for inter-time feature extraction.

    The grid varies window size, stride, and time scaling while keeping the
    dataset and log source fixed for a given run.
    """
    base = dict(
        dataset=dataset,
        log_files=resolve_log_files(dataset, log_type),
        window_mode="inter_times",
        window_drop_last=True,
        max_lines_per_file=None,
        prefix_with_log_type=False,
        preprocess_mode="raw",  # timestamps extracted from raw
        inter_time_clip_max=clip_max,
    )

    out: List[NamedLoad] = []

    for window_size in [1, 5, 10, 25, 50]:
        for stride in [window_size, max(1, window_size // 2)]:
            out.append(NamedLoad(
                name=f"seconds_ws{window_size}_st{stride}",
                cfg=LoadConfig(
                    **base,
                    window_size=window_size,
                    window_stride=stride,
                    inter_time_unit="seconds",
                    inter_time_add_epsilon=0.0,
                ),
            ))
            out.append(NamedLoad(
                name=f"log10_ws{window_size}_st{stride}",
                cfg=LoadConfig(
                    **base,
                    window_size=window_size,
                    window_stride=stride,
                    inter_time_unit="log10_seconds",
                    inter_time_add_epsilon=1e-6,
                ),
            ))

    return out


# ---- Model candidate grid ----
def make_model_candidates(model: str) -> List[Candidate]:
    """Return the hyperparameter grid for one fixed model family.

    Model identity is treated as part of the experimental design rather than a
    tuned choice, so the search only spans parameters within the selected family.
    """
    candidates: List[Candidate] = []

    if model == "dummy_most_frequent":
        candidates.append(Candidate(
            "dummy_most_frequent",
            {},
            use_scaler=False
        ))
        return candidates

    if model == "dummy_stratified":
        candidates.append(Candidate(
            "dummy_stratified",
            {},
            use_scaler=False
        ))
        return candidates

    if model == "gnb":
        for var_smoothing in np.logspace(-12, -6, 10):
            candidates.append(Candidate(
                "gnb",
                {"var_smoothing": float(var_smoothing)},
                use_scaler=False
            ))
        return candidates

    if model == "logreg":
        for C in [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]:
            candidates.append(Candidate(
                "logreg",
                {"C": C, "class_weight": "balanced"},
                use_scaler=True
            ))
        return candidates

    if model == "svm":
        for C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
            candidates.append(Candidate(
                "svm",
                {"C": C, "class_weight": "balanced"},
                use_scaler=True
            ))
        return candidates

    if model == "sgd_hinge":
        for alpha in [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]:
            candidates.append(Candidate(
                "sgd_hinge",
                {"alpha": alpha, "class_weight": "balanced", "max_iter": 5000, "tol": 1e-3},
                use_scaler=True
            ))
        return candidates

    if model == "sgd_log":
        for alpha in [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]:
            candidates.append(Candidate(
                "sgd_log",
                {"alpha": alpha, "class_weight": "balanced", "max_iter": 5000, "tol": 1e-3},
                use_scaler=True
            ))
        return candidates

    if model == "ridge":
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 300.0, 1000.0]:
            candidates.append(Candidate(
                "ridge",
                {"alpha": alpha},
                use_scaler=True
            ))
        return candidates

    if model == "knn":
        for k in [1, 3, 5, 7, 11, 21, 31]:
            for w in ["uniform", "distance"]:
                candidates.append(Candidate(
                    "knn",
                    {"n_neighbors": k, "weights": w},
                    use_scaler=True
                ))
        return candidates

    raise ValueError(f"Unknown model: {model}")


def _run_one_outer_split(
    outer_i: int,
    total_outer: int,
    val_groups: Tuple[str, str],
    test_groups: Tuple[str, str],
    *,
    model: str,
    metric: str,
    clip_max: float,
    dataset: str,
    log_type: str,
    benchmark: bool,
) -> Optional[Dict[str, object]]:
    """Evaluate one outer validation/test split under nested model selection.

    For the given group split, the function searches over preprocessing and
    model hyperparameters using validation data only, then records metrics for
    the selected configuration on both validation and held-out test data.
    """
    print("\n" + "=" * 100)
    print(f"[OUTER {outer_i:03d}/{total_outer}] val={val_groups} test={test_groups}")
    print("=" * 100)

    load_grid = make_load_configs(clip_max=clip_max, dataset=dataset, log_type=log_type)
    cand_grid = make_model_candidates(model)

    best_overall = None

    for li, named in enumerate(load_grid, 1):
        print(f"\n  --- LoadConfig [{li:02d}/{len(load_grid)}] {named.name} ---")

        try:
            examples = []
            with bench(
                benchmark,
                f"load_examples({named.name})",
                meta_fn=lambda: {"n": len(examples)},
            ):
                examples = load_examples(named.cfg)
        except Exception as e:
            print(f"  ⚠ load failed for {named.name}: {e}")
            continue

        if not examples:
            print("  ⚠ No examples produced. Skipping.")
            continue

        y = np.array([e.label for e in examples], dtype=object)
        groups = np.array([e.group for e in examples], dtype=object)

        # The outer split is defined at the group level so the chosen human/AI
        # sources stay isolated between validation and test.
        split = make_splits(
            y,
            groups=groups,
            val_groups=val_groups,
            test_groups=test_groups,
        )

        n_train = int(len(split.train_idx))
        n_val = int(len(split.val_idx))
        n_test = int(len(split.test_idx))
        if n_train == 0 or n_val == 0 or n_test == 0:
            print(f"  ⚠ Bad split sizes train={n_train} val={n_val} test={n_test}. Skipping.")
            continue

        # Small splits can make some outer combinations too unstable to compare
        # meaningfully, especially for sparse log sources.
        min_train = 200
        min_val = 100
        min_test = 100
        if n_train < min_train or n_val < min_val or n_test < min_test:
            print(f"  ⚠ Too small split train={n_train} val={n_val} test={n_test}. Skipping.")
            continue

        with bench(benchmark, f"search({named.name})"):
            # Test is evaluated only for the validation-selected candidate to
            # preserve the nested-CV separation between selection and reporting.
            best_cand, best_val_res, best_test_res, _all_val = search(
                examples,
                split,
                cand_grid,
                metric=metric,
                evaluate_test_for_all=False,
                verbose=False,
            )

        val_metric = _safe_float(getattr(best_val_res, metric, np.nan))
        test_metric = _safe_float(getattr(best_test_res, metric, np.nan))

        print(f"  best VAL {metric}={val_metric:.4f} | TEST {metric}={test_metric:.4f} | {best_cand}")

        # The outer winner is the configuration with the strongest validation
        # score; test performance is recorded but never used for selection.
        if best_overall is None or val_metric > best_overall[0]:
            best_overall = (val_metric, named, best_cand, best_val_res, best_test_res, (n_train, n_val, n_test))

    if best_overall is None:
        print("⚠ No valid result for this outer split.")
        return None

    val_metric, named, best_cand, best_val_res, best_test_res, (n_train, n_val, n_test) = best_overall

    row = {
        "outer_i": outer_i,
        "val_human": val_groups[0],
        "val_ai": val_groups[1],
        "test_human": test_groups[0],
        "test_ai": test_groups[1],
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "selected_load_name": named.name,
        "selected_model": best_cand.model_name,
        "selected_model_params": repr(best_cand.model_params),
        "selected_use_scaler": int(bool(best_cand.use_scaler)),
        "val_accuracy": _safe_float(getattr(best_val_res, "accuracy", np.nan)),
        "val_balanced_accuracy": _safe_float(getattr(best_val_res, "balanced_accuracy", np.nan)),
        "val_f1_macro": _safe_float(getattr(best_val_res, "f1_macro", np.nan)),
        "val_f1_weighted": _safe_float(getattr(best_val_res, "f1_weighted", np.nan)),
        "test_accuracy": _safe_float(getattr(best_test_res, "accuracy", np.nan)),
        "test_balanced_accuracy": _safe_float(getattr(best_test_res, "balanced_accuracy", np.nan)),
        "test_f1_macro": _safe_float(getattr(best_test_res, "f1_macro", np.nan)),
        "test_f1_weighted": _safe_float(getattr(best_test_res, "f1_weighted", np.nan)),
        "selection_metric": metric,
        "selection_val_score": _safe_float(getattr(best_val_res, metric, np.nan)),
        "selection_test_score": _safe_float(getattr(best_test_res, metric, np.nan)),
        "val_precision_macro": _safe_float(getattr(best_val_res, "precision_macro", np.nan)),
        "val_precision_weighted": _safe_float(getattr(best_val_res, "precision_weighted", np.nan)),
        "val_recall_macro": _safe_float(getattr(best_val_res, "recall_macro", np.nan)),
        "val_recall_weighted": _safe_float(getattr(best_val_res, "recall_weighted", np.nan)),
        "val_mcc": _safe_float(getattr(best_val_res, "mcc", np.nan)),
        "val_cohen_kappa": _safe_float(getattr(best_val_res, "cohen_kappa", np.nan)),
        "val_per_class_metrics": json.dumps(getattr(best_val_res, "per_class_metrics", {}), sort_keys=True),
        "test_precision_macro": _safe_float(getattr(best_test_res, "precision_macro", np.nan)),
        "test_precision_weighted": _safe_float(getattr(best_test_res, "precision_weighted", np.nan)),
        "test_recall_macro": _safe_float(getattr(best_test_res, "recall_macro", np.nan)),
        "test_recall_weighted": _safe_float(getattr(best_test_res, "recall_weighted", np.nan)),
        "test_mcc": _safe_float(getattr(best_test_res, "mcc", np.nan)),
        "test_cohen_kappa": _safe_float(getattr(best_test_res, "cohen_kappa", np.nan)),
        "test_per_class_metrics": json.dumps(getattr(best_test_res, "per_class_metrics", {}), sort_keys=True),
    }

    print("\n>>> SELECTED (by VAL only)")
    print(f"    LoadConfig: {named.name}")
    print(f"    Candidate : {best_cand}")
    print(f"    VAL  {metric}={row['selection_val_score']:.4f}")
    print(f"    TEST {metric}={row['selection_test_score']:.4f}")

    return row


# ---- Main entry point ----
def main():
    """Run the full nested evaluation over all requested outer splits.

    The script executes each outer split serially or in parallel, writes one row
    per completed split, and prints a compact summary across test metrics.
    """
    args = parse_args()
    model = args.model
    metric = args.metric
    out_csv = _resolve_out_csv(args.out_csv)
    n_jobs = max(1, int(args.n_jobs))

    all_outer_splits = make_val_test_splits(args.dataset)
    outer_splits = all_outer_splits
    if args.limit_outer and args.limit_outer > 0:
        outer_splits = outer_splits[: args.limit_outer]

    load_grid = make_load_configs(
        clip_max=float(args.clip_max),
        dataset=args.dataset,
        log_type=args.log_type,
    )
    cand_grid = make_model_candidates(model)

    print(f"Dataset     : {args.dataset}")
    print(f"Log type    : {args.log_type}")
    print(f"Model       : {model}")
    print(f"Metric      : {metric}")
    print(f"Outer splits: {len(outer_splits)} (of {len(all_outer_splits)})")
    print(f"LoadConfigs : {len(load_grid)}")
    print(f"Candidates  : {len(cand_grid)}")
    print(f"Parallel jobs: {n_jobs}")
    print(f"Writing CSV : {out_csv}")

    rows: List[Dict[str, object]] = []
    worker_args = [
        (outer_i, len(outer_splits), val_groups, test_groups)
        for outer_i, (val_groups, test_groups) in enumerate(outer_splits, 1)
    ]

    if n_jobs == 1:
        for outer_i, total_outer, val_groups, test_groups in worker_args:
            row = _run_one_outer_split(
                outer_i,
                total_outer,
                val_groups,
                test_groups,
                model=model,
                metric=metric,
                clip_max=float(args.clip_max),
                dataset=args.dataset,
                log_type=args.log_type,
                benchmark=args.benchmark,
            )
            if row is not None:
                rows.append(row)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [
                ex.submit(
                    _run_one_outer_split,
                    outer_i,
                    total_outer,
                    val_groups,
                    test_groups,
                    model=model,
                    metric=metric,
                    clip_max=float(args.clip_max),
                    dataset=args.dataset,
                    log_type=args.log_type,
                    benchmark=args.benchmark,
                )
                for outer_i, total_outer, val_groups, test_groups in worker_args
            ]
            for fut in as_completed(futures):
                row = fut.result()
                if row is not None:
                    rows.append(row)

    # ---- Write CSV ----
    if not rows:
        print("\nNo rows collected; nothing to write.")
        return

    rows.sort(key=lambda r: int(r["outer_i"]))

    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # ---- Quick summary ----
    test_f1s = np.array([r["test_f1_macro"] for r in rows], dtype=float)
    test_bals = np.array([r["test_balanced_accuracy"] for r in rows], dtype=float)

    print("\n" + "#" * 100)
    print("DONE. Summary over outer splits (selected-by-VAL per split):")
    print(f"Rows written: {len(rows)} -> {out_csv}")
    print(f"TEST f1_macro: mean={np.nanmean(test_f1s):.4f} median={np.nanmedian(test_f1s):.4f} std={np.nanstd(test_f1s):.4f}")
    print(f"TEST bal_acc : mean={np.nanmean(test_bals):.4f} median={np.nanmedian(test_bals):.4f} std={np.nanstd(test_bals):.4f}")
    print("#" * 100)


if __name__ == "__main__":
    main()
