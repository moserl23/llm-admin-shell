"""Run nested CNN model selection across predefined validation/test group splits.

The script evaluates a small preprocessing grid and CNN hyperparameter grid for
each outer split, selects configurations by validation performance only, and
records the corresponding held-out test metrics to CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.core.shared.loader import load_examples, LoadConfig
from src.core.ml.splits import make_splits
from src.core.ml.val_test_combs import make_val_test_splits
from src.core.ml.benchmark import bench

from src.ml_pipelines.cnn_pipeline import Candidate, CNNConfig, search


# -------------------------
# Argument parsing
# -------------------------
def parse_args():
    """Parse CLI options for dataset selection, scoring metric, and output control.

    The runner supports a debug mode that limits the number of outer splits and
    an optional timing mode for the slowest stages.
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=str,
        default="Nextcloud",
        choices=["Nextcloud", "WordPress", "Data", "Data_WP"],
        help="Which aggregated dataset root to use. Preferred names: Nextcloud, WordPress. Legacy aliases: Data, Data_WP.",
    )
    p.add_argument("--out_csv", type=str, default="results/cnn_360_nested_results.csv")
    p.add_argument(
        "--metric",
        type=str,
        default="f1_macro",
        choices=["f1_macro", "f1_weighted", "accuracy", "balanced_accuracy"],
    )
    p.add_argument("--limit_outer", type=int, default=0, help="If >0, only run first N outer splits (debug)")
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Print timing for critical sections (load_examples, search).",
    )
    return p.parse_args()


# -------------------------
# LoadConfig grid
# -------------------------
@dataclass(frozen=True)
class NamedLoad:
    name: str
    cfg: LoadConfig


def make_load_configs(dataset: str) -> List[NamedLoad]:
    """Build the small preprocessing grid explored inside each outer split.

    The grid is intentionally narrow because each option triggers a full nested
    CNN search. Returned entries pair a readable name with a `LoadConfig`.
    """
    base = dict(
        dataset=dataset,
        log_files=("audit.log",),
        prefix_with_log_type=False,
        max_lines_per_file=None,
    )

    out: List[NamedLoad] = []

    # CID windows are the main representation of interest for this CNN setup.
    for preprocess_mode in ["soft"]:
        for ws in [30]:
            st = max(1, ws // 2)
            out.append(NamedLoad(
                name=f"cids_ws{ws}_st{st}_{preprocess_mode}",
                cfg=LoadConfig(
                    **base,
                    preprocess_mode=preprocess_mode,
                    window_mode="cids",
                    window_size=ws,
                    window_stride=st,
                    window_drop_last=True,
                    cid_prefix="CID",
                ),
            ))

    # Line-window variants are kept disabled because truncation undermines the
    # benefit of preserving longer raw-line context in this pipeline.
    '''
    for preprocess_mode in ["soft", "aggressive"]:
        for ws in [25, 50]:
            st = max(1, ws // 2)
            out.append(NamedLoad(
                name=f"lines_ws{ws}_st{st}_{preprocess_mode}",
                cfg=LoadConfig(
                    **base,
                    preprocess_mode=preprocess_mode,
                    window_mode="lines",
                    window_size=ws,
                    window_stride=st,
                    window_drop_last=True,
                    join_token=" <EOL> ",
                ),
            ))
    '''

    # The no-window baseline keeps one line per example for comparison.
    for preprocess_mode in ["soft"]:
        out.append(NamedLoad(
            name=f"none_{preprocess_mode}",
            cfg=LoadConfig(
                **base,
                preprocess_mode=preprocess_mode,
                window_mode="none",
            ),
        ))

    return out


# -------------------------
# CNN hyperparameter grid
# -------------------------
def make_candidates() -> List[Candidate]:
    """Create the compact CNN search grid used by the inner loop.

    The space is deliberately conservative so nested evaluation remains
    computationally feasible while still testing the main architectural choices.
    """
    candidates: List[Candidate] = []

    for lr in [1e-3]:
        for num_filters in [64, 128]:
            for dropout in [0.5]:
                candidates.append(
                    Candidate(
                        cfg=CNNConfig(
                            lr=lr,
                            weight_decay=1e-4,
                            embed_dim=64,
                            num_filters=num_filters,
                            fc_dim=128,
                            dropout=dropout,
                            kernel_sizes=(3, 5, 7),
                            batch_size=32,
                            epochs=12,
                            early_stopping=True,
                            patience=3,
                            eval_every=1,
                            max_len_cap=512,
                            len_percentile=95.0,
                            seed=42,
                        )
                    )
                )

    return candidates


def _safe_float(x: object) -> float:
    """Convert metric-like values to float, falling back to `nan` on failure."""
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def _resolve_out_csv(path: str) -> str:
    """Normalize the output CSV path and ensure its parent directory exists."""
    p = Path(path)
    if str(p.parent) == ".":
        p = Path("results") / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


# -------------------------
# Main experiment loop
# -------------------------
def main():
    """Run the nested evaluation and write one summary row per outer split.

    For each predefined validation/test group pairing, the script selects the
    best preprocessing-model combination on validation data only and stores the
    corresponding validation and test metrics.
    """

    args = parse_args()
    metric = args.metric
    out_csv = _resolve_out_csv(args.out_csv)

    all_outer_splits = make_val_test_splits(args.dataset)
    outer_splits = all_outer_splits
    if args.limit_outer and args.limit_outer > 0:
        outer_splits = outer_splits[: args.limit_outer]

    load_grid = make_load_configs(args.dataset)
    cand_grid = make_candidates()

    print(f"Dataset     : {args.dataset}")
    print(f"Outer splits: {len(outer_splits)} (of {len(all_outer_splits)})")
    print(f"LoadConfigs : {len(load_grid)}")
    print(f"Candidates  : {len(cand_grid)}")
    print(f"Metric      : {metric}")
    print(f"Writing CSV : {out_csv}")

    rows: List[Dict[str, object]] = []

    for outer_i, (val_groups, test_groups) in enumerate(outer_splits, 1):
        print("\n" + "=" * 100)
        print(f"[OUTER {outer_i:03d}/{len(outer_splits)}] val={val_groups} test={test_groups}")
        print("=" * 100)

        # Track the validation-selected winner for this outer split only.
        best_overall = None

        for li, named in enumerate(load_grid, 1):
            print(f"\n  --- LoadConfig [{li:02d}/{len(load_grid)}] {named.name} ---")

            try:
                examples = []
                with bench(
                    args.benchmark,
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

            # Small group-based splits can become unstable; skip underpowered folds.
            min_train = 400
            min_val = 100
            min_test = 100
            if n_train < min_train or n_val < min_val or n_test < min_test:
                print(f"  ⚠ Too small split train={n_train} val={n_val} test={n_test}. Skipping.")
                continue

            # Test metrics are computed only for the validation-selected model to
            # preserve the nested-evaluation boundary.
            with bench(args.benchmark, f"search({named.name})"):
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

            print(f"  best VAL {metric}={val_metric:.4f} | TEST {metric}={test_metric:.4f} | {best_cand.cfg}")

            # Outer-split model selection is based strictly on validation metrics.
            if best_overall is None or val_metric > best_overall[0]:
                best_overall = (val_metric, named, best_cand, best_val_res, best_test_res, (n_train, n_val, n_test))

        if best_overall is None:
            print("⚠ No valid result for this outer split.")
            continue

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
            "selected_cnn_cfg": repr(best_cand.cfg),

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
        print(f"    CNN cfg   : {best_cand.cfg}")
        print(f"    VAL  {metric}={row['selection_val_score']:.4f} | TEST {metric}={row['selection_test_score']:.4f}")

        rows.append(row)

    # -------------------------
    # Write results
    # -------------------------
    if not rows:
        print("\nNo rows collected; nothing to write.")
        return

    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # -------------------------
    # Aggregate summary
    # -------------------------
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
