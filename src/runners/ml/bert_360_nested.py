
"""Nested evaluation runner for BERT-style log classification experiments.

This script combines predefined outer validation/test group splits with a small
loader and transformer search space, selecting configurations by validation
performance only and recording held-out test results per outer split.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import numpy as np

from src.core.shared.loader import load_examples, LoadConfig
from src.core.ml.splits import make_splits
from src.core.ml.val_test_combs import make_val_test_splits
from src.core.ml.benchmark import bench

from src.ml_pipelines.bert_pipeline import Candidate, TransformerConfig, search


def parse_args():
    """Parse command-line arguments for the nested BERT evaluation runner.

    The interface is intentionally small because the outer split definitions and
    candidate grids are controlled in code for reproducibility.
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=str,
        default="Nextcloud",
        choices=["Nextcloud", "WordPress", "Data", "Data_WP"],
        help="Which aggregated dataset root to use. Preferred names: Nextcloud, WordPress. Legacy aliases: Data, Data_WP.",
    )
    p.add_argument("--out_csv", type=str, default="results/bert_360_nested_results.csv")
    p.add_argument("--metric", type=str, default="f1_macro", choices=["f1_macro", "f1_weighted", "accuracy", "balanced_accuracy"])
    p.add_argument("--limit_outer", type=int, default=0, help="If >0, only run first N outer splits (debug)")
    p.add_argument("--benchmark", action="store_true", help="Print timing for critical sections (load_examples, search).")    
    return p.parse_args()


@dataclass(frozen=True)
class NamedLoad:
    name: str
    cfg: LoadConfig


def make_load_configs(dataset: str) -> List[NamedLoad]:
    """Build a compact grid of loader settings for the outer search.

    The grid is intentionally conservative because each configuration is paired
    with transformer tuning inside the nested evaluation.
    """
    base = dict(
        dataset=dataset,
        log_files=("audit.log",),
        prefix_with_log_type=False,
        max_lines_per_file=None,
    )

    out: List[NamedLoad] = []

    # ---- Per-line inputs ----
    for preprocess_mode in ["soft"]:
        out.append(NamedLoad(
            name=f"none_{preprocess_mode}",
            cfg=LoadConfig(
                **base,
                preprocess_mode=preprocess_mode,
                window_mode="none",
            ),
        ))

    # ---- Line windows ----
    # Left disabled here because longer concatenated sequences increase runtime
    # quickly while the transformer context length remains fixed.
    '''
    for preprocess_mode in ["soft", "aggressive"]:
        for ws in [10, 25]:
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
    # ---- CID windows ----
    # CID windows trade raw line order for template-level structure and assume
    # Drain3 is available in the local environment.
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

    return out


def make_candidates() -> List[Candidate]:
    """Return the transformer candidates considered in the inner search.

    The candidate set is deliberately small so that each outer split remains
    computationally feasible while still probing a meaningful model variant.
    """
    candidates: List[Candidate] = []

    for model_name in ["distilroberta-base"]: #"roberta-base"]: # "bert-base-uncased" # "distilroberta-base"
        for lr in [2e-5]:
            for max_len in [128]:
                for wd in [0.01]:
                    candidates.append(
                        Candidate(
                            cfg=TransformerConfig(
                                model_name=model_name,
                                max_length=max_len,
                                batch_size=64,
                                lr=lr,
                                epochs=4,
                                weight_decay=wd,
                                warmup_ratio=0.1,
                                seed=42,
                                patience=1,
                            )
                        )
                    )

    return candidates


def _safe_float(x: object) -> float:
    """Convert a metric-like value to float and fall back to ``nan`` on failure."""
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def _resolve_out_csv(path: str) -> str:
    """Resolve the output CSV path and ensure its parent directory exists."""
    p = Path(path)
    if str(p.parent) == ".":
        p = Path("results") / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def main():
    """Run nested outer/inner evaluation and write one summary row per split.

    For each predefined outer split, the script selects the loader/model pair by
    validation performance only, then records the corresponding held-out test
    metrics for downstream aggregation.
    """
    args = parse_args()
    metric = args.metric
    out_csv = _resolve_out_csv(args.out_csv)

    # ---- Set up search space and outer splits ----
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

    # ---- Outer evaluation loop ----
    for outer_i, (val_groups, test_groups) in enumerate(outer_splits, 1):
        print("\n" + "=" * 100)
        print(f"[OUTER {outer_i:03d}/{len(outer_splits)}] val={val_groups} test={test_groups}")
        print("=" * 100)

        # Track the winner for this outer split using validation performance only.
        best_overall = None

        for li, named in enumerate(load_grid, 1):
            print(f"\n  --- LoadConfig [{li:02d}/{len(load_grid)}] {named.name} ---")


            try:
                examples = []
                with bench(
                    args.benchmark,
                    f"load_examples({named.name})",
                    meta_fn=lambda: {"n": len(examples)}
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

            # The outer split is group-based: validation and test groups are held
            # out explicitly, and the remaining groups form the training pool.
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

            min_train = 400
            min_val = 100
            min_test = 100
            if n_train < min_train or n_val < min_val or n_test < min_test:
                print(f"  ⚠ Too small split train={n_train} val={n_val} test={n_test}. Skipping.")
                continue

            # In the inner search, test evaluation is restricted to the selected
            # candidate to avoid leaking test feedback into model selection.
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

            # Nested evaluation requires that outer-level selection depends only
            # on validation performance, never on the outer test result.
            if best_overall is None or val_metric > best_overall[0]:
                best_overall = (val_metric, named, best_cand, best_val_res, best_test_res, (n_train, n_val, n_test))

        if best_overall is None:
            print("⚠ No valid result for this outer split.")
            continue

        val_metric, named, best_cand, best_val_res, best_test_res, (n_train, n_val, n_test) = best_overall

        # Record a stable set of metrics so later analyses do not depend on the
        # selection metric used for this particular run.
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
            "selected_transformer_cfg": repr(best_cand.cfg),

            "val_accuracy": _safe_float(getattr(best_val_res, "accuracy", np.nan)),
            "val_balanced_accuracy": _safe_float(getattr(best_val_res, "balanced_accuracy", np.nan)),
            "val_f1_macro": _safe_float(getattr(best_val_res, "f1_macro", np.nan)),
            "val_f1_weighted": _safe_float(getattr(best_val_res, "f1_weighted", np.nan)),

            "test_accuracy": _safe_float(getattr(best_test_res, "accuracy", np.nan)),
            "test_balanced_accuracy": _safe_float(getattr(best_test_res, "balanced_accuracy", np.nan)),
            "test_f1_macro": _safe_float(getattr(best_test_res, "f1_macro", np.nan)),
            "test_f1_weighted": _safe_float(getattr(best_test_res, "f1_weighted", np.nan)),

            "selection_metric": metric,
            "selection_val_score": _safe_float(val_metric),
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
        print(f"    BERT cfg  : {best_cand.cfg}")
        print(f"    VAL  {metric}={row['selection_val_score']:.4f} | TEST {metric}={row['selection_test_score']:.4f}")

        rows.append(row)

    # ---- Write per-split results ----
    if not rows:
        print("\nNo rows collected; nothing to write.")
        return

    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # ---- Aggregate summary ----
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
