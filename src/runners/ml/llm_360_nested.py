
"""Run nested validation/test evaluation for the LLM-360 RAG pipeline.

The script selects preprocessing and retrieval settings by validation score within
each outer split and reports the corresponding held-out test result once per split.
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

from src.ml_pipelines.llm_pipeline import Candidate, RAGLLMConfig, search


# -------------------------
# Args
# -------------------------
def parse_args():
    """Parse CLI arguments for the nested LLM-360 evaluation runner.

    The options control dataset selection, the outer-split subset to run, and
    whether uncertain retrieval decisions may fall back to the chat model.
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=str,
        default="Nextcloud",
        choices=["Nextcloud", "WordPress", "Data", "Data_WP"],
        help="Which aggregated dataset root to use. Preferred names: Nextcloud, WordPress. Legacy aliases: Data, Data_WP.",
    )
    p.add_argument("--out_csv", type=str, default="results/llm_360_nested_results.csv")
    p.add_argument(
        "--metric",
        type=str,
        default="f1_macro",
        choices=["f1_macro", "f1_weighted", "accuracy", "balanced_accuracy"],
    )
    p.add_argument("--limit_outer", type=int, default=0, help="If >0, only run first N outer splits (debug)")
    p.add_argument("--use_llm_fallback", type=int, default=1, choices=[0, 1], help="0=local-only, 1=use LLM fallback")
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Print timing for critical sections (load_examples, search).",
    )
    return p.parse_args()


def _safe_float(x: object) -> float:
    """Convert a value to float while preserving failures as NaN.

    This keeps downstream metric aggregation robust when a result object is
    missing a field or exposes a non-numeric placeholder.
    """
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def _resolve_out_csv(path: str) -> str:
    """Normalize the output CSV path and create its parent directory.

    Bare filenames are written under `results/` so repeated runs default to a
    consistent output location.
    """
    p = Path(path)
    if str(p.parent) == ".":
        p = Path("results") / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


# -------------------------
# 1) LoadConfig grid
# -------------------------
@dataclass(frozen=True)
class NamedLoad:
    """Pair a human-readable config name with the corresponding load settings."""
    name: str
    cfg: LoadConfig





def make_load_configs(dataset: str) -> List[NamedLoad]:
    """Build the small preprocessing grid used in the outer evaluation.

    The grid is intentionally narrow so the nested search stays interpretable
    and computationally manageable across many outer splits.
    """
    base = dict(
        dataset=dataset,
        log_files=("audit.log",),
        prefix_with_log_type=False,
        max_lines_per_file=None,
    )

    out: List[NamedLoad] = []

    preprocess_mode = "soft"

    # ---- Raw lines ----
    out.append(NamedLoad(
        name=f"none_{preprocess_mode}",
        cfg=LoadConfig(
            **base,
            preprocess_mode=preprocess_mode,
            window_mode="none",
        ),
    ))

    # ---- CID windows ----
    # Overlapping CID windows expose local event context without exploding the
    # configuration space during the nested search.
    ws = 10
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
    

# -------------------------
# 2) LLM/RAG candidate grid
# -------------------------

def make_candidates(*, use_llm_fallback: bool) -> List[Candidate]:
    """Construct the restricted candidate grid for retrieval and fallback.

    The search space is deliberately small because each candidate is evaluated
    repeatedly across outer splits and LLM fallback can add API cost.
    """
    candidates: List[Candidate] = []

    for bundle_size in [10]:
        for per_class_k in [5]:
            for margin in ([0.0] if use_llm_fallback else [0.0]):
                for agg in ["mean"]:
                    candidates.append(
                        Candidate(
                            cfg=RAGLLMConfig(
                                # Bundle adjacent events so retrieval operates on
                                # short local contexts rather than single lines.
                                bundle_size=bundle_size,
                                bundle_strategy="fixed",
                                sliding_stride=max(1, bundle_size // 2),
                                drop_last_incomplete=True,

                                # Retrieval stays local-first; only uncertain
                                # cases may escalate to the chat model.
                                per_class_k=per_class_k,
                                max_chars_per_retrieved=1000,

                                # Use the deterministic numpy backend here to
                                # avoid dependency-sensitive benchmark drift.
                                retrieval_backend="numpy",
                                faiss_hnsw_m=32,

                                # Local embeddings drive the primary retrieval step.
                                local_embedding_model="BAAI/bge-base-en-v1.5",
                                local_embedding_batch_size=32,
                                local_normalize_embeddings=True,

                                # Batch query embeddings to keep inference cost stable.
                                predict_embedding_batch_size=64,

                                # The margin defines when retrieval is considered
                                # too uncertain and LLM fallback is allowed.
                                use_llm_fallback=use_llm_fallback,
                                llm_uncertainty_margin=float(margin),
                                score_agg=agg,

                                # Deterministic fallback settings keep repeated
                                # outer-split comparisons reproducible.
                                chat_model="gpt-4.1-mini",
                                temperature=0.0,
                                max_output_tokens=30,

                                seed=42,
                            )
                        )
                    )
    return candidates


# -------------------------
# 3) Main
# -------------------------
def main():
    """Run nested model selection and export one result row per outer split.

    Selection is based strictly on validation performance; the corresponding test
    metrics are recorded only for the chosen configuration in each outer split.
    """
    args = parse_args()
    metric = args.metric
    out_csv = _resolve_out_csv(args.out_csv)
    use_llm_fallback = bool(args.use_llm_fallback)

    all_outer_splits = make_val_test_splits(args.dataset)
    outer_splits = all_outer_splits
    if args.limit_outer and args.limit_outer > 0:
        outer_splits = outer_splits[: args.limit_outer]

    load_grid = make_load_configs(args.dataset)
    cand_grid = make_candidates(use_llm_fallback=use_llm_fallback)

    print(f"Dataset     : {args.dataset}")
    print(f"Outer splits: {len(outer_splits)} (of {len(all_outer_splits)})")
    print(f"LoadConfigs : {len(load_grid)}")
    print(f"Candidates  : {len(cand_grid)}")
    print(f"Metric      : {metric}")
    print(f"use_llm_fallback: {use_llm_fallback}")
    print(f"Writing CSV : {out_csv}")

    rows: List[Dict[str, object]] = []

    # ---- Outer evaluation loop ----
    for outer_i, (val_groups, test_groups) in enumerate(outer_splits, 1):
        print("\n" + "=" * 100)
        print(f"[OUTER {outer_i:03d}/{len(outer_splits)}] val={val_groups} test={test_groups}")
        print("=" * 100)

        # best_overall = (val_metric, NamedLoad, best_candidate, best_val_res, best_test_res, (n_train,n_val,n_test))
        best_overall = None

        # Each load configuration is evaluated under the same human/AI group
        # assignment so preprocessing is selected within the nested protocol.
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

            # `make_val_test_splits` defines the outer human/AI groups; this call
            # projects those group choices onto the example-level indices.
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

            # Skip degenerate outer folds so validation-based selection is not
            # driven by very small sample counts.
            min_train = 400
            min_val = 100
            min_test = 100
            if n_train < min_train or n_val < min_val or n_test < min_test:
                print(f"  ⚠ Too small split train={n_train} val={n_val} test={n_test}. Skipping.")
                continue

            # Nested evaluation constraint: search on validation only and expose
            # test performance only for the configuration selected in this fold.
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

            # Selection is based only on validation performance to keep the test
            # set untouched until after model/configuration choice.
            if best_overall is None or val_metric > best_overall[0]:
                best_overall = (val_metric, named, best_cand, best_val_res, best_test_res, (n_train, n_val, n_test))

        if best_overall is None:
            print("⚠ No valid result for this outer split.")
            continue

        val_metric, named, best_cand, best_val_res, best_test_res, (n_train, n_val, n_test) = best_overall

        # ---- Persist selected outer-fold result ----
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
            "selected_rag_cfg": repr(best_cand.cfg),
            "use_llm_fallback": int(use_llm_fallback),

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
        print(f"    RAG cfg   : {best_cand.cfg}")
        print(f"    VAL  {metric}={row['selection_val_score']:.4f} | TEST {metric}={row['selection_test_score']:.4f}")

        rows.append(row)

    # -------------------------
    # Write CSV
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
    # Quick summary
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
