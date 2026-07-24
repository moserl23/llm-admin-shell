
"""Nested TF-IDF benchmark runner for cross-group log classification.

The script evaluates one model family across predefined load configurations and
outer validation/test splits, with optional actor-label randomization for null
hypothesis experiments.
"""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import argparse

from src.core.shared.loader import load_examples, LoadConfig, get_num_actor_label_assignments
from src.core.ml.splits import make_splits
from src.core.ml.val_test_combs import make_val_test_splits
from src.core.ml.benchmark import bench

from src.ml_pipelines.tfidf_pipeline import Candidate, VectorizerConfig, search

def resolve_log_files(dataset: str, log_type: str) -> Tuple[str, ...]:
    """Map a dataset and logical source name to the concrete log file(s).

    The mapping is dataset-specific because not every corpus exposes the same
    sources. Returns the tuple consumed by ``LoadConfig``.
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

def parse_args():
    """Parse the command-line interface for the nested TF-IDF benchmark.

    The flags control dataset selection, model family, optional null-label
    randomization, and parallel execution settings.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="Nextcloud",
        choices=["Nextcloud", "WordPress", "Data", "Data_WP"],
        help="Which aggregated dataset root to use. Preferred names: Nextcloud, WordPress. Legacy aliases: Data, Data_WP.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "dummy_most_frequent",
            "dummy_stratified",
            "svm",
            "logreg",
            "sgd_hinge",
            "sgd_log",
            "pa_like",
            "ridge",
            "mnb",
            "cnb",
            "bnb",
        ],
        help="Model family to evaluate (one at a time)."
    )

    parser.add_argument(
        "--log_type",
        type=str,
        required=True,
        choices=["audit", "syslog", "nextcloud"],
        help="Logical log source to evaluate. "
             "For dataset=Nextcloud: audit, syslog, nextcloud. "
             "For dataset=WordPress: audit, syslog.",
    )

    parser.add_argument("--out_csv", type=str, default="results/tfidf_360_nested_results.csv")
    parser.add_argument("--limit_outer", type=int, default=0, help="If >0, run only first N outer splits (debug).")
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of outer splits to run in parallel. Use 1 to keep serial behavior.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Print timing for critical sections (load_examples, search).",
    )

    parser.add_argument(
        "--randomize_actor_labels",
        action="store_true",
        help="Enable enumerated actor-level label reassignment for the null hypothesis.",    
    )

    parser.add_argument(
        "--assignment_idx",
        type=int,
        default=None,
        help="Index of the enumerated actor-label assignment to use when randomize_actor_labels is enabled.",
    )

    return parser.parse_args()


# -------------------------
# Load configuration grid
# -------------------------
@dataclass(frozen=True)
class NamedLoad:
    """Named wrapper for a loader configuration used in the outer search loop."""
    name: str
    cfg: LoadConfig


def make_load_configs(
    dataset: str,
    log_type: str,
    *,
    randomize_actor_labels: bool = False,
    assignment_idx: Optional[int] = None,
) -> List[NamedLoad]:
    """Build the preprocessing and windowing configurations to compare.

    Audit logs are evaluated with CID-based windows, while the other sources use
    line-level views. Null-label settings are forwarded unchanged to each config.
    """
    base = dict(
        dataset=dataset,
        log_files=resolve_log_files(dataset, log_type),
        prefix_with_log_type=False,
        max_lines_per_file=None,
        randomize_actor_labels=randomize_actor_labels,
        assignment_idx=assignment_idx,
    )

    out: List[NamedLoad] = []

    preprocess_mode = "soft"

    if log_type == "audit":
        configs = [
            (30, 30, "CID"),
            (30, 15, "CID"),
            (40, 20, "CID"),
            (20, 10, "CID"),
        ]

        for ws, st, prefix in configs:
            out.append(NamedLoad(
                name=f"cids_ws{ws}_st{st}_{preprocess_mode}_{'cid' if prefix else 'num'}",
                cfg=LoadConfig(
                    **base,
                    preprocess_mode=preprocess_mode,
                    window_mode="cids",
                    window_size=ws,
                    window_stride=st,
                    window_drop_last=True,
                    cid_prefix=prefix,
                ),
            ))

        return out

    # Keep a per-line baseline in addition to short context windows.
    out.append(NamedLoad(
        name=f"none_{preprocess_mode}",
        cfg=LoadConfig(
            **base,
            preprocess_mode=preprocess_mode,
            window_mode="none",
        ),
    ))

    # Overlapping windows preserve local context without collapsing entire files.
    for ws in [1, 2, 4]:
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

    return out


# -------------------------
# Candidate grid
# -------------------------
def make_candidates() -> List[Candidate]:
    """Construct the TF-IDF/model search space for one outer split.

    The grid intentionally mixes strong character-level baselines with smaller
    word-level variants and lightweight dummy references.
    """
    candidates: List[Candidate] = []

    # Dummy baselines share a small vectorizer even though the estimator ignores it.
    dummy_vec_cfg = VectorizerConfig(
        analyzer="word",
        ngram_range=(1, 1),
        min_df=1,
        max_df=1.0,
        sublinear_tf=False,
        lowercase=True,
        max_features=1000,
        binary=False,
    )
    candidates.append(Candidate(
        vectorizer=dummy_vec_cfg,
        model_name="dummy_most_frequent",
        model_params={},
    ))
    candidates.append(Candidate(
        vectorizer=dummy_vec_cfg,
        model_name="dummy_stratified",
        model_params={},
    ))

    # Character n-grams are often the strongest default for heterogeneous log text.
    for ngram in [(3,5), (4,6), (5,7)]:
        for min_df in [2]:
            vec_cfg = VectorizerConfig(
                analyzer="char",
                ngram_range=ngram,
                min_df=min_df,
                max_df=0.95,
                sublinear_tf=True,
                lowercase=False,
                max_features=200_000,
                binary=False,
            )

            for C in [0.3, 1.0, 3.0]:
                candidates.append(Candidate(
                    vectorizer=vec_cfg,
                    model_name="svm",
                    model_params={"C": C, "class_weight": "balanced"},
                ))

            for C in [0.3, 1.0, 3.0]:
                candidates.append(Candidate(
                    vectorizer=vec_cfg,
                    model_name="logreg",
                    model_params={"C": C, "class_weight": "balanced"},
                ))

            for alpha in [1e-5, 1e-4]:
                candidates.append(Candidate(
                    vectorizer=vec_cfg,
                    model_name="sgd_hinge",
                    model_params={
                        "alpha": alpha,
                        "class_weight": "balanced",
                        "max_iter": 5000,
                        "tol": 1e-3,
                    },
                ))

            for alpha in [1e-5, 1e-4]:
                candidates.append(Candidate(
                    vectorizer=vec_cfg,
                    model_name="sgd_log",
                    model_params={
                        "alpha": alpha,
                        "class_weight": "balanced",
                        "max_iter": 5000,
                        "tol": 1e-3,
                    },
                ))

            candidates.append(Candidate(
                vectorizer=vec_cfg,
                model_name="pa_like",
                model_params={},
            ))

            for alpha in [1.0, 10.0, 100.0]:
                candidates.append(Candidate(
                    vectorizer=vec_cfg,
                    model_name="ridge",
                    model_params={"alpha": alpha},
                ))

            for alpha in [0.01, 0.1, 1.0]:
                candidates.append(Candidate(
                    vectorizer=vec_cfg,
                    model_name="mnb",
                    model_params={"alpha": alpha},
                ))
                candidates.append(Candidate(
                    vectorizer=vec_cfg,
                    model_name="cnb",
                    model_params={"alpha": alpha},
                ))
                candidates.append(Candidate(
                    vectorizer=vec_cfg,
                    model_name="bnb",
                    model_params={"alpha": alpha},
                ))

    # Word features remain useful as a complementary baseline for cleaner sources.
    for ngram in [(1, 1), (1, 2)]:
        for min_df in [2]:
            vec_cfg = VectorizerConfig(
                analyzer="word",
                ngram_range=ngram,
                min_df=min_df,
                max_df=0.95,
                sublinear_tf=True,
                lowercase=True,
                max_features=200_000,
                binary=False,
            )

            for C in [0.3, 1.0, 3.0]:
                candidates.append(Candidate(
                    vectorizer=vec_cfg,
                    model_name="logreg",
                    model_params={"C": C, "class_weight": "balanced"},
                ))

            for C in [0.3, 1.0, 3.0]:
                candidates.append(Candidate(
                    vectorizer=vec_cfg,
                    model_name="svm",
                    model_params={"C": C, "class_weight": "balanced"},
                ))

            for alpha in [1e-5, 1e-4]:
                candidates.append(Candidate(
                    vectorizer=vec_cfg,
                    model_name="sgd_hinge",
                    model_params={
                        "alpha": alpha,
                        "class_weight": "balanced",
                        "max_iter": 5000,
                        "tol": 1e-3,
                    },
                ))

            for alpha in [1e-5, 1e-4]:
                candidates.append(Candidate(
                    vectorizer=vec_cfg,
                    model_name="sgd_log",
                    model_params={
                        "alpha": alpha,
                        "class_weight": "balanced",
                        "max_iter": 5000,
                        "tol": 1e-3,
                    },
                ))

            candidates.append(Candidate(
                vectorizer=vec_cfg,
                model_name="pa_like",
                model_params={},
            ))

            for alpha in [1.0, 10.0, 100.0]:
                candidates.append(Candidate(
                    vectorizer=vec_cfg,
                    model_name="ridge",
                    model_params={"alpha": alpha},
                ))

            for alpha in [0.01, 0.1, 1.0]:
                candidates.append(Candidate(
                    vectorizer=vec_cfg,
                    model_name="mnb",
                    model_params={"alpha": alpha},
                ))
                candidates.append(Candidate(
                    vectorizer=vec_cfg,
                    model_name="cnb",
                    model_params={"alpha": alpha},
                ))

    # BernoulliNB is evaluated separately with binary indicators rather than TF-IDF.
    for ngram in [(1, 2)]:
        for min_df in [2]:
            vec_cfg_bin = VectorizerConfig(
                analyzer="word",
                ngram_range=ngram,
                min_df=min_df,
                max_df=0.95,
                sublinear_tf=False,   # keep counts/binary sane
                lowercase=True,
                max_features=200_000,
                binary=True,
            )

            for alpha in [0.1, 1.0]:
                candidates.append(Candidate(
                    vectorizer=vec_cfg_bin,
                    model_name="bnb",
                    model_params={"alpha": alpha},
                ))

    return candidates


def _safe_float(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def _resolve_out_csv(path: str) -> str:
    """Normalize the CSV destination and ensure the parent directory exists."""
    p = Path(path)
    if str(p.parent) == ".":
        p = Path("results") / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _run_one_outer_split(
    outer_i: int,
    total_outer: int,
    val_groups: Tuple[str, str],
    test_groups: Tuple[str, str],
    *,
    model_name: str,
    metric: str,
    dataset: str,
    log_type: str,
    benchmark: bool,
    randomize_actor_labels: bool,
    assignment_idx: Optional[int],
) -> Optional[Dict[str, object]]:
    """Run model selection and evaluation for one outer split.

    Each outer split searches over load configurations on the validation groups,
    then reports the corresponding test result for the best validation setting.
    Returns one CSV row, or ``None`` when no valid configuration survives.
    """
    print("\n" + "=" * 100)
    print(f"[OUTER {outer_i:03d}/{total_outer}] val={val_groups} test={test_groups}")
    print("=" * 100)

    load_grid = make_load_configs(
        dataset,
        log_type,
        randomize_actor_labels=randomize_actor_labels,
        assignment_idx=assignment_idx,
    )
    cand_grid = [c for c in make_candidates() if c.model_name == model_name]
    best_overall = None  # (val_score, NamedLoad, best_candidate, best_val_res, best_test_res, counts)

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

        # Validation and test groups are fixed at the outer level to prevent
        # tuning decisions from leaking across actor partitions.
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

        # These thresholds keep fragile tiny splits out of the comparison while
        # still allowing the smaller syslog partitions to remain evaluable.
        min_train = 200
        min_val = 100
        min_test = 100
        if n_train < min_train or n_val < min_val or n_test < min_test:
            print(f"  ⚠ Too small split train={n_train} val={n_val} test={n_test}. Skipping.")
            continue

        with bench(benchmark, f"search({named.name})"):
            best_cand, best_val_res, best_test_res, _all_val = search(
                examples,
                split,
                cand_grid,
                metric=metric,
                evaluate_test_for_all=False,
                verbose=False,
            )

        # Outer-model selection is based only on validation performance; the
        # paired test score is carried along for the final unbiased summary.
        val_f1 = _safe_float(getattr(best_val_res, "f1_macro", np.nan))
        val_bal = _safe_float(getattr(best_val_res, "balanced_accuracy", np.nan))
        test_f1 = _safe_float(getattr(best_test_res, "f1_macro", np.nan))
        test_bal = _safe_float(getattr(best_test_res, "balanced_accuracy", np.nan))

        print(f"  best VAL f1_macro={val_f1:.4f} | TEST f1_macro={test_f1:.4f} | model={best_cand.model_name}")

        if best_overall is None or val_f1 > best_overall[0]:
            best_overall = (
                val_f1,
                named,
                best_cand,
                best_val_res,
                best_test_res,
                (n_train, n_val, n_test),
                (val_bal, test_bal),
            )

    if best_overall is None:
        print("⚠ No valid result for this outer split.")
        return None

    val_f1, named, best_cand, best_val_res, best_test_res, (n_train, n_val, n_test), (val_bal, test_bal) = best_overall
    test_f1 = _safe_float(getattr(best_test_res, "f1_macro", np.nan))

    print("\n>>> SELECTED (by VAL f1_macro)")
    print(f"    LoadConfig: {named.name}")
    print(f"    Candidate : {best_cand}")
    print(f"    VAL  f1_macro={val_f1:.4f}  bal_acc={val_bal:.4f}")
    print(f"    TEST f1_macro={test_f1:.4f}  bal_acc={test_bal:.4f}")

    return {
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
        "selected_vectorizer": repr(getattr(best_cand, "vectorizer", None)),
        "val_f1_macro": _safe_float(getattr(best_val_res, "f1_macro", np.nan)),
        "val_balanced_accuracy": _safe_float(getattr(best_val_res, "balanced_accuracy", np.nan)),
        "test_f1_macro": _safe_float(getattr(best_test_res, "f1_macro", np.nan)),
        "test_balanced_accuracy": _safe_float(getattr(best_test_res, "balanced_accuracy", np.nan)),
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
        "randomize_actor_labels": randomize_actor_labels,
        "assignment_idx": assignment_idx,
    }


def main():
    """Execute the full nested benchmark and write one row per outer split.

    The runner supports the standard evaluation setting and a null setting in
    which actor labels are reassigned via a fixed enumerated permutation.
    """
    args = parse_args()

    # Null-label runs must use an explicit permutation index so the evaluation
    # is reproducible and can be matched across models.
    if args.randomize_actor_labels and args.assignment_idx is None:
        raise ValueError(
            "--randomize_actor_labels requires --assignment_idx"
        )

    if (not args.randomize_actor_labels) and (args.assignment_idx is not None):
        raise ValueError(
            "--assignment_idx should only be used together with --randomize_actor_labels"
        )
    
    if args.randomize_actor_labels:
        n_assignments = get_num_actor_label_assignments(args.dataset)
        if not (0 <= args.assignment_idx < n_assignments):
            raise ValueError(
                f"--assignment_idx={args.assignment_idx} out of range for dataset={args.dataset}. "
                f"Valid range: 0..{n_assignments - 1}"
            )

    model_name = args.model
    metric = "f1_macro"
    out_csv = _resolve_out_csv(args.out_csv)
    n_jobs = max(1, int(args.n_jobs))

    # Outer splits encode the human/AI group pairings used for held-out
    # evaluation, optionally under a randomized actor-label assignment.
    all_outer_splits = make_val_test_splits(
        args.dataset,
        randomize_actor_labels=args.randomize_actor_labels,
        assignment_idx=args.assignment_idx,
    )
    outer_splits = all_outer_splits
    if args.limit_outer and args.limit_outer > 0:
        outer_splits = outer_splits[: args.limit_outer]

    load_grid = make_load_configs(
        args.dataset,
        args.log_type,
        randomize_actor_labels=args.randomize_actor_labels,
        assignment_idx=args.assignment_idx,
    )
    cand_grid = [c for c in make_candidates() if c.model_name == model_name]

    if not cand_grid:
        raise RuntimeError(f"No candidates for model {model_name}")
    
    print(f"Dataset     : {args.dataset}")
    print(f"Log type    : {args.log_type}")
    print("MODEL:", model_name)
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
                model_name=model_name,
                metric=metric,
                dataset=args.dataset,
                log_type=args.log_type,
                benchmark=args.benchmark,
                randomize_actor_labels=args.randomize_actor_labels,
                assignment_idx=args.assignment_idx,
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
                    model_name=model_name,
                    metric=metric,
                    dataset=args.dataset,
                    log_type=args.log_type,
                    benchmark=args.benchmark,
                    randomize_actor_labels=args.randomize_actor_labels,
                    assignment_idx=args.assignment_idx,
                )
                for outer_i, total_outer, val_groups, test_groups in worker_args
            ]
            for fut in as_completed(futures):
                row = fut.result()
                if row is not None:
                    rows.append(row)

    # -------------------------
    # Write results
    # -------------------------
    if not rows:
        print("\nNo rows collected; nothing to write.")
        return

    rows.sort(key=lambda r: int(r["outer_i"]))

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
