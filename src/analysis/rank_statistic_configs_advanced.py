#!/usr/bin/env python3
"""Rank candidate statistic configurations across one or more result CSV files.

The script applies a configurable hard filter, removes globally duplicated metric
profiles, and reports the strongest configurations using multi-metric ranking.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _to_float(value: str | None) -> float:
    """Parse a CSV cell as float and return `nan` for missing or invalid values.

    This keeps downstream ranking logic numeric while treating malformed entries
    as unavailable rather than failing the full analysis.
    """
    if value is None:
        return float("nan")
    text = value.strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _load_json_dict(raw: str | None) -> dict[str, Any]:
    """Load a JSON object from a CSV cell and fall back to an empty dict.

    Non-dict JSON payloads are ignored because hyperparameters are expected to be
    stored as key-value mappings.
    """
    if raw is None:
        return {}
    text = raw.strip()
    if not text:
        return {}
    value = json.loads(text)
    return value if isinstance(value, dict) else {}


def _is_finite(value: float) -> bool:
    """Return whether a numeric value is finite."""
    return math.isfinite(value)


def _float_equal(a: float, b: float, *, rel_tol: float = 1e-12, abs_tol: float = 1e-12) -> bool:
    """Compare finite floats with a tight tolerance for rank tie handling.

    Non-finite values are treated as unequal because they are excluded from the
    ranking calculations rather than assigned tied positions.
    """
    if not (_is_finite(a) and _is_finite(b)):
        return False
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def _safe_div(numerator: float, denominator: float) -> float:
    """Divide two finite values and return `nan` when the ratio is undefined."""
    if not (_is_finite(numerator) and _is_finite(denominator)):
        return float("nan")
    if denominator == 0.0:
        return float("nan")
    return numerator / denominator


def _compute_scores(row: dict[str, str]) -> dict[str, float]:
    """Derive ranking metrics from one statistics row.

    The derived scores emphasize separation between AI-human and within-group
    means, plus overall cluster quality and effect-size strength.
    """
    ai_ai_mean = _to_float(row.get("ai_ai_mean"))
    ai_human_mean = _to_float(row.get("ai_human_mean"))
    human_human_mean = _to_float(row.get("human_human_mean"))
    silhouette_overall = _to_float(row.get("silhouette_overall_mean"))
    cliffs_ai_ai = _to_float(row.get("mw_ai_human_vs_ai_ai_cliffs_delta"))
    cliffs_human_human = _to_float(row.get("mw_ai_human_vs_human_human_cliffs_delta"))

    mean_separation = float("nan")
    normalized_mean_separation = float("nan")
    if all(_is_finite(v) for v in (ai_ai_mean, ai_human_mean, human_human_mean)):
        max_within = max(ai_ai_mean, human_human_mean)
        mean_separation = ai_human_mean - max_within
        normalized_mean_separation = _safe_div(mean_separation, max_within)

    cliffs_delta_mean = float("nan")
    cliffs_delta_min = float("nan")
    if all(_is_finite(v) for v in (cliffs_ai_ai, cliffs_human_human)):
        cliffs_delta_mean = (cliffs_ai_ai + cliffs_human_human) / 2.0
        cliffs_delta_min = min(cliffs_ai_ai, cliffs_human_human)

    return {
        "mean_separation": mean_separation,
        "normalized_mean_separation": normalized_mean_separation,
        "silhouette_overall": silhouette_overall,
        "cliffs_delta_mean": cliffs_delta_mean,
        "cliffs_delta_min": cliffs_delta_min,
    }


def _load_rows(csv_path: Path) -> list[dict[str, Any]]:
    """Load one statistics CSV and attach parsed fields plus derived scores.

    Each returned row keeps the original summary values alongside normalized
    metadata needed for filtering, deduplication, and reporting.
    """
    rows: list[dict[str, Any]] = []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            scores = _compute_scores(raw_row)
            hyperparameters = _load_json_dict(raw_row.get("hyperparameters_json"))
            rows.append(
                {
                    "source_csv": str(csv_path),
                    "timestamp_utc": raw_row.get("timestamp_utc", ""),
                    "approach": raw_row.get("approach", ""),
                    "distance_name": raw_row.get("distance_name", ""),
                    "hyperparameters": hyperparameters,
                    "scores": scores,
                    "ai_ai_mean": _to_float(raw_row.get("ai_ai_mean")),
                    "ai_human_mean": _to_float(raw_row.get("ai_human_mean")),
                    "human_human_mean": _to_float(raw_row.get("human_human_mean")),
                    "silhouette_overall_mean": _to_float(raw_row.get("silhouette_overall_mean")),
                    "mw_ai_human_vs_ai_ai_cliffs_delta": _to_float(
                        raw_row.get("mw_ai_human_vs_ai_ai_cliffs_delta")
                    ),
                    "mw_ai_human_vs_human_human_cliffs_delta": _to_float(
                        raw_row.get("mw_ai_human_vs_human_human_cliffs_delta")
                    ),
                    "mw_ai_human_vs_ai_ai_p": _to_float(
                        raw_row.get("mw_ai_human_vs_ai_ai_p")
                    ),
                    "mw_ai_human_vs_human_human_p": _to_float(
                        raw_row.get("mw_ai_human_vs_human_human_p")
                    ),
                }
            )

    return rows


def _format_hyperparameters(hyperparameters: dict[str, Any]) -> str:
    """Render hyperparameters in a stable order for readable console output."""
    if not hyperparameters:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in sorted(hyperparameters.items()))


def _format_float_or_na(value: float, fmt: str = ".6f") -> str:
    """Format a finite float or return `nan` for missing numeric results."""
    if not _is_finite(value):
        return "nan"
    return format(value, fmt)


def _passes_hard_filter(
    row: dict[str, Any],
    *,
    require_significant: bool,
    alpha: float,
) -> bool:
    """Check whether a row satisfies the minimum quality constraints.

    The filter keeps only configurations where the between-group mean exceeds
    both within-group means and the supporting separation statistics are positive.
    """
    ai_ai_mean = row["ai_ai_mean"]
    ai_human_mean = row["ai_human_mean"]
    human_human_mean = row["human_human_mean"]
    silhouette = row["silhouette_overall_mean"]
    cliff_1 = row["mw_ai_human_vs_ai_ai_cliffs_delta"]
    cliff_2 = row["mw_ai_human_vs_human_human_cliffs_delta"]
    p_1 = row["mw_ai_human_vs_ai_ai_p"]
    p_2 = row["mw_ai_human_vs_human_human_p"]

    required = (
        ai_ai_mean,
        ai_human_mean,
        human_human_mean,
        silhouette,
        cliff_1,
        cliff_2,
    )
    if not all(_is_finite(v) for v in required):
        return False

    # Require the cross-group signal to dominate both within-group baselines.
    if not (ai_human_mean > ai_ai_mean and ai_human_mean > human_human_mean):
        return False
    if not (silhouette > 0.0):
        return False
    if not (cliff_1 > 0.0 and cliff_2 > 0.0):
        return False

    if require_significant:
        if not (_is_finite(p_1) and _is_finite(p_2)):
            return False
        if not (p_1 < alpha and p_2 < alpha):
            return False

    return True


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    require_significant: bool,
    alpha: float,
    use_hard_filter: bool,
) -> list[dict[str, Any]]:
    """Apply the optional hard filter and return the surviving rows."""
    if not use_hard_filter:
        return rows[:]
    return [
        row
        for row in rows
        if _passes_hard_filter(
            row,
            require_significant=require_significant,
            alpha=alpha,
        )
    ]


def _dedup_key_from_row(
    row: dict[str, Any],
    *,
    digits: int,
) -> tuple[float | str, float | str, float | str]:
    """Build the rounded metric tuple used for global duplicate removal.

    Deduplication is based on derived ranking behavior rather than raw metadata,
    so equivalent score profiles collapse even across different source files.
    """
    metric_names = (
        "normalized_mean_separation",
        "silhouette_overall",
        "cliffs_delta_mean",
    )

    key: list[float | str] = []
    for metric_name in metric_names:
        value = row["scores"][metric_name]
        if _is_finite(value):
            key.append(round(value, digits))
        else:
            key.append("nan")

    return tuple(key)


def _remove_duplicate_rows(
    rows: list[dict[str, Any]],
    *,
    digits: int = 12,
) -> tuple[list[dict[str, Any]], int]:
    """Drop rows with identical rounded metric tuples and report how many were removed."""
    seen: set[tuple[float | str, float | str, float | str]] = set()
    unique_rows: list[dict[str, Any]] = []

    for row in rows:
        key = _dedup_key_from_row(row, digits=digits)
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)

    removed_count = len(rows) - len(unique_rows)
    return unique_rows, removed_count


def _minmax_normalize(rows: list[dict[str, Any]], metric_name: str) -> None:
    """Min-max normalize one metric in place across the current candidate set.

    When all finite values are identical, each finite row receives 1.0 so the
    metric remains neutral rather than forcing an arbitrary spread.
    """
    values = [
        row["scores"][metric_name]
        for row in rows
        if _is_finite(row["scores"][metric_name])
    ]
    if not values:
        for row in rows:
            row["scores"][f"{metric_name}_norm"] = float("nan")
        return

    min_value = min(values)
    max_value = max(values)

    if max_value == min_value:
        for row in rows:
            if _is_finite(row["scores"][metric_name]):
                row["scores"][f"{metric_name}_norm"] = 1.0
            else:
                row["scores"][f"{metric_name}_norm"] = float("nan")
        return

    for row in rows:
        value = row["scores"][metric_name]
        if _is_finite(value):
            row["scores"][f"{metric_name}_norm"] = (value - min_value) / (max_value - min_value)
        else:
            row["scores"][f"{metric_name}_norm"] = float("nan")


def _assign_descending_ranks(rows: list[dict[str, Any]], metric_name: str) -> None:
    """Assign competition ranks for one metric, with larger values ranked higher.

    Equal values receive the same rank, and the next distinct value skips ahead
    accordingly to preserve standard ranking semantics.
    """
    ranked = [
        row for row in rows
        if _is_finite(row["scores"][metric_name])
    ]
    ranked.sort(key=lambda row: row["scores"][metric_name], reverse=True)

    for row in rows:
        row["scores"][f"{metric_name}_rank"] = float("nan")

    if not ranked:
        return

    current_rank = 1
    previous_value: float | None = None

    for index, row in enumerate(ranked, start=1):
        value = row["scores"][metric_name]

        if previous_value is None:
            current_rank = 1
        elif not _float_equal(value, previous_value):
            current_rank = index

        row["scores"][f"{metric_name}_rank"] = float(current_rank)
        previous_value = value


def _compute_composite_scores(rows: list[dict[str, Any]]) -> None:
    """Compute the weighted composite score from normalized metric values.

    This score mixes raw metric scales after normalization and is retained as a
    secondary view beside the rank-based recommendation.
    """
    _minmax_normalize(rows, "normalized_mean_separation")
    _minmax_normalize(rows, "silhouette_overall")
    _minmax_normalize(rows, "cliffs_delta_mean")

    for row in rows:
        sep = row["scores"]["normalized_mean_separation_norm"]
        sil = row["scores"]["silhouette_overall_norm"]
        cliff = row["scores"]["cliffs_delta_mean_norm"]

        if all(_is_finite(v) for v in (sep, sil, cliff)):
            row["scores"]["composite_score"] = 0.2 * sep + 0.3 * sil + 0.5 * cliff
        else:
            row["scores"]["composite_score"] = float("nan")


def _compute_rank_aggregation(rows: list[dict[str, Any]]) -> None:
    """Compute the primary rank-aggregation score across the three core metrics.

    Lower values are better because the score is a weighted average of ranks
    rather than normalized raw values.
    """
    _assign_descending_ranks(rows, "normalized_mean_separation")
    _assign_descending_ranks(rows, "silhouette_overall")
    _assign_descending_ranks(rows, "cliffs_delta_mean")

    for row in rows:
        sep_rank = row["scores"]["normalized_mean_separation_rank"]
        sil_rank = row["scores"]["silhouette_overall_rank"]
        cliff_rank = row["scores"]["cliffs_delta_mean_rank"]

        if all(_is_finite(v) for v in (sep_rank, sil_rank, cliff_rank)):
            row["scores"]["rank_aggregate_score"] = (
                0.2 * sep_rank + 0.3 * sil_rank + 0.5 * cliff_rank
            )
        else:
            row["scores"]["rank_aggregate_score"] = float("nan")


def _rank_rows(
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    top_k: int,
    reverse: bool = True,
) -> list[dict[str, Any]]:
    """Return the top finite rows for one score, ordered for display."""
    ranked = [row for row in rows if _is_finite(row["scores"][metric_name])]
    ranked.sort(key=lambda row: row["scores"][metric_name], reverse=reverse)
    return ranked[:top_k]


def _print_row_details(
    row: dict[str, Any],
    *,
    display_score_name: str,
) -> None:
    """Print the detailed metrics associated with one ranked configuration."""
    print(f"   source_csv: {row['source_csv']}")
    print(
        f"   means: ai_ai={_format_float_or_na(row['ai_ai_mean'])}, "
        f"ai_human={_format_float_or_na(row['ai_human_mean'])}, "
        f"human_human={_format_float_or_na(row['human_human_mean'])}"
    )
    print(
        f"   derived: mean_sep={_format_float_or_na(row['scores']['mean_separation'])}, "
        f"norm_mean_sep={_format_float_or_na(row['scores']['normalized_mean_separation'])}, "
        f"silhouette={_format_float_or_na(row['silhouette_overall_mean'])}, "
        f"cliffs_mean={_format_float_or_na(row['scores']['cliffs_delta_mean'])}, "
        f"cliffs_min={_format_float_or_na(row['scores']['cliffs_delta_min'])}"
    )
    print(
        f"   cliffs=({_format_float_or_na(row['mw_ai_human_vs_ai_ai_cliffs_delta'])}, "
        f"{_format_float_or_na(row['mw_ai_human_vs_human_human_cliffs_delta'])})"
    )
    print(
        f"   mw_p=({_format_float_or_na(row['mw_ai_human_vs_ai_ai_p'])}, "
        f"{_format_float_or_na(row['mw_ai_human_vs_human_human_p'])})"
    )
    if display_score_name == "composite_score":
        print(
            f"   normalized: sep={_format_float_or_na(row['scores']['normalized_mean_separation_norm'])}, "
            f"sil={_format_float_or_na(row['scores']['silhouette_overall_norm'])}, "
            f"cliff={_format_float_or_na(row['scores']['cliffs_delta_mean_norm'])}"
        )
    if display_score_name == "rank_aggregate_score":
        print(
            f"   ranks: sep={_format_float_or_na(row['scores']['normalized_mean_separation_rank'], '.1f')}, "
            f"sil={_format_float_or_na(row['scores']['silhouette_overall_rank'], '.1f')}, "
            f"cliff={_format_float_or_na(row['scores']['cliffs_delta_mean_rank'], '.1f')}"
        )


def _print_metric_block(
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    label: str,
    description: str,
    top_k: int,
    reverse: bool = True,
) -> None:
    """Print a ranked result block for one metric or aggregate score."""
    print(f"\n=== {label} ===")
    print(description)

    ranked = _rank_rows(rows, metric_name=metric_name, top_k=top_k, reverse=reverse)
    if not ranked:
        print("No finite rows for this metric.")
        return

    for index, row in enumerate(ranked, start=1):
        print(
            f"{index}. score={_format_float_or_na(row['scores'][metric_name])} | "
            f"source={row['source_csv']} | "
            f"distance={row['distance_name']} | "
            f"hyperparameters: {_format_hyperparameters(row['hyperparameters'])}"
        )
        _print_row_details(row, display_score_name=metric_name)


def main() -> None:
    """Parse inputs, rank candidate configurations, and print the analysis summary."""
    parser = argparse.ArgumentParser(
        description=(
            "Rank hyperparameter configurations from one or more statistic CSV files using a "
            "hard filter, duplicate removal, composite score, and rank aggregation over "
            "normalized mean separation, silhouette, and Mann-Whitney effect size."
        )
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="Path(s) to one or more statistic CSV files.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top configurations to print (default: 10).",
    )
    parser.add_argument(
        "--no-hard-filter",
        action="store_true",
        help=(
            "Disable the hard filter. By default, rows must satisfy: "
            "ai_human > ai_ai, ai_human > human_human, silhouette > 0, "
            "and both Cliff's deltas > 0."
        ),
    )
    parser.add_argument(
        "--require-significant",
        action="store_true",
        help="Require both Mann-Whitney p-values to be below alpha in the hard filter.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for --require-significant (default: 0.05).",
    )
    parser.add_argument(
        "--show-individual-metrics",
        action="store_true",
        help="Also print rankings for the individual metrics.",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help=(
            "Disable duplicate removal. By default, rows with identical "
            "normalized_mean_separation, silhouette_overall, and cliffs_delta_mean "
            "are deduplicated globally across all input CSV files."
        ),
    )
    parser.add_argument(
        "--dedup-round-digits",
        type=int,
        default=12,
        help=(
            "Number of decimal digits used when comparing duplicate metric tuples "
            "(default: 12)."
        ),
    )
    args = parser.parse_args()

    # ---- Load data ----
    csv_paths = [Path(p) for p in args.csv_files]

    rows: list[dict[str, Any]] = []
    rows_per_file: dict[str, int] = {}

    for csv_path in csv_paths:
        file_rows = _load_rows(csv_path)
        rows_per_file[str(csv_path)] = len(file_rows)
        rows.extend(file_rows)

    if not rows:
        raise ValueError("No rows found in any provided CSV file.")

    approach_names = sorted({row["approach"] for row in rows if row["approach"]})

    # ---- Filter and deduplicate candidates ----
    filtered_rows = _filter_rows(
        rows,
        require_significant=args.require_significant,
        alpha=args.alpha,
        use_hard_filter=not args.no_hard_filter,
    )

    if not filtered_rows:
        print("CSV files:")
        for csv_path in csv_paths:
            print(f"  - {csv_path} ({rows_per_file[str(csv_path)]} rows)")
        print(f"Total rows: {len(rows)}")
        print(f"Approach: {', '.join(approach_names) if approach_names else '-'}")
        print("Filtered rows: 0")
        print("No configurations passed the current filter settings.")
        return

    deduplicated_rows = filtered_rows[:]
    duplicates_removed = 0
    if not args.no_dedup:
        deduplicated_rows, duplicates_removed = _remove_duplicate_rows(
            filtered_rows,
            digits=args.dedup_round_digits,
        )

    if not deduplicated_rows:
        print("CSV files:")
        for csv_path in csv_paths:
            print(f"  - {csv_path} ({rows_per_file[str(csv_path)]} rows)")
        print(f"Total rows: {len(rows)}")
        print(f"Approach: {', '.join(approach_names) if approach_names else '-'}")
        print(f"Filtered rows: {len(filtered_rows)}")
        print("Rows after deduplication: 0")
        print("No configurations remain after duplicate removal.")
        return

    # ---- Compute ranking scores ----
    _compute_composite_scores(deduplicated_rows)
    _compute_rank_aggregation(deduplicated_rows)

    # ---- Report results ----
    print("CSV files:")
    for csv_path in csv_paths:
        print(f"  - {csv_path} ({rows_per_file[str(csv_path)]} rows)")
    print(f"Total rows: {len(rows)}")
    print(f"Approach: {', '.join(approach_names) if approach_names else '-'}")
    print(f"Top k: {args.top_k}")
    print(f"Hard filter enabled: {not args.no_hard_filter}")
    print(f"Require significance: {args.require_significant}")
    print(f"Alpha: {args.alpha}")
    print(f"Filtered rows: {len(filtered_rows)}")
    print(f"Deduplication enabled: {not args.no_dedup}")
    print(f"Dedup round digits: {args.dedup_round_digits}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Rows after deduplication: {len(deduplicated_rows)}")

    print("\nRecommended primary ranking: Rank Aggregation")
    print(
        "score = 0.2 * rank(norm_mean_separation) + "
        "0.3 * rank(silhouette_overall) + "
        "0.5 * rank(cliffs_delta_mean), lower is better"
    )
    _print_metric_block(
        deduplicated_rows,
        metric_name="rank_aggregate_score",
        label="Top Configurations by Rank Aggregation",
        description=(
            "Primary recommendation. Lower is better. This is robust because it "
            "combines the three metrics by rank instead of mixing raw scales."
        ),
        top_k=args.top_k,
        reverse=False,
    )

    '''
    _print_metric_block(
        deduplicated_rows,
        metric_name="composite_score",
        label="Top Configurations by Weighted Composite Score",
        description=(
            "Secondary ranking. Higher is better. "
            "score = 0.2 * norm(norm_mean_separation) + "
            "0.3 * norm(silhouette_overall) + "
            "0.5 * norm(cliffs_delta_mean)"
        ),
        top_k=args.top_k,
        reverse=True,
    )
    '''

    if args.show_individual_metrics:
        _print_metric_block(
            deduplicated_rows,
            metric_name="normalized_mean_separation",
            label="Normalized Mean Separation",
            description=(
                "score = (ai_human_mean - max(ai_ai_mean, human_human_mean)) / "
                "max(ai_ai_mean, human_human_mean)"
            ),
            top_k=args.top_k,
            reverse=True,
        )
        _print_metric_block(
            deduplicated_rows,
            metric_name="silhouette_overall",
            label="Silhouette",
            description="score = silhouette_overall_mean",
            top_k=args.top_k,
            reverse=True,
        )
        _print_metric_block(
            deduplicated_rows,
            metric_name="cliffs_delta_mean",
            label="Mann-Whitney Effect Size",
            description=(
                "score = mean of "
                "(mw_ai_human_vs_ai_ai_cliffs_delta, "
                "mw_ai_human_vs_human_human_cliffs_delta)"
            ),
            top_k=args.top_k,
            reverse=True,
        )


if __name__ == "__main__":
    main()
