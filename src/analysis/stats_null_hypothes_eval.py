#!/usr/bin/env python3
"""Evaluate whether the best observed configuration exceeds a null distribution.

For each null CSV, the script keeps only the best-scoring row under a chosen
metric, compares that distribution against the best row from the correctly
labeled run, and reports an empirical p-value plus a compact visualization.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_CHOICES = ["silhouette", "cliffs", "norm_mean_diff"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for null-distribution evaluation and plotting.

    The interface separates null runs from the correctly labeled run and
    exposes plotting and empirical p-value options used in the analysis.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot null-hypothesis distributions for statistic CSVs by selecting "
            "the best row per CSV according to a chosen metric, then overlay the "
            "best row from the correctly labeled CSV."
        )
    )

    parser.add_argument(
        "--metric",
        required=True,
        choices=METRIC_CHOICES,
        help=(
            "Metric to optimize per CSV. "
            "'silhouette' uses silhouette_overall_mean. "
            "'cliffs' uses the mean of the two Cliff's delta columns. "
            "'norm_mean_diff' uses a normalized mean-difference / normalized mean-separation column."
        ),
    )
    parser.add_argument(
        "--null-dir",
        required=True,
        type=Path,
        help="Directory containing null-hypothesis CSV files.",
    )
    parser.add_argument(
        "--actual-csv",
        required=True,
        type=Path,
        help="CSV file for the correctly labeled run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the figure. If omitted, show interactively.",
    )
    parser.add_argument(
        "--glob",
        default="*.csv",
        help="Glob pattern for null CSV files inside --null-dir.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title.",
    )
    parser.add_argument(
        "--xlabel",
        default="Null distribution",
        help="Label for the x-axis category.",
    )
    parser.add_argument(
        "--ylabel",
        default=None,
        help="Optional x-axis label override for the metric scale.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(10.0, 3.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for jitter placement.",
    )
    parser.add_argument(
        "--two-sided",
        action="store_true",
        default=False,
        help="Use a two-sided empirical p-value based on deviation from null mean.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate that the required input paths exist and have the expected type.

    The script assumes a directory of null CSVs and a single observed CSV,
    so path errors are surfaced before any analysis starts.
    """
    if not args.null_dir.exists():
        raise FileNotFoundError(f"Null directory does not exist: {args.null_dir}")
    if not args.null_dir.is_dir():
        raise NotADirectoryError(f"Null path is not a directory: {args.null_dir}")
    if not args.actual_csv.exists():
        raise FileNotFoundError(f"Actual CSV does not exist: {args.actual_csv}")


def compute_metric_series(df: pd.DataFrame, metric: str) -> pd.Series:
    """Compute the ranking metric used to select the best row in one CSV.

    The derived metric must be comparable within a file; invalid or undefined
    values are coerced to `NaN` so they are excluded from row selection.
    """
    if metric == "silhouette":
        col = "silhouette_overall_mean"
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found.")
        return pd.to_numeric(df[col], errors="coerce")

    if metric == "cliffs":
        col1 = "mw_ai_human_vs_ai_ai_cliffs_delta"
        col2 = "mw_ai_human_vs_human_human_cliffs_delta"
        missing = [c for c in (col1, col2) if c not in df.columns]
        if missing:
            raise KeyError(f"Required Cliff's delta columns missing: {missing}")

        s1 = pd.to_numeric(df[col1], errors="coerce")
        s2 = pd.to_numeric(df[col2], errors="coerce")
        return (s1 + s2) / 2.0

    if metric == "norm_mean_diff":
        required = ["ai_ai_mean", "ai_human_mean", "human_human_mean"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(
                f"Required columns for 'norm_mean_diff' missing: {missing}"
            )

        ai_ai = pd.to_numeric(df["ai_ai_mean"], errors="coerce")
        ai_human = pd.to_numeric(df["ai_human_mean"], errors="coerce")
        human_human = pd.to_numeric(df["human_human_mean"], errors="coerce")

        # Compare the cross-group mean to the average within-group baseline and
        # normalize by the combined scale to keep values comparable across CSVs.
        within_mean = (ai_ai + human_human) / 2.0
        denom = ai_human + within_mean

        out = pd.Series(np.nan, index=df.index, dtype=float)
        valid = denom != 0
        out.loc[valid] = (ai_human.loc[valid] - within_mean.loc[valid]) / denom.loc[valid]
        return out

    raise ValueError(f"Unsupported metric: {metric}")


def extract_best_row_value(csv_path: Path, metric: str) -> Tuple[float, pd.Series]:
    """Return the highest valid metric value and its source row from one CSV.

    Rows with undefined metric values are ignored. The returned row is copied
    and annotated with the derived metric for downstream reporting.
    """
    df = pd.read_csv(csv_path)

    metric_values = compute_metric_series(df, metric)
    valid_mask = metric_values.notna()

    if not valid_mask.any():
        raise ValueError(f"No valid rows for metric '{metric}' in {csv_path}")

    valid_metric_values = metric_values[valid_mask]
    best_idx = valid_metric_values.idxmax()
    best_value = float(valid_metric_values.loc[best_idx])

    best_row = df.loc[best_idx].copy()
    best_row["_derived_metric_value"] = best_value
    return best_value, best_row


def collect_null_best_values(
    null_dir: Path,
    glob_pattern: str,
    metric: str,
) -> List[Tuple[Path, float, pd.Series]]:
    """Collect the best row from each null CSV under the chosen metric.

    Each null file contributes one value to the empirical null distribution,
    matching the same "best-of-file" selection rule used for the observed run.
    """
    csv_paths = sorted(null_dir.glob(glob_pattern))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in {null_dir} matching pattern '{glob_pattern}'"
        )

    results: List[Tuple[Path, float, pd.Series]] = []
    for csv_path in csv_paths:
        best_value, best_row = extract_best_row_value(csv_path, metric)
        results.append((csv_path, best_value, best_row))
    return results


def empirical_p_value(
    null_values: np.ndarray,
    observed: float,
    two_sided: bool = False,
) -> float:
    """Compute a smoothed empirical p-value from the null distribution.

    The one-sided test asks how often null values are at least as large as the
    observed value. The two-sided variant measures extremeness around the null mean.
    """
    n = len(null_values)
    if n == 0:
        raise ValueError("Null distribution is empty.")

    if two_sided:
        # Centering at the null mean gives a symmetric notion of extremeness for
        # the empirical two-sided test without assuming a parametric null model.
        center = float(np.mean(null_values))
        obs_dev = abs(observed - center)
        count = int(np.sum(np.abs(null_values - center) >= obs_dev))
        return (1 + count) / (n + 1)

    count = int(np.sum(null_values >= observed))
    return (1 + count) / (n + 1)


def metric_axis_label(metric: str, ylabel_override: str | None) -> str:
    """Return the axis label for the selected metric.

    A user-provided override takes precedence; otherwise a readable default is
    chosen so the plot remains interpretable without extra context.
    """
    if ylabel_override is not None:
        return ylabel_override
    if metric == "silhouette":
        return "Best silhouette overall mean per CSV"
    if metric == "cliffs":
        return "Best mean Cliff's delta per CSV"
    if metric == "norm_mean_diff":
        return "Best normalized mean difference per CSV"
    return metric


def metric_default_xlim(metric: str) -> tuple[float, float] | None:
    """Return default x-limits when the metric has a known bounded range.

    Metrics shown here are bounded to [-1, 1], which keeps null plots visually
    comparable across runs without requiring manual axis tuning.
    """
    if metric == "silhouette":
        return (-1.0, 1.0)
    if metric == "cliffs":
        return (-1.0, 1.0)
    if metric == "norm_mean_diff":
        return (-1.0, 1.0)
    return None


def make_plot(
    null_values: np.ndarray,
    observed_value: float,
    metric: str,
    output_path: Path | None,
    title: str | None,
    xlabel: str,
    ylabel: str | None,
    figsize: tuple[float, float],
    seed: int,
    p_value: float,
) -> None:
    """Plot the null best-value distribution with the observed value overlaid.

    The figure combines a compact boxplot, jittered null points, and an
    annotation for the observed best score and empirical p-value.
    """
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = 1.0

    ax.boxplot(
        [null_values],
        positions=[y_pos],
        widths=0.5,
        vert=False,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="lightgray", edgecolor="black", linewidth=1.2),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
    )

    y_jitter = rng.uniform(y_pos - 0.08, y_pos + 0.08, size=len(null_values))
    ax.scatter(
        null_values,
        y_jitter,
        alpha=0.8,
        s=35,
        zorder=3,
        label="Null best values",
    )

    ax.scatter(
        [observed_value],
        [y_pos],
        s=90,
        color="red",
        marker="o",
        zorder=5,
        label="Observed best value",
    )

    ax.axvline(
        observed_value,
        color="red",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        zorder=2,
    )

    ax.set_yticks([y_pos])
    ax.set_yticklabels([xlabel])

    ax.set_xlabel(metric_axis_label(metric, ylabel))

    xlim = metric_default_xlim(metric)
    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.set_ylim(y_pos - 0.35, y_pos + 0.35)

    if title is not None:
        ax.set_title(title)

    x_min, x_max = ax.get_xlim()
    # Keep the annotation near the observed line while avoiding the right edge.
    annotation_x = min(
        observed_value + 0.03 * (x_max - x_min),
        x_max - 0.05 * (x_max - x_min),
    )
    annotation_y = y_pos + 0.12

    ax.text(
        annotation_x,
        annotation_y,
        f"observed = {observed_value:.4f}\np = {p_value:.4g}",
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.9,
            edgecolor="gray",
        ),
    )

    ax.legend(loc="upper left")
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def print_best_row_summary(
    label: str,
    csv_path: Path,
    best_value: float,
    best_row: pd.Series,
) -> None:
    """Print a compact summary of the selected best row for one CSV.

    The summary highlights the file, method metadata, and the metric value that
    determined why this row was chosen.
    """
    distance_name = best_row.get("distance_name", "n/a")
    approach = best_row.get("approach", "n/a")
    hyperparams = best_row.get("hyperparameters_json", "n/a")

    print(f"{label}:")
    print(f"  file:           {csv_path.name}")
    print(f"  approach:       {approach}")
    print(f"  distance_name:  {distance_name}")
    print(f"  best_value:     {best_value:.6f}")
    print(f"  hyperparams:    {hyperparams}")


def main() -> None:
    """Run the null-vs-observed evaluation and generate the final report.

    The workflow builds the null distribution from per-file maxima, computes
    the empirical p-value against the observed run, and renders a summary plot.
    """
    args = parse_args()
    validate_args(args)

    # ---- Load best-per-file values ----
    null_results = collect_null_best_values(args.null_dir, args.glob, args.metric)
    null_values = np.array([value for _, value, _ in null_results], dtype=float)

    observed_value, observed_best_row = extract_best_row_value(
        args.actual_csv,
        args.metric,
    )

    # ---- Evaluate observed result against the null distribution ----
    p_value = empirical_p_value(
        null_values,
        observed_value,
        two_sided=args.two_sided,
    )

    # ---- Report numeric summary and selected rows ----
    print("\n=== Summary ===")
    print(f"Metric:                {args.metric}")
    print(f"Number of null files:  {len(null_values)}")
    print(f"Observed best value:   {observed_value:.6f}")
    print(f"Null mean:             {np.mean(null_values):.6f}")
    if len(null_values) > 1:
        print(f"Null std:              {np.std(null_values, ddof=1):.6f}")
    else:
        print("Null std:              0.000000")
    print(f"Null min:              {np.min(null_values):.6f}")
    print(f"Null median:           {np.median(null_values):.6f}")
    print(f"Null max:              {np.max(null_values):.6f}")
    print(f"Empirical p-value:     {p_value:.6g}")

    print("\nObserved best row:")
    print_best_row_summary("Observed", args.actual_csv, observed_value, observed_best_row)

    print("\nNull file best values:")
    for csv_path, best_value, best_row in null_results:
        print_best_row_summary("Null", csv_path, best_value, best_row)

    # ---- Visualize null distribution and observed best value ----
    make_plot(
        null_values=null_values,
        observed_value=observed_value,
        metric=args.metric,
        output_path=args.output,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        figsize=tuple(args.figsize),
        seed=args.seed,
        p_value=p_value,
    )


if __name__ == "__main__":
    main()
