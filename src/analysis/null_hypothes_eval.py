#!/usr/bin/env python3
"""
Visualize an observed run against a null distribution of permutation-based runs.

The script aggregates one mean metric value per null CSV, computes an empirical
p-value against the correctly labeled run, and renders the null distribution as a
compact horizontal boxplot with jittered points.

Usage:
    python plot_null_boxplot.py \
        --metric test_f1_macro \
        --null-dir results/null_hypotheses \
        --actual-csv results/correct_assignment.csv \
        --output plots/null_test_f1_macro.png

Example:
    python plot_null_boxplot.py \
        --metric test_mcc \
        --null-dir ./null_csvs \
        --actual-csv ./correct.csv \
        --larger-is-better \
        --output ./mcc_null_plot.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for null-distribution visualization.

    The interface supports different empirical p-value conventions and optional
    figure export. Returns the populated argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create a boxplot of mean metric values from null-hypothesis CSV files "
            "and overlay the observed mean from a correctly labeled CSV."
        )
    )
    parser.add_argument(
        "--metric",
        required=True,
        help=(
            "Metric column to use, e.g. test_f1_macro, test_mcc, "
            "val_f1_macro, test_balanced_accuracy"
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
        help="CSV file containing the correctly labeled run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the figure. If omitted, the plot is shown interactively.",
    )
    parser.add_argument(
        "--glob",
        default="*.csv",
        help="Glob pattern for null CSV files inside --null-dir (default: *.csv).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom plot title. By default, no title is shown.",
    )
    parser.add_argument(
        "--ylabel",
        default=None,
        help="Optional custom y-axis label. Defaults to the metric name.",
    )
    parser.add_argument(
        "--xlabel",
        default="Null distribution",
        help="Label for the x-axis category (default: Null distribution).",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(5.5, 6.5),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 5.5 6.5).",
    )
    parser.add_argument(
        "--larger-is-better",
        action="store_true",
        default=False,
        help=(
            "Use a right-tail empirical p-value: p = P(null >= observed). "
            "Recommended for metrics like F1, balanced accuracy, MCC."
        ),
    )
    parser.add_argument(
        "--smaller-is-better",
        action="store_true",
        default=False,
        help=(
            "Use a left-tail empirical p-value: p = P(null <= observed). "
            "Useful for error metrics if you ever use them."
        ),
    )
    parser.add_argument(
        "--two-sided",
        action="store_true",
        default=False,
        help="Use a simple two-sided empirical p-value based on absolute deviation from the null mean.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for jitter placement (default: 42).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate paths and ensure the p-value tail specification is unambiguous.

    The script allows at most one explicit tail mode so the null comparison has
    a single interpretation.
    """
    if not args.null_dir.exists():
        raise FileNotFoundError(f"Null directory does not exist: {args.null_dir}")
    if not args.null_dir.is_dir():
        raise NotADirectoryError(f"Null path is not a directory: {args.null_dir}")
    if not args.actual_csv.exists():
        raise FileNotFoundError(f"Actual CSV does not exist: {args.actual_csv}")

    tail_flags = sum([args.larger_is_better, args.smaller_is_better, args.two_sided])
    if tail_flags > 1:
        raise ValueError(
            "Choose at most one of --larger-is-better, --smaller-is-better, or --two-sided."
        )


def read_metric_mean(csv_path: Path, metric: str) -> float:
    """Read one CSV and return the mean of the requested metric column.

    Non-numeric entries are coerced to missing values and dropped. Raises if the
    metric column is absent or if no valid numeric values remain.
    """
    df = pd.read_csv(csv_path)

    if metric not in df.columns:
        raise KeyError(f"Metric column '{metric}' not found in {csv_path}")

    series = pd.to_numeric(df[metric], errors="coerce").dropna()

    if series.empty:
        raise ValueError(f"No valid numeric values found for metric '{metric}' in {csv_path}")

    return float(series.mean())


def collect_null_means(null_dir: Path, glob_pattern: str, metric: str) -> List[Tuple[Path, float]]:
    """Collect per-file mean metric values for all null-hypothesis CSVs.

    Files are processed in sorted order to keep summaries and plots stable across
    runs. Returns `(path, mean_value)` pairs for downstream reporting.
    """
    csv_paths = sorted(null_dir.glob(glob_pattern))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in {null_dir} matching pattern '{glob_pattern}'"
        )

    results: List[Tuple[Path, float]] = []
    for csv_path in csv_paths:
        mean_value = read_metric_mean(csv_path, metric)
        results.append((csv_path, mean_value))
    return results


def empirical_p_value(
    null_values: np.ndarray,
    observed: float,
    larger_is_better: bool = False,
    smaller_is_better: bool = False,
    two_sided: bool = False,
) -> float:
    """Compute an empirical p-value from the null distribution.

    The default is a right-tailed test, which matches score metrics where larger
    values indicate better performance. A plus-one correction avoids zero p-values.
    """
    n = len(null_values)
    if n == 0:
        raise ValueError("Null distribution is empty.")

    if two_sided:
        # Center the two-sided test on the null mean rather than on zero.
        center = float(np.mean(null_values))
        obs_dev = abs(observed - center)
        count = int(np.sum(np.abs(null_values - center) >= obs_dev))
        return (1 + count) / (n + 1)

    if smaller_is_better:
        count = int(np.sum(null_values <= observed))
        return (1 + count) / (n + 1)

    # Default to the right tail for metrics such as F1, MCC, or balanced accuracy.
    count = int(np.sum(null_values >= observed))
    return (1 + count) / (n + 1)


def make_plot(
    null_values: np.ndarray,
    observed_value: float,
    metric: str,
    output_path: Path | None,
    title: str | None,
    ylabel: str | None,
    xlabel: str,
    figsize: Tuple[float, float],
    seed: int,
    p_value: float,
) -> None:
    """Render the null distribution and overlay the observed mean.

    The plot is intentionally compact and single-panel because the goal is to
    compare one observed score against one permutation-based null distribution.
    """
    rng = np.random.default_rng(seed)

    # A single horizontal panel makes the observed-vs-null comparison immediate.
    fig, ax = plt.subplots(figsize=(10, 3))

    y_pos = 1.0

    # Outliers are already shown as jittered points, so boxplot fliers stay hidden.
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

    # Jitter only separates overlapping null means visually; it has no analytic meaning.
    y_jitter = rng.uniform(y_pos - 0.08, y_pos + 0.08, size=len(null_values))
    ax.scatter(
        null_values,
        y_jitter,
        alpha=0.8,
        s=35,
        zorder=3,
        label="Null means",
    )

    ax.scatter(
        [observed_value],
        [y_pos],
        s=90,
        color="red",
        marker="o",
        zorder=5,
        label="Observed mean",
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

    ax.set_xlabel(ylabel or metric)

    # This script assumes bounded score-like metrics on [0, 1].
    ax.set_xlim(0.0, 1.0)

    ax.set_ylim(y_pos - 0.35, y_pos + 0.35)

    if title is not None:
        ax.set_title(title)

    annotation_x = min(observed_value + 0.03, 0.97)
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

def main() -> None:
    """Run the null-hypothesis summary and plotting workflow.

    The script reduces each CSV to one mean metric value, compares the observed
    run to the null distribution, prints summary statistics, and creates the plot.
    """
    # ---- Parse inputs and validate configuration ----
    args = parse_args()
    validate_args(args)

    # ---- Load null and observed summary values ----
    null_results = collect_null_means(args.null_dir, args.glob, args.metric)
    null_values = np.array([value for _, value in null_results], dtype=float)

    observed_value = read_metric_mean(args.actual_csv, args.metric)

    # Unless specified otherwise, treat larger scores as better and test the right tail.
    p_value = empirical_p_value(
        null_values,
        observed_value,
        larger_is_better=args.larger_is_better or (
            not args.smaller_is_better and not args.two_sided
        ),
        smaller_is_better=args.smaller_is_better,
        two_sided=args.two_sided,
    )

    # ---- Report null-distribution summary ----
    print("\n=== Summary ===")
    print(f"Metric:                {args.metric}")
    print(f"Number of null files:  {len(null_values)}")
    print(f"Observed mean:         {observed_value:.6f}")
    print(f"Null mean:             {np.mean(null_values):.6f}")
    if len(null_values) > 1:
        print(f"Null std:              {np.std(null_values, ddof=1):.6f}")
    else:
        print("Null std:              0.000000")
    print(f"Null min:              {np.min(null_values):.6f}")
    print(f"Null median:           {np.median(null_values):.6f}")
    print(f"Null max:              {np.max(null_values):.6f}")
    print(f"Empirical p-value:     {p_value:.6g}")

    print("\nNull file means:")
    for csv_path, mean_value in null_results:
        print(f"  {csv_path.name}: {mean_value:.6f}")

    # ---- Render figure ----
    make_plot(
        null_values=null_values,
        observed_value=observed_value,
        metric=args.metric,
        output_path=args.output,
        title=args.title,
        ylabel=args.ylabel,
        xlabel=args.xlabel,
        figsize=tuple(args.figsize),
        seed=args.seed,
        p_value=p_value,
    )


if __name__ == "__main__":
    main()
