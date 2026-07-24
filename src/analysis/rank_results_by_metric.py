#!/usr/bin/env python3
"""Rank model result files by a selected metric and visualize the ordering.

The script aggregates per-run metric values from matching CSV files, sorts models
by their mean score, and optionally compares the top three with paired Wilcoxon tests.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path


def find_matching_csvs(results_dir: Path, split_size: int) -> list[Path]:
    """Return result CSVs whose filenames encode the requested split size.

    The match is purely name-based and assumes the split size appears as
    `_{split_size}_` in each relevant filename.
    """
    pattern = f"_{split_size}_"
    return sorted(
        path
        for path in results_dir.glob("*.csv")
        if pattern in path.name
    )


def load_metric_values(csv_path: Path, metric: str) -> list[float]:
    """Load all non-empty values for one metric column from a result CSV.

    Empty entries are skipped so partially populated result tables remain usable.
    Raises `ValueError` when the metric is missing or has no valid numeric values.
    """
    values: list[float] = []

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or metric not in reader.fieldnames:
            raise ValueError(f"Metric '{metric}' not found in {csv_path.name}.")

        for row in reader:
            raw_value = row.get(metric, "")
            if raw_value is None:
                continue

            value = raw_value.strip()
            if not value:
                continue

            values.append(float(value))

    if not values:
        raise ValueError(f"Metric '{metric}' has no non-NaN values in {csv_path.name}.")

    return values


def compute_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def derive_output_path(results_dir: Path, split_size: int, metric: str) -> Path:
    """Construct the default output path for the ranked boxplot PDF."""
    safe_metric = metric.replace("/", "_")
    return results_dir / f"boxplot_ranked_{safe_metric}_{split_size}.pdf"


def format_model_label(filename: str) -> str:
    """Convert a result filename into a presentation-friendly model label.

    The parser assumes filenames start with a model identifier before any
    `_nested_...` suffix and strips trailing numeric run markers when present.
    """
    stem = Path(filename).stem
    model_part = stem.split("_nested_", maxsplit=1)[0]
    model_tokens = model_part.split("_")

    if model_tokens and model_tokens[-1].isdigit():
        model_tokens = model_tokens[:-1]

    normalized = "_".join(model_tokens).lower()

    label_map = {
        "cnn": "CNN",
        "transformer": "Transformer",
        "llm": "LLM",
        "sentence_transformer": "SentenceTransformer",
        "sentence-transformer": "SentenceTransformer",
        "tfidf_logreg": "TF-IDF + Logistic Regression",
        "tfidf_svm": "TF-IDF + SVM",
        "tfidf_linear_svm": "TF-IDF + SVM",
        "tfidf_logistic_regression": "TF-IDF + Logistic Regression",
        "dummy_stratified": "Dummy (stratified)",
        "dummy_most_frequent": "Dummy (most frequent)",
        "most_frequent": "Dummy (most frequent)",
        "stratified": "Dummy (stratified)",
        "ridge": "Ridge Classifier",
        "gaussian_nb": "Gaussian Naive Bayes",
        "multinomial_nb": "Multinomial Naive Bayes",
        "bernoulli_nb": "Bernoulli Naive Bayes",
        "complement_nb": "Complement Naive Bayes",
        "sgd_hinge": "SGD (hinge)",
        "sgd_log": "SGD (log-loss)",
        "sgd_log_loss": "SGD (log-loss)",
        "pa_like": "PA-like",
        "knn": "kNN",
        "random_forest": "Random Forest",
    }

    # Preserve established labels for common model names used across experiments.
    if normalized in label_map:
        return label_map[normalized]

    readable_tokens = []
    for token in model_tokens:
        lower = token.lower()
        if lower == "tfidf":
            readable_tokens.append("TF-IDF")
        elif lower == "svm":
            readable_tokens.append("SVM")
        elif lower == "cnn":
            readable_tokens.append("CNN")
        elif lower == "llm":
            readable_tokens.append("LLM")
        elif lower == "knn":
            readable_tokens.append("kNN")
        elif lower == "nb":
            readable_tokens.append("NB")
        elif lower == "logreg":
            readable_tokens.append("Logistic Regression")
        elif lower == "pa":
            readable_tokens.append("PA")
        else:
            readable_tokens.append(token.replace("-", " ").replace("_", " ").title())

    return " ".join(readable_tokens)


def format_metric_label(metric: str) -> str:
    """Map internal metric column names to concise plot labels."""
    metric_map = {
        "test_mcc": "MCC",
        "mcc": "MCC",
        "test_balanced_accuracy": "Balanced Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "test_accuracy": "Accuracy",
        "accuracy": "Accuracy",
        "test_f1": "F1",
        "f1": "F1",
        "test_macro_f1": "Macro F1",
        "macro_f1": "Macro F1",
        "test_weighted_f1": "Weighted F1",
        "weighted_f1": "Weighted F1",
        "test_precision": "Precision",
        "precision": "Precision",
        "test_recall": "Recall",
        "recall": "Recall",
    }
    return metric_map.get(metric.lower(), metric.replace("_", " ").title())


def create_ranked_boxplot(
    rows: list[tuple[str, float, list[float]]],
    metric: str,
    split_size: int,
    output_path: Path,
) -> None:
    """Create a horizontal boxplot ordered by mean metric value.

    Each row is expected to contain a filename, its mean score, and the underlying
    per-run values that define the boxplot distribution.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to create the boxplot. "
            "Install it with: python3 -m pip install matplotlib"
        ) from exc

    labels = [format_model_label(filename) for filename, _, _ in rows]
    values = [metric_values for _, _, metric_values in rows]
    means = [mean_value for _, mean_value, _ in rows]

    height = max(6, 0.5 * len(rows) + 2)
    fig, ax = plt.subplots(figsize=(12, height))

    boxplot = ax.boxplot(
        values,
        vert=False,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "#d62728",
            "markeredgecolor": "#7f0000",
            "markersize": 5,
        },
    )

    for patch in boxplot["boxes"]:
        patch.set_facecolor("#9ecae1")
        patch.set_edgecolor("#1f3b4d")
        patch.set_alpha(0.8)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(format_metric_label(metric))
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    # The best model is sorted to the top for easier visual comparison.
    ax.invert_yaxis()

    # Annotate means directly because boxplots emphasize distributions over ranking.
    for position, mean_value in enumerate(means, start=1):
        text_x = min(mean_value + 0.015, 0.98)
        ax.text(
            text_x,
            position,
            f"{mean_value:.3f}",
            va="center",
            ha="left",
            fontsize=8,
            color="#7f0000",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_wilcoxon_top_three(
    rows: list[tuple[str, float, list[float]]],
) -> list[tuple[str, str, float, float]]:
    """Run pairwise Wilcoxon signed-rank tests for the top three ranked models.

    The test is applied to paired per-run metric values, so each compared model
    must contribute the same number of observations.
    """
    try:
        from scipy.stats import wilcoxon
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scipy is required to run Wilcoxon tests. "
            "Install it with: python3 -m pip install scipy"
        ) from exc

    top_rows = rows[:3]
    results: list[tuple[str, str, float, float]] = []

    # Restrict comparisons to the leading models to keep the follow-up test targeted.
    for (name_a, _, values_a), (name_b, _, values_b) in combinations(top_rows, 2):
        if len(values_a) != len(values_b):
            raise ValueError(
                "Wilcoxon test requires paired samples with the same length: "
                f"{name_a} has {len(values_a)} values, {name_b} has {len(values_b)} values."
            )

        statistic, p_value = wilcoxon(values_a, values_b)
        results.append((name_a, name_b, float(statistic), float(p_value)))

    return results


def main() -> None:
    """Parse arguments, rank matching result files, and report the summary outputs."""
    parser = argparse.ArgumentParser(
        description=(
            "Find result CSVs for a given split size, compute the mean of a metric, "
            "and print the files sorted by that mean."
        )
    )
    parser.add_argument(
        "--split-size",
        type=int,
        required=True,
        help="Select CSVs whose filename contains _{split-size}_ (for example: 50).",
    )
    parser.add_argument(
        "--metric",
        default="test_mcc",
        help="Metric column to average (default: test_mcc).",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing result CSVs (default: results).",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort from lowest mean to highest mean. Default is highest first.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output path for the combined boxplot image "
            "(default: results/boxplot_ranked_{metric}_{split-size}.pdf)."
        ),
    )
    args = parser.parse_args()

    # ---- Load and rank results ----
    results_dir = Path(args.results_dir)
    csv_paths = find_matching_csvs(results_dir, args.split_size)
    if not csv_paths:
        raise ValueError(
            f"No CSV files found in {results_dir} with _{args.split_size}_ in the filename."
        )

    rows: list[tuple[str, float, list[float]]] = []
    for csv_path in csv_paths:
        metric_values = load_metric_values(csv_path, args.metric)
        mean_value = compute_mean(metric_values)
        rows.append((csv_path.name, mean_value, metric_values))

    rows.sort(key=lambda item: item[1], reverse=not args.ascending)

    # ---- Create outputs ----
    output_path = Path(args.output) if args.output else derive_output_path(
        results_dir, args.split_size, args.metric
    )
    create_ranked_boxplot(rows, args.metric, args.split_size, output_path)
    wilcoxon_results = run_wilcoxon_top_three(rows) if len(rows) >= 3 else []

    # ---- Print ranked summary ----
    print(f"Metric: {args.metric}")
    print(f"Split size: {args.split_size}")
    print(f"Matched files: {len(rows)}")
    print(f"Plot: {output_path}")
    print()

    for index, (filename, mean_value, _) in enumerate(rows, start=1):
        print(f"{index:>2}. {mean_value:.6f}  {format_model_label(filename)}")

    if wilcoxon_results:
        print()
        print("Wilcoxon signed-rank test for top 3:")
        for name_a, name_b, statistic, p_value in wilcoxon_results:
            print(f"- {format_model_label(name_a)} vs {format_model_label(name_b)}")
            print(f"  statistic={statistic:.6f}, p-value={p_value:.6f}")


if __name__ == "__main__":
    main()
