"""Shared runner for a single group-structure evaluation pass.

This module turns pairwise distance outputs into a distance matrix, evaluates
human-vs-AI group separation under a chosen assignment mode, and optionally
exports summary statistics and diagnostic plots.
"""

from typing import Callable, Any

from src.core.stats.pairwise_group_stats import (
    analyze_binary_group_structure,
    print_binary_group_structure_report,
)
from src.core.stats.pairwise_viz import (
    build_symmetric_distance_matrix,
    group_labels_humans_first,
    plot_distance_heatmap,
    plot_mds_embedding,
)
from src.core.stats.statistic_evaluation_csv import append_statistic_evaluation_row


def evaluate_single_run(
    *,
    tool_name: str,
    labels: list[str],
    pairwise_results: list[dict],
    distance_name: str,
    distance_extractor: Callable[[dict], float],
    hyperparameters: dict,
    assignment_mode: str = "true",
    assignment_idx: int | None = None,
    output_path: str | None = None,
    plot: bool = False,
    anonymize_humans: bool = False,
) -> dict:
    """Run one evaluation pass from pairwise distances to summary outputs.

    Labels are reordered to keep the human/AI block structure interpretable in
    downstream statistics and plots. Returns the ordered labels, full distance
    matrix, and computed group-level statistics.
    """
    # Keep the matrix layout consistent across runs so group-level structure is
    # directly interpretable in both the statistics and the optional plots.
    ordered_labels = group_labels_humans_first(labels)

    # Reconstruct the full symmetric matrix expected by the statistical
    # analysis from the pairwise result records produced upstream.
    matrix = build_symmetric_distance_matrix(
        ordered_labels,
        pairwise_results,
        extract_pair=lambda item: (
            item["label_1"],
            item["label_2"],
            distance_extractor(item),
        ),
    )

    # `assignment_mode` controls whether we evaluate the true human/AI split or
    # a null/permuted assignment used for comparison or significance testing.
    group_stats = analyze_binary_group_structure(
        matrix,
        ordered_labels,
        assignment_mode=assignment_mode,
        assignment_idx=assignment_idx,
    )

    print_binary_group_structure_report(group_stats)

    if output_path is not None:
        append_statistic_evaluation_row(
            approach=tool_name,
            distance_name=distance_name,
            ordered_labels=ordered_labels,
            hyperparameters=hyperparameters,
            group_stats=group_stats,
            output_path=output_path,
        )

    if plot:
        # The plotting path mirrors the evaluated ordering/assignment so visual
        # diagnostics stay aligned with the reported statistics.
        plot_distance_heatmap(
            matrix,
            ordered_labels,
            title="",
            anonymize_humans=anonymize_humans,
        )
        plot_mds_embedding(
            matrix,
            ordered_labels,
            title="",
            anonymize_humans=anonymize_humans,
        )

    return {
        "ordered_labels": ordered_labels,
        "matrix": matrix,
        "group_stats": group_stats,
    }
