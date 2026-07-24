from __future__ import annotations

"""Utilities for comparing pairwise distance structure across binary actor groups.

The module supports both the true human/AI labeling and stratified null assignments
that preserve group sizes for permutation-style comparisons.
"""

from itertools import combinations
from math import comb
from typing import Sequence, Optional

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import silhouette_samples


def get_num_binary_group_assignments(
    labels: Sequence[str],
    *,
    ai_marker: str = "GPT",
    exclude_true_assignment: bool = True,
) -> int:
    """Count distinct binary assignments that preserve the observed group sizes.

    Assignments are defined only by which labels fall into the AI group, so within-group
    ordering does not matter. The true assignment can optionally be excluded.
    """
    labels = list(labels)
    true_groups = {
        label: ("ai" if ai_marker in label else "human")
        for label in labels
    }
    n_ai = sum(group == "ai" for group in true_groups.values())
    n_total = len(labels)

    if n_ai == 0 or n_ai == n_total:
        raise ValueError("Need at least one actor in each class.")

    total = comb(n_total, n_ai)
    return total - 1 if exclude_true_assignment else total


def infer_binary_groups(
    labels: Sequence[str],
    *,
    ai_marker: str = "GPT",
    assignment_mode: str = "true",
    rng: Optional[np.random.Generator] = None,
    assignment_idx: Optional[int] = None,
    exclude_true_assignment: bool = True,
) -> dict[str, str]:
    """Infer a binary human/AI grouping under a true or null assignment scheme.

    Null assignments preserve the observed number of AI labels so comparisons stay
    matched to the original class balance. Returns a mapping from label to group.
    """
    labels = list(labels)

    true_groups = {
        label: ("ai" if ai_marker in label else "human")
        for label in labels
    }

    if assignment_mode == "true":
        return true_groups

    n_ai = sum(group == "ai" for group in true_groups.values())
    n_total = len(labels)

    if n_ai == 0 or n_ai == n_total:
        raise ValueError("Need at least one actor in each class.")

    if assignment_mode == "random_stratified":
        if rng is None:
            rng = np.random.default_rng()

        # Shuffle identities but keep the original AI-group size fixed.
        permuted = np.array(labels, dtype=object)
        rng.shuffle(permuted)

        ai_labels = set(permuted[:n_ai])
        return {
            label: ("ai" if label in ai_labels else "human")
            for label in labels
        }

    if assignment_mode == "indexed_stratified":
        if assignment_idx is None:
            raise ValueError(
                "assignment_mode='indexed_stratified' requires assignment_idx."
            )

        all_labels = tuple(sorted(labels))
        true_ai_labels = tuple(sorted(
            label for label, group in true_groups.items() if group == "ai"
        ))

        # Sorting makes assignment_idx stable across runs and environments.
        all_assignments = [
            ai_comb
            for ai_comb in combinations(all_labels, n_ai)
            if (not exclude_true_assignment) or tuple(sorted(ai_comb)) != true_ai_labels
        ]

        if not (0 <= assignment_idx < len(all_assignments)):
            raise ValueError(
                f"assignment_idx={assignment_idx} out of range. "
                f"Valid range: 0..{len(all_assignments) - 1}"
            )

        ai_labels = set(all_assignments[assignment_idx])
        return {
            label: ("ai" if label in ai_labels else "human")
            for label in labels
        }

    raise ValueError(
        "assignment_mode must be 'true', 'random_stratified', or 'indexed_stratified'"
    )


def extract_group_distance_vectors(
    distance_matrix: np.ndarray,
    labels: Sequence[str],
    *,
    ai_marker: str = "GPT",
    assignment_mode: str = "true",
    rng: Optional[np.random.Generator] = None,
    assignment_idx: Optional[int] = None,
    exclude_true_assignment: bool = True,
) -> dict[str, np.ndarray]:
    """Split upper-triangle distances into within-AI, within-human, and cross-group sets.

    The grouping can come from the true labels or from a stratified null assignment,
    enabling downstream comparison of observed and randomized structure.
    """
    if distance_matrix.shape != (len(labels), len(labels)):
        raise ValueError("distance_matrix shape must match labels.")

    groups = infer_binary_groups(
        labels,
        ai_marker=ai_marker,
        assignment_mode=assignment_mode,
        rng=rng,
        assignment_idx=assignment_idx,
        exclude_true_assignment=exclude_true_assignment,
    )

    ai_ai: list[float] = []
    human_human: list[float] = []
    ai_human: list[float] = []

    for i, j in combinations(range(len(labels)), 2):
        value = float(distance_matrix[i, j])
        g1 = groups[labels[i]]
        g2 = groups[labels[j]]

        if g1 == "ai" and g2 == "ai":
            ai_ai.append(value)
        elif g1 == "human" and g2 == "human":
            human_human.append(value)
        else:
            ai_human.append(value)

    return {
        "ai_ai": np.array(ai_ai, dtype=float),
        "human_human": np.array(human_human, dtype=float),
        "ai_human": np.array(ai_human, dtype=float),
    }


def summarize_vector(values: np.ndarray) -> dict[str, float]:
    """Return a compact descriptive summary for a one-dimensional distance vector.

    Empty inputs yield NaNs for location and spread so downstream reporting remains
    explicit rather than silently dropping a comparison.
    """
    if len(values) == 0:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
        }

    return {
        "n": float(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta for two samples.

    The statistic lies in [-1, 1]; positive values indicate that `x` tends to
    contain larger values than `y`.
    """
    if len(x) == 0 or len(y) == 0:
        return float("nan")

    greater = 0
    smaller = 0

    for xv in x:
        greater += int(np.sum(xv > y))
        smaller += int(np.sum(xv < y))

    return float((greater - smaller) / (len(x) * len(y)))


def mannwhitney_with_effect_size(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Run a Mann-Whitney test together with Cliff's delta.

    This pairs significance testing with a nonparametric effect size so group
    differences can be interpreted beyond the p-value alone.
    """
    if len(x) == 0 or len(y) == 0:
        return {
            "u_statistic": float("nan"),
            "p_value": float("nan"),
            "cliffs_delta": float("nan"),
        }

    test = mannwhitneyu(x, y, alternative=alternative)
    return {
        "u_statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "cliffs_delta": cliffs_delta(x, y),
    }


def silhouette_summary(
    distance_matrix: np.ndarray,
    labels: Sequence[str],
    *,
    ai_marker: str = "GPT",
    assignment_mode: str = "true",
    rng: Optional[np.random.Generator] = None,
    assignment_idx: Optional[int] = None,
    exclude_true_assignment: bool = True,
) -> dict[str, float]:
    """Summarize silhouette separation for a binary grouping on a distance matrix.

    Uses the precomputed pairwise distances and reports mean silhouette values for
    all samples together and for each group separately.
    """
    group_map = infer_binary_groups(
        labels,
        ai_marker=ai_marker,
        assignment_mode=assignment_mode,
        rng=rng,
        assignment_idx=assignment_idx,
        exclude_true_assignment=exclude_true_assignment,
    )
    class_labels = np.array(
        [1 if group_map[label] == "ai" else 0 for label in labels],
        dtype=int,
    )

    if len(np.unique(class_labels)) < 2:
        raise ValueError("Silhouette requires at least two groups.")

    # Silhouette is evaluated on the same binary assignment used for the distance splits.
    values = silhouette_samples(distance_matrix, class_labels, metric="precomputed")
    ai_mask = class_labels == 1
    human_mask = class_labels == 0

    return {
        "overall_mean": float(np.mean(values)),
        "ai_mean": float(np.mean(values[ai_mask])),
        "human_mean": float(np.mean(values[human_mask])),
    }


def analyze_binary_group_structure(
    distance_matrix: np.ndarray,
    labels: Sequence[str],
    *,
    ai_marker: str = "GPT",
    assignment_mode: str = "true",
    rng: Optional[np.random.Generator] = None,
    assignment_idx: Optional[int] = None,
    exclude_true_assignment: bool = True,
) -> dict[str, object]:
    """Assemble descriptive and inferential summaries for a binary group assignment.

    The output combines distance summaries, silhouette separation, and two
    Mann-Whitney comparisons that benchmark cross-group distances against each
    within-group baseline.
    """
    # ---- Distance partitions ----
    vectors = extract_group_distance_vectors(
        distance_matrix,
        labels,
        ai_marker=ai_marker,
        assignment_mode=assignment_mode,
        rng=rng,
        assignment_idx=assignment_idx,
        exclude_true_assignment=exclude_true_assignment,
    )

    ai_ai = vectors["ai_ai"]
    human_human = vectors["human_human"]
    ai_human = vectors["ai_human"]

    # ---- Combined report ----
    return {
        "group_assignment": infer_binary_groups(
            labels,
            ai_marker=ai_marker,
            assignment_mode=assignment_mode,
            rng=rng,
            assignment_idx=assignment_idx,
            exclude_true_assignment=exclude_true_assignment,
        ),
        "group_summaries": {
            "ai_ai": summarize_vector(ai_ai),
            "human_human": summarize_vector(human_human),
            "ai_human": summarize_vector(ai_human),
        },
        "silhouette": silhouette_summary(
            distance_matrix,
            labels,
            ai_marker=ai_marker,
            assignment_mode=assignment_mode,
            rng=rng,
            assignment_idx=assignment_idx,
            exclude_true_assignment=exclude_true_assignment,
        ),
        "mannwhitney": {
            "ai_human_vs_ai_ai": mannwhitney_with_effect_size(ai_human, ai_ai),
            "ai_human_vs_human_human": mannwhitney_with_effect_size(ai_human, human_human),
        },
    }


def print_binary_group_structure_report(
    result: dict[str, object],
) -> None:
    """Print a compact text report for `analyze_binary_group_structure` output.

    The report mirrors the analysis structure so descriptive summaries, silhouette
    values, and inferential tests can be inspected together.
    """
    summaries = result["group_summaries"]
    silhouette = result["silhouette"]
    tests = result["mannwhitney"]
    assignment = result.get("group_assignment", None)

    if assignment is not None:
        print("\nGroup assignment:")
        for label, group in assignment.items():
            print(f"  {label}: {group}")

    print("\nGroup distance means:")
    for name in ("ai_ai", "ai_human", "human_human"):
        block = summaries[name]
        print(
            f"  {name}: "
            f"n={int(block['n'])}, "
            f"mean={block['mean']:.6f}, "
            f"median={block['median']:.6f}, "
            f"std={block['std']:.6f}"
        )

    print("\nSilhouette summary:")
    print(f"  overall_mean={silhouette['overall_mean']:.6f}")
    print(f"  ai_mean={silhouette['ai_mean']:.6f}")
    print(f"  human_mean={silhouette['human_mean']:.6f}")

    print("\nMann-Whitney tests:")
    for name in ("ai_human_vs_ai_ai", "ai_human_vs_human_human"):
        block = tests[name]
        print(
            f"  {name}: "
            f"U={block['u_statistic']:.6f}, "
            f"p={block['p_value']:.6g}, "
            f"cliffs_delta={block['cliffs_delta']:.6f}"
        )
