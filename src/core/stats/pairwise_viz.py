from __future__ import annotations

"""Visualization helpers for pairwise distance analyses.

The module turns symmetric pairwise comparison results into publication-ready
heatmaps and MDS plots, with optional grouping of human and AI actors.
"""

from pathlib import Path
import re
from collections.abc import Callable, Sequence
from inspect import signature
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.manifold import MDS


PairExtractor = Callable[[Any], tuple[str, str, float]]
PointStyler = Callable[[str], dict[str, Any]]
GroupAssigner = Callable[[str], str]


DEFAULT_PLOT_DIR = Path("results")


def _slugify_filename(value: str) -> str:
    """Convert a plot title into a filesystem-safe stem.

    Returns a compact fallback name when the cleaned title would be empty.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or "plot"


def _resolve_output_path(
    *,
    save_path: str | Path | None,
    title: str,
    suffix: str,
    output_dir: str | Path = DEFAULT_PLOT_DIR,
    extension: str = "pdf",
) -> Path | None:
    """Resolve where a figure should be written.

    Returns `None` when saving is disabled and otherwise normalizes either an
    explicit path or a title-based default inside the plot directory.
    """
    if save_path is False:
        return None

    if save_path is None:
        output_dir = Path(output_dir)
        return output_dir / f"{_slugify_filename(title)}_{suffix}.{extension}"

    save_path = Path(save_path)
    if save_path.suffix:
        return save_path
    return save_path.with_suffix(f".{extension}")


def _finalize_figure(
    fig: plt.Figure,
    *,
    save_path: str | Path | None,
    dpi: int,
) -> Path | None:
    """Save, display, and close a figure in one place.

    Returns the resolved output path when the figure is written to disk.
    """
    output_path = Path(save_path) if save_path is not None else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output_path


def group_labels_humans_first(
    labels: Sequence[str],
    *,
    ai_marker: str = "GPT",
) -> list[str]:
    """Reorder labels so human entries precede AI entries.

    Grouping labels this way makes the heatmap block structure easier to read.
    """
    humans = [label for label in labels if ai_marker not in label]
    ais = [label for label in labels if ai_marker in label]
    return humans + ais


def default_group_assigner(label: str, *, ai_marker: str = "GPT") -> str:
    """Assign a label to the default Human or AI display group."""
    return "AI" if ai_marker in label else "Human"


def anonymize_actor_labels(
    labels: Sequence[str],
    *,
    ai_marker: str = "GPT",
    human_prefix: str = "Human",
) -> list[str]:
    """Anonymize human labels while preserving AI identifiers.

    Repeated human labels receive the same alias so pairwise structure remains
    interpretable across plots.
    """
    anonymized: list[str] = []
    human_aliases: dict[str, str] = {}

    for label in labels:
        if ai_marker in label:
            anonymized.append(label)
            continue

        alias = human_aliases.get(label)
        if alias is None:
            alias = f"{human_prefix}{len(human_aliases) + 1}"
            human_aliases[label] = alias
        anonymized.append(alias)

    return anonymized


def build_symmetric_distance_matrix(
    labels: Sequence[str],
    pairwise_results: Sequence[Any],
    *,
    extract_pair: PairExtractor,
    diagonal_value: float = 0.0,
) -> np.ndarray:
    """Construct a dense symmetric matrix from pairwise distance results.

    `extract_pair` must return `(label_1, label_2, distance)`. The function
    raises if labels are duplicated, unknown, or missing from the pairwise set.
    """
    label_list = list(labels)
    label_to_idx = {label: i for i, label in enumerate(label_list)}

    if len(label_to_idx) != len(label_list):
        raise ValueError("labels must be unique.")

    matrix = np.full((len(label_list), len(label_list)), np.nan, dtype=float)
    np.fill_diagonal(matrix, diagonal_value)

    for item in pairwise_results:
        label_1, label_2, distance = extract_pair(item)

        if label_1 not in label_to_idx or label_2 not in label_to_idx:
            raise KeyError(f"Unknown label pair: {label_1!r}, {label_2!r}")

        i = label_to_idx[label_1]
        j = label_to_idx[label_2]
        matrix[i, j] = distance
        matrix[j, i] = distance

    missing_mask = np.isnan(matrix)
    np.fill_diagonal(missing_mask, False)
    if np.any(missing_mask):
        missing_pairs = []
        for i in range(len(label_list)):
            for j in range(i + 1, len(label_list)):
                if missing_mask[i, j]:
                    missing_pairs.append((label_list[i], label_list[j]))
        raise ValueError(f"Missing distances for pairs: {missing_pairs}")

    return matrix


def plot_distance_heatmap(
    distance_matrix: np.ndarray,
    labels: Sequence[str],
    *,
    title: str = "Pairwise distance heatmap",
    cmap: str = "magma",
    annotate: bool = True,
    value_format: str = ".2f",
    upper_triangle_only: bool = False,
    annotate_upper_triangle_only: bool = True,
    mask_diagonal: bool = False,
    colorbar_label: str = "Distance",
    group_separator_index: int | None = None,
    anonymize_humans: bool = True,
    ai_marker: str = "GPT",
    human_prefix: str = "Human",
    save_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_PLOT_DIR,
    dpi: int = 300,
) -> None:
    """Render a symmetric distance matrix as a publication-style heatmap.

    The plot supports human/AI label anonymization and optional group
    separators. For symmetric matrices, upper-triangle display is usually the
    clearest presentation.
    """
    matrix = np.asarray(distance_matrix, dtype=float)
    labels = list(labels)
    display_labels = (
        anonymize_actor_labels(labels, ai_marker=ai_marker, human_prefix=human_prefix)
        if anonymize_humans
        else labels
    )

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array.")

    if matrix.shape[0] != len(display_labels):
        raise ValueError("Number of labels must match matrix dimensions.")

    display_matrix = matrix.copy()

    if upper_triangle_only:
        # The matrix is symmetric, so hiding the mirrored half reduces clutter.
        lower_mask = np.tril(np.ones_like(display_matrix, dtype=bool), k=-1)
        display_matrix[lower_mask] = np.nan

    if mask_diagonal:
        np.fill_diagonal(display_matrix, np.nan)

    n = len(display_labels)
    fig_size = max(7.5, min(13.5, 0.68 * n + 2.8))

    fig, ax = plt.subplots(
        figsize=(fig_size, fig_size),
        facecolor="#f7f7f5",
    )
    ax.set_facecolor("#f7f7f5")

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#ede7df")

    image = ax.imshow(
        display_matrix,
        cmap=cmap_obj,
        interpolation="nearest",
        aspect="equal",
    )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label, fontsize=11, color="#2f2a24")
    colorbar.ax.tick_params(labelsize=10, colors="#4b443c")
    colorbar.outline.set_visible(False)

    positions = np.arange(n)
    ax.set_xticks(positions)
    ax.set_yticks(positions)

    ax.set_xticklabels(
        display_labels,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=10,
        color="#4b443c",
    )
    ax.set_yticklabels(
        display_labels,
        fontsize=10,
        color="#4b443c",
    )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="semibold",
        color="#2f2a24",
        pad=18,
    )
    ax.tick_params(axis="both", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # A light grid keeps cell boundaries legible without competing with the colormap.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=(1, 1, 1, 0.75), linestyle="-", linewidth=1.15)
    ax.tick_params(which="minor", bottom=False, left=False)

    # This separator highlights block structure after labels are reordered by group.
    if group_separator_index is not None and 0 < group_separator_index < n:
        sep = group_separator_index - 0.5
        ax.axhline(sep, color="#c9c1b8", linewidth=2.2, zorder=3)
        ax.axvline(sep, color="#c9c1b8", linewidth=2.2, zorder=3)

    if annotate:
        norm = image.norm

        for i in range(n):
            for j in range(n):
                if annotate_upper_triangle_only and j <= i:
                    continue

                value = display_matrix[i, j]
                if np.isnan(value):
                    continue

                rgba = cmap_obj(norm(value))
                # Switch text color by local luminance so annotations stay readable.
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                text_color = "white" if luminance < 0.48 else "#1f1a17"

                ax.text(
                    j,
                    i,
                    format(value, value_format),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8 if n > 14 else 9,
                    fontweight="semibold",
                )

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    fig.tight_layout()
    resolved_path = _resolve_output_path(
        save_path=save_path,
        title=title,
        suffix="heatmap",
        output_dir=output_dir,
        extension="pdf",
    )
    _finalize_figure(fig, save_path=resolved_path, dpi=dpi)


def _add_group_ellipse(
    ax: plt.Axes,
    points: np.ndarray,
    *,
    color: str,
    n_std: float = 1.8,
    alpha: float = 0.16,
    zorder: int = 1,
) -> None:
    """Draw a soft ellipse summarizing the spread of a point group.

    With two points, the ellipse is derived from the connecting segment; with
    larger groups, it is estimated from the covariance structure.
    """
    if len(points) < 2:
        return

    if len(points) == 2:
        center = points.mean(axis=0)
        diff = points[1] - points[0]
        angle = np.degrees(np.arctan2(diff[1], diff[0]))
        distance = np.linalg.norm(diff)
        width = max(distance * 1.35, 0.05)
        height = max(distance * 0.55, 0.05)
    else:
        center = points.mean(axis=0)
        cov = np.cov(points[:, 0], points[:, 1])

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        eigvals = np.maximum(eigvals, 1e-12)
        width, height = 2 * n_std * np.sqrt(eigvals)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

        width = max(width, 0.05)
        height = max(height, 0.05)

    ellipse = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        linewidth=1.4,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(ellipse)


def plot_mds_embedding(
    distance_matrix: np.ndarray,
    labels: Sequence[str],
    *,
    title: str = "MDS plot",
    random_state: int = 0,
    n_init: int = 20,
    metric: bool = True,
    point_styler: PointStyler | None = None,
    group_assigner: GroupAssigner | None = None,
    ai_marker: str = "GPT",
    group_markers: dict[str, str] | None = None,
    group_colors: dict[str, str] | None = None,
    anonymize_humans: bool = True,
    human_prefix: str = "Human",
    draw_group_ellipses: bool = True,
    ellipse_n_std: float = 1.8,
    ellipse_alpha: float = 0.14,
    show_legend: bool = True,
    text_offset: tuple[float, float] = (6, 6),   # now in points
    save_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_PLOT_DIR,
    dpi: int = 300,
) -> tuple[np.ndarray, float]:
    """Project a distance matrix into two dimensions with MDS and plot it.

    Points can be grouped and styled separately for human versus AI actors.
    Returns the 2D coordinates and the fitted stress value.
    """
    labels = list(labels)
    display_labels = (
        anonymize_actor_labels(labels, ai_marker=ai_marker, human_prefix=human_prefix)
        if anonymize_humans
        else labels
    )

    mds_params = signature(MDS).parameters

    # scikit-learn changed the MDS API; support both signatures without branching elsewhere.
    if "metric_mds" in mds_params:
        mds = MDS(
            metric="precomputed",
            metric_mds=metric,
            n_init=n_init,
            init="random",
            random_state=random_state,
        )
    else:
        mds = MDS(
            dissimilarity="precomputed",
            metric=metric,
            n_init=n_init,
            init="random",
            random_state=random_state,
        )

    coords = mds.fit_transform(distance_matrix)

    if group_assigner is None:
        group_assigner = lambda label: default_group_assigner(label, ai_marker=ai_marker)

    if group_markers is None:
        group_markers = {
            "Human": "o",
            "AI": "s",
        }

    if group_colors is None:
        group_colors = {
            "Human": "#1f6f8b",
            "AI": "#c06014",
        }

    fig, ax = plt.subplots(figsize=(8.6, 6.8), facecolor="#f7f7f5")
    ax.set_facecolor("#fcfbf8")
    ax.grid(True, color="#d8d1c7", linewidth=0.8, alpha=0.7)
    ax.axhline(0, linewidth=0.9, color="#8f877d", alpha=0.9, zorder=1)
    ax.axvline(0, linewidth=0.9, color="#8f877d", alpha=0.9, zorder=1)

    groups: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        group = group_assigner(label)
        groups.setdefault(group, []).append(i)

    if draw_group_ellipses:
        for group_name, indices in groups.items():
            group_points = coords[indices]
            ellipse_color = group_colors.get(group_name, "#7d7d7d")
            # Ellipses emphasize group-level separation without altering point positions.
            _add_group_ellipse(
                ax,
                group_points,
                color=ellipse_color,
                n_std=ellipse_n_std,
                alpha=ellipse_alpha,
                zorder=1,
            )

    for group_name, indices in groups.items():
        marker = group_markers.get(group_name, "o")
        color = group_colors.get(group_name, "#1f6f8b")

        x = coords[indices, 0]
        y = coords[indices, 1]

        ax.scatter(
            x,
            y,
            s=150,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=1.2,
            alpha=0.92,
            zorder=3,
            label=group_name,
        )

    for i, label in enumerate(labels):
        style = point_styler(label) if point_styler is not None else {}
        x = coords[i, 0]
        y = coords[i, 1]
        group_name = group_assigner(label)

        point_style = {
            "color": group_colors.get(group_name, "#1f6f8b"),
            "marker": group_markers.get(group_name, "o"),
            "edgecolors": "white",
            "linewidths": 1.2,
            "alpha": 0.92,
        }
        point_style.update(style)

        ax.scatter(x, y, s=150, zorder=4, **point_style)

        ax.annotate(
            display_labels[i],
            xy=(x, y),
            xytext=text_offset,
            textcoords="offset points",
            fontsize=11,
            color="#2f2a24",
            zorder=5,
        )

    if show_legend and len(groups) > 1:
        ax.legend(
            frameon=True,
            facecolor="white",
            edgecolor="#d8d1c7",
            fontsize=10,
            loc="best",
        )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        f"{title}\nStress = {mds.stress_:.4f}",
        fontsize=15,
        fontweight="semibold",
        color="#2f2a24",
        pad=16,
    )
    ax.set_xlabel("MDS dimension 1", fontsize=11, color="#4b443c")
    ax.set_ylabel("MDS dimension 2", fontsize=11, color="#4b443c")
    ax.tick_params(axis="both", labelsize=10, colors="#4b443c")

    fig.tight_layout()
    resolved_path = _resolve_output_path(
        save_path=save_path,
        title=title,
        suffix="mds",
        output_dir=output_dir,
        extension="pdf",
    )
    _finalize_figure(fig, save_path=resolved_path, dpi=dpi)

    return coords, float(mds.stress_)
