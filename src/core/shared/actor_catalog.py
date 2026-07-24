"""Utilities for resolving dataset-specific actor directories and groups.

This module centralizes the mapping from dataset aliases to canonical names and
the discovery of human versus AI actor folders used in downstream analyses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal


DatasetName = Literal["Nextcloud", "WordPress"]
DatasetInput = Literal["Nextcloud", "WordPress", "Data", "Data_WP"]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
AI_MARKER = "GPT"


def normalize_dataset_name(dataset: DatasetInput | str) -> DatasetName:
    """Map accepted dataset aliases to the canonical dataset name.

    This keeps path construction consistent across modules that still use legacy
    identifiers. Raises a `ValueError` for unknown dataset labels.
    """

    # Historical aliases are normalized here so callers can remain permissive
    # while filesystem access stays tied to a single canonical layout.
    aliases: dict[str, DatasetName] = {
        "Nextcloud": "Nextcloud",
        "WordPress": "WordPress",
        "Data": "Nextcloud",
        "Data_WP": "WordPress",
    }
    try:
        return aliases[dataset]
    except KeyError as exc:
        valid = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unknown dataset={dataset!r}. Expected one of: {valid}"
        ) from exc


def experiment_aggregated_dir(dataset: DatasetInput | str = "Nextcloud") -> Path:
    """Return the aggregated experiment directory for a dataset.

    The input may use either the canonical dataset name or a supported legacy
    alias. The returned path is expected to contain one directory per actor.
    """

    canonical_dataset = normalize_dataset_name(dataset)
    return PROJECT_ROOT / "data" / canonical_dataset / "combine" / "ExperimentAggregated"


def is_ai_actor(actor: str, *, ai_marker: str = AI_MARKER) -> bool:
    """Identify whether an actor label should be treated as AI-generated."""

    return ai_marker in actor


def discover_actors(dataset: DatasetInput | str = "Nextcloud") -> tuple[str, ...]:
    """Discover actor directory names for the given dataset.

    The function reads the aggregated experiment directory and returns actor
    names in sorted order. It raises if the directory is missing or empty.
    """

    root = experiment_aggregated_dir(dataset)
    if not root.exists():
        raise FileNotFoundError(f"Actor root not found: {root}")

    actors = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not actors:
        raise FileNotFoundError(f"No actor directories found under: {root}")
    return tuple(actors)


def discover_actor_groups(
    dataset: DatasetInput | str = "Nextcloud",
    *,
    ai_marker: str = AI_MARKER,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split discovered actors into human and AI groups.

    The grouping assumes AI actors are identified by a marker substring in the
    directory name. Returns `(human_groups, ai_groups)` and raises if either
    side is missing.
    """

    actors = discover_actors(dataset)
    # The split is name-based by design so directory naming remains the single
    # source of truth for downstream human-vs-AI comparisons.
    human_groups = tuple(actor for actor in actors if ai_marker not in actor)
    ai_groups = tuple(actor for actor in actors if ai_marker in actor)

    if not human_groups:
        raise ValueError(f"No human actor directories found for dataset={dataset!r}")
    if not ai_groups:
        raise ValueError(f"No AI actor directories found for dataset={dataset!r}")

    return human_groups, ai_groups


def analysis_actors(dataset: DatasetInput | str = "Nextcloud") -> tuple[str, ...]:
    """Return actors in the analysis order used downstream.

    Human groups are placed first, followed by AI groups, to provide a stable
    and explicit ordering for comparative analyses.
    """

    human_groups, ai_groups = discover_actor_groups(dataset)
    return human_groups + ai_groups
