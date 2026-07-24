from __future__ import annotations

"""Lightweight catalog helpers for actor-level aggregated log data.

This module centralizes how analysis code resolves actor directories and the
canonical log files available for each dataset actor.
"""

from pathlib import Path
from src.core.shared.actor_catalog import (
    DatasetInput,
    analysis_actors as discover_analysis_actors,
    discover_actors,
    experiment_aggregated_dir,
)

LOG_FILE_NAMES = ("audit.log", "auth.log", "nextcloud.log", "syslog.log")


def known_actors(dataset: DatasetInput | str = "Nextcloud") -> tuple[str, ...]:
    """Return the actors available in the aggregated experiment dataset.

    This is a thin wrapper around the shared catalog to keep stats code
    independent of the lower-level discovery implementation.
    """
    return discover_actors(dataset)


def analysis_actors(dataset: DatasetInput | str = "Nextcloud") -> tuple[str, ...]:
    """Return the subset of actors intended for downstream analysis.

    The underlying catalog may exclude actors that are present on disk but not
    part of the analysis-ready split.
    """
    return discover_analysis_actors(dataset)


def actor_dir(actor: str, dataset: DatasetInput | str = "Nextcloud") -> Path:
    """Resolve the directory that stores aggregated logs for one actor.

    Paths are constructed relative to the dataset-specific aggregated experiment
    root used throughout the analysis code.
    """
    return experiment_aggregated_dir(dataset) / actor


def get_actor_logs(actor: str, dataset: DatasetInput | str = "Nextcloud") -> dict[str, Path]:
    """Return the canonical log-file mapping for a single actor.

    The keys use stable short names so downstream code can refer to logs
    uniformly without depending on filename formatting.
    """
    base = actor_dir(actor, dataset=dataset)
    return {
        "audit": base / "audit.log",
        "auth": base / "auth.log",
        "nextcloud": base / "nextcloud.log",
        "syslog": base / "syslog.log",
    }


def get_log_path(actor: str, log_name: str, dataset: DatasetInput | str = "Nextcloud") -> Path:
    """Return the path to one canonical log file for an actor.

    Accepts either the short log key or the filename with a `.log` suffix and
    raises `KeyError` for unknown log identifiers.
    """
    # Normalizing here keeps callers flexible while preserving a single internal key scheme.
    key = log_name.removesuffix(".log")
    logs = get_actor_logs(actor, dataset=dataset)

    if key not in logs:
        valid = ", ".join(sorted(logs))
        raise KeyError(f"Unknown log name: {log_name!r}. Expected one of: {valid}")

    return logs[key]


def list_known_actor_logs(dataset: DatasetInput | str = "Nextcloud") -> dict[str, dict[str, Path]]:
    """Return canonical log mappings for every known actor in a dataset.

    The result is convenient for iteration in analyses that need to enumerate
    actors and their log paths in one step.
    """
    return {actor: get_actor_logs(actor, dataset=dataset) for actor in known_actors(dataset)}
