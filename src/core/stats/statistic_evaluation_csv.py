from __future__ import annotations

"""Persist summary statistics for AI vs. human distance evaluations as CSV rows.

The file stores both flattened scalar metrics and JSON-encoded structured outputs so
downstream analysis can use a simple tabular format without losing detailed results.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---- Output schema ----
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "statistic_evaluation.csv"


CSV_COLUMNS = [
    "timestamp_utc",
    "approach",
    "distance_name",
    "ordered_labels_json",
    "hyperparameters_json",
    "group_summaries_json",
    "silhouette_json",
    "mannwhitney_json",
    "ai_ai_n",
    "ai_ai_mean",
    "ai_ai_median",
    "ai_ai_std",
    "ai_human_n",
    "ai_human_mean",
    "ai_human_median",
    "ai_human_std",
    "human_human_n",
    "human_human_mean",
    "human_human_median",
    "human_human_std",
    "silhouette_overall_mean",
    "silhouette_ai_mean",
    "silhouette_human_mean",
    "mw_ai_human_vs_ai_ai_u",
    "mw_ai_human_vs_ai_ai_p",
    "mw_ai_human_vs_ai_ai_cliffs_delta",
    "mw_ai_human_vs_human_human_u",
    "mw_ai_human_vs_human_human_p",
    "mw_ai_human_vs_human_human_cliffs_delta",
]


def _json_dumps(value: Any) -> str:
    """Serialize nested results deterministically for stable CSV storage.

    Keys are sorted and output is ASCII-safe so repeated runs remain easy to diff
    and robust across downstream tooling.
    """
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def append_statistic_evaluation_row(
    *,
    approach: str,
    distance_name: str,
    ordered_labels: list[str],
    hyperparameters: dict[str, Any],
    group_stats: dict[str, Any],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Append one evaluation result to the statistics CSV and return its path.

    The input is expected to follow the grouped summary structure produced by the
    statistical evaluation pipeline, with both scalar summaries and test outputs.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Keep the nested outputs intact for later inspection while also exposing the
    # most commonly queried summary metrics as flat columns.
    summaries = group_stats["group_summaries"]
    silhouette = group_stats["silhouette"]
    tests = group_stats["mannwhitney"]

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "approach": approach,
        "distance_name": distance_name,
        "ordered_labels_json": _json_dumps(ordered_labels),
        "hyperparameters_json": _json_dumps(hyperparameters),
        "group_summaries_json": _json_dumps(summaries),
        "silhouette_json": _json_dumps(silhouette),
        "mannwhitney_json": _json_dumps(tests),
        "ai_ai_n": summaries["ai_ai"]["n"],
        "ai_ai_mean": summaries["ai_ai"]["mean"],
        "ai_ai_median": summaries["ai_ai"]["median"],
        "ai_ai_std": summaries["ai_ai"]["std"],
        "ai_human_n": summaries["ai_human"]["n"],
        "ai_human_mean": summaries["ai_human"]["mean"],
        "ai_human_median": summaries["ai_human"]["median"],
        "ai_human_std": summaries["ai_human"]["std"],
        "human_human_n": summaries["human_human"]["n"],
        "human_human_mean": summaries["human_human"]["mean"],
        "human_human_median": summaries["human_human"]["median"],
        "human_human_std": summaries["human_human"]["std"],
        "silhouette_overall_mean": silhouette["overall_mean"],
        "silhouette_ai_mean": silhouette["ai_mean"],
        "silhouette_human_mean": silhouette["human_mean"],
        "mw_ai_human_vs_ai_ai_u": tests["ai_human_vs_ai_ai"]["u_statistic"],
        "mw_ai_human_vs_ai_ai_p": tests["ai_human_vs_ai_ai"]["p_value"],
        "mw_ai_human_vs_ai_ai_cliffs_delta": tests["ai_human_vs_ai_ai"]["cliffs_delta"],
        "mw_ai_human_vs_human_human_u": tests["ai_human_vs_human_human"]["u_statistic"],
        "mw_ai_human_vs_human_human_p": tests["ai_human_vs_human_human"]["p_value"],
        "mw_ai_human_vs_human_human_cliffs_delta": tests["ai_human_vs_human_human"]["cliffs_delta"],
    }

    # Write the header only once so repeated appends remain valid CSV exports.
    file_exists = output.exists()
    with output.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return output
