from __future__ import annotations

"""Utilities for enumerating human/AI group combinations for evaluation.

The key distinction is between the observed assignment and null-style runs,
where actor labels may be randomized before validation/test splits are built.
"""

import random
from itertools import product
from typing import List, Tuple, Optional

from src.core.shared.loader import LoadConfig, resolve_human_ai_groups

DatasetName = str
SPLIT_SHUFFLE_SEED = 42


def make_human_ai_pairs(
    dataset: DatasetName = "Nextcloud",
    *,
    randomize_actor_labels: bool = False,
    assignment_idx: Optional[int] = None,
) -> List[List[str]]:
    """Return all human/AI group pairs under the current label assignment.

    If label randomization is enabled, pairs are built after the requested
    null assignment has been applied.
    """
    # ---- Resolve groups under the requested assignment ----
    cfg = LoadConfig(
        dataset=dataset,
        randomize_actor_labels=randomize_actor_labels,
        assignment_idx=assignment_idx,
    )
    human_groups, ai_groups = resolve_human_ai_groups(cfg)
    return [[h, a] for h, a in product(human_groups, ai_groups)]


def make_val_test_splits(
    dataset: DatasetName = "Nextcloud",
    *,
    randomize_actor_labels: bool = False,
    assignment_idx: Optional[int] = None,
) -> List[Tuple[List[str], List[str]]]:
    """Return all disjoint validation/test group combinations.

    Splits are constructed under the current assignment, including null
    permutations when requested, and enforce distinct human and AI groups
    across validation and test.
    """
    # ---- Resolve groups under the requested assignment ----
    cfg = LoadConfig(
        dataset=dataset,
        randomize_actor_labels=randomize_actor_labels,
        assignment_idx=assignment_idx,
    )
    human_groups, ai_groups = resolve_human_ai_groups(cfg)

    # ---- Enumerate valid val/test combinations ----
    splits: List[Tuple[List[str], List[str]]] = []
    for hv, av in product(human_groups, ai_groups):
        for ht, at in product(human_groups, ai_groups):
            # Validation and test must be disjoint within each actor type.
            if hv == ht:
                continue
            if av == at:
                continue
            splits.append(([hv, av], [ht, at]))

    # Keep ordering stable across runs while avoiding systematic pair ordering.
    random.Random(SPLIT_SHUFFLE_SEED).shuffle(splits)
    return splits
