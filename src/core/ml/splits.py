"""Utilities for constructing train/validation/test splits.

Supports standard random splits, group-aware splits, and predefined
group assignments for fixed evaluation protocols.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Set
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit

@dataclass(frozen=True)
class Split:
    """Container for train, validation, and test index arrays."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def _split_by_predefined_groups(
    groups: np.ndarray,
    *,
    val_groups: Sequence[str],
    test_groups: Sequence[str],
) -> Split:
    """Build a split from explicitly assigned validation and test groups.

    Group labels are matched after string conversion to avoid dtype-dependent
    behavior. Any sample not assigned to validation or test remains in train.
    """
    if groups is None:
        raise ValueError("groups must be provided when using predefined group splits")

    val_set: Set[str] = set(map(str, val_groups))
    test_set: Set[str] = set(map(str, test_groups))

    overlap = val_set & test_set
    if overlap:
        raise ValueError(f"val_groups and test_groups overlap: {sorted(overlap)}")

    # Normalize group labels before matching so predefined splits are stable
    # across mixed input types such as numeric ids and strings.
    groups_str = np.array(list(map(str, groups.tolist())), dtype=object)

    val_mask = np.isin(groups_str, list(val_set))
    test_mask = np.isin(groups_str, list(test_set))
    train_mask = ~(val_mask | test_mask)

    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    train_idx = np.where(train_mask)[0]

    if val_idx.size == 0:
        raise ValueError(f"Predefined split produced empty VAL. val_groups={sorted(val_set)}")
    if test_idx.size == 0:
        raise ValueError(f"Predefined split produced empty TEST. test_groups={sorted(test_set)}")
    if train_idx.size == 0:
        raise ValueError("Predefined split produced empty TRAIN (you assigned all groups to val/test)")

    return Split(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def make_splits(
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    *,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True,
    val_groups: Optional[Sequence[str]] = None,
    test_groups: Optional[Sequence[str]] = None,
) -> Split:
    """Create train/validation/test splits under random, grouped, or fixed protocols.

    When predefined group lists are supplied, they take precedence over random
    splitting. Otherwise, samples are split either with stratification or with
    group preservation, depending on whether `groups` is provided.
    """
    idx = np.arange(len(y))

    # ---- Predefined group split ----
    if val_groups is not None or test_groups is not None:
        if val_groups is None or test_groups is None:
            raise ValueError("Provide both val_groups and test_groups (or neither).")
        return _split_by_predefined_groups(groups, val_groups=val_groups, test_groups=test_groups)

    # ---- Standard sample-wise split ----
    if groups is None:
        strat = y if stratify else None
        train_val_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=random_state, stratify=strat
        )
        # The second split is taken from the remaining train+val pool, so the
        # validation fraction must be rescaled to preserve the requested global size.
        val_frac_of_train_val = val_size / (1.0 - test_size)
        strat_train_val = y[train_val_idx] if stratify else None
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_frac_of_train_val,
            random_state=random_state,
            stratify=strat_train_val,
        )
        return Split(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    # ---- Group-aware split ----
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss1.split(idx, y, groups=groups))

    # Validation is sampled from the post-test remainder while keeping groups intact.
    val_frac_of_train_val = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac_of_train_val, random_state=random_state)
    train_rel, val_rel = next(gss2.split(train_val_idx, y[train_val_idx], groups=groups[train_val_idx]))

    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]
    return Split(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
