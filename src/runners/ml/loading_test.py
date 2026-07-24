"""Quick inspection runner for the ML loading and split pipeline.

This script loads preprocessed examples, constructs a fixed group-based split,
and prints summary statistics plus a small sample for sanity checking.
"""

from src.core.shared.loader import load_examples, LoadConfig
from src.core.ml.splits import make_splits
from src.core.ml.reporting import print_split_stats
import numpy as np

def main():
    """Load examples, build a predefined split, and print inspection output.

    The split is intentionally group-based to keep selected sources isolated
    between validation and test partitions for debugging and sanity checks.
    """
    # ---- Load data ----
    examples = load_examples(LoadConfig(
        log_files=("syslog.log",),   # Restrict the check to the syslog source.
        prefix_with_log_type=False,
        preprocess_mode="aggressive",
        # No explicit windowing here; downstream representations use Drain3 IDs.
        window_mode="none",
        window_size=50,
        window_stride=25,
        cid_prefix="CID",
    ))

    # ---- Extract labels and groups ----
    y = np.array([e.label for e in examples], dtype=object)
    groups = np.array([e.group for e in examples], dtype=object)

    # Keep specific groups fixed in validation and test to inspect whether the
    # loader and split logic preserve the intended source separation.
    split = make_splits(y, groups=groups, val_groups=["Alice", "GPT4.1"], test_groups=["GPT5", "GPT4.1_V2"])

    # ---- Report and spot-check ----
    print_split_stats(examples, split)

    print(examples[0:50])

    print(y[0:50])

    print(groups[0:50])

    print(len(y))

if __name__ == "__main__":
    main()
