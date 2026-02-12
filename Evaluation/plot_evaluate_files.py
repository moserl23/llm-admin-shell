import os
import re
from typing import Callable, List, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

# custom
from Evaluation.evaluation_class import Evaluation
import all_file_paths


def make_labels(files: List[str]) -> List[str]:
    """
    Create short, unique labels like 'Armin', 'Benni', ... from paths such as:
    .../LOGS_Result_Armin/exp7/LLM_audit.log
    Falls back to basename if pattern not found.
    """
    labels: List[str] = []
    pat = re.compile(r"LOGS_Result_([^/]+)")
    for f in files:
        f_str = str(f)
        m = pat.search(f_str)
        labels.append(m.group(1) if m else os.path.basename(f_str))
    return labels


def _to_nonneg_int(x) -> int:
    """
    Convert evaluator output to a non-negative int.
    - bool -> 0/1
    - int  -> int
    Raises if it's something unexpected.
    """
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    raise TypeError(f"Evaluator returned {type(x)}; expected bool or int.")


def make_eval_fn(method_name: str) -> Callable[[str, str], int]:
    """
    Returns a function f(file1, file2) -> int
    that runs Evaluation.<method_name>() with proper setup.
    """
    def run(file1: str, file2: str) -> int:
        ev = Evaluation()
        ev.set_files(file1, file2)

        # Only build templates for methods that require them
        needs_templates = method_name in {
            "n_gram_evaluate",
            "one_gram_evaluate",
            "complexity_index_evaluate",
        }
        if needs_templates:
            ev.build_templates()

        result = getattr(ev, method_name)()
        return max(0, _to_nonneg_int(result))

    run.__name__ = method_name
    return run


def plot_pairwise_differences(
    files: List[str],
    difference_functions: List[Callable[[str, str], int]],
    labels: List[str] | None = None,
    title: str = "Pairwise file differences",
):
    """
    Produces an NxN heatmap where each cell is the SUM of integer "hits"
    returned by all evaluation functions for that file pair.
    """
    if not files:
        raise ValueError("files is empty")
    if not difference_functions:
        raise ValueError("difference_functions is empty")

    n_files = len(files)

    if labels is None:
        labels = make_labels(files)
    if len(labels) != n_files:
        raise ValueError("labels must have same length as files")

    M = np.zeros((n_files, n_files), dtype=int)

    def safe_call(fn: Callable[[str, str], int], f1: str, f2: str) -> int:
        try:
            return max(0, _to_nonneg_int(fn(f1, f2)))
        except Exception as e:
            print(f"[WARN] {getattr(fn, '__name__', 'diff_fn')} failed for:\n  {f1}\n  {f2}\n  error: {e}")
            return 0

    # Compute only upper triangle and mirror (assumes symmetry)
    for i in range(n_files):
        for j in range(i + 1, n_files):
            val = sum(safe_call(fn, files[i], files[j]) for fn in difference_functions)
            M[i, j] = val
            M[j, i] = val

    # Adapt color scale to observed max
    max_val = int(M.max())
    cmap = plt.get_cmap("viridis", max_val + 1 if max_val >= 0 else 1)
    bounds = np.arange(-0.5, max_val + 1.5, 1) if max_val >= 0 else np.array([-0.5, 0.5])
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(M, cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_title(f"{title} (0–{max_val})")
    ax.set_xlabel("Files")
    ax.set_ylabel("Files")

    ax.set_xticks(range(n_files))
    ax.set_yticks(range(n_files))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Write values in each cell
    for i in range(n_files):
        for j in range(n_files):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax, ticks=range(max_val + 1))
    cbar.set_label("Total # of hits across all evaluation functions")

    plt.tight_layout()
    plt.show()

    return M

def slice_paths(experiment_number: int, log_type: str)-> list:
    return [
        name_dict[experiment_number][log_type]
        for name_dict in all_file_paths.files.values()
    ]

if __name__ == "__main__":

    difference_functions = [
        make_eval_fn("event_time_evaluate"),        # returns 0/1
        make_eval_fn("n_gram_evaluate"),            # can return 0..#models
        #make_eval_fn("combo_detector_evaluate"),    # returns 0/1
        make_eval_fn("complexity_index_evaluate"),   # can return 0..#metrics
    ]

    files = slice_paths(7, "audit")

    labels = make_labels(files)

    plot_pairwise_differences(
        files=files,
        difference_functions=difference_functions,
        labels=labels,
        title="LLM audit differences (exp7)",
    )
