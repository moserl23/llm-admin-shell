import re
from pathlib import Path
from typing import Dict, List, Sequence, Pattern, Optional
from collections import defaultdict, Counter
import matplotlib.pyplot as plt





# ----------------------------------------------------------------------
# Order-insensitive extractor for key=value pairs (auditd-friendly)
# ----------------------------------------------------------------------

def extract_kv_from_file(
    file_path: str | Path,
    *,
    prefix_regex: Pattern[str],
    keys: Sequence[str] = ("arch", "syscall", "success"),
    # value pattern: default matches non-whitespace token (works for arch=c000003e, syscall=42, success=yes)
    value_pattern: str = r"\S+",
    require_all_keys: bool = True,
    ignore_case: bool = False,
) -> Dict[str, List[str]]:
    """
    Read a log file and extract values for the given keys from lines that match `prefix_regex`,
    without requiring any particular ordering of the key=value pairs.

    Parameters
    ----------
    file_path:
        Path to the log file.
    prefix_regex:
        Compiled regex that must match for a line to be considered (e.g. only SYSCALL lines).
        You can include named groups here (e.g. (?P<audit>...)) if you want; they are ignored by default.
    keys:
        Keys to extract, e.g. ["arch","syscall","success"].
    value_pattern:
        Regex for the value part (without capture parentheses). Default: \\S+.
        If you want to support quoted values, see `QUOTED_OR_BARE_VALUE_PATTERN` below.
    require_all_keys:
        If True, only lines that contain ALL keys are used.
        If False, lines can contribute to some keys even if others are missing.
    ignore_case:
        If True, key matching is case-insensitive (useful if logs sometimes have ARCH=... etc).

    Returns
    -------
    Dict[str, List[str]]:
        { key1: [v1, v2, ...], key2: [...], ... }
    """
    path = Path(file_path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    flags = re.IGNORECASE if ignore_case else 0

    # Compile per-key regex patterns. Use word boundary to avoid matching e.g. "syscallx=".
    key_regexes: Dict[str, Pattern[str]] = {
        k: re.compile(rf"\b{re.escape(k)}=(?P<val>{value_pattern})(?=\s|$)", flags=flags)
        for k in keys
    }

    out: Dict[str, List[str]] = defaultdict(list)

    for line in lines:
        if not prefix_regex.search(line):
            continue

        found: Dict[str, str] = {}

        for k, rx in key_regexes.items():
            m = rx.search(line)
            if m:
                found[k] = m.group("val")

        if require_all_keys and any(k not in found for k in keys):
            continue

        for k in keys:
            if k in found:
                out[k].append(found[k])

    return dict(out)


# ----------------------------------------------------------------------
# Optional helper patterns
# ----------------------------------------------------------------------

# If you later want to support values that might be quoted ("cron") OR bare (cron),
# you can use this pattern for value_pattern:
#   - captures quoted contents without quotes
#   - or captures bare token
#
# BUT: since extract_kv_from_file uses one named group (?P<val>...), we need a pattern
# that does not introduce extra capture groups. So we use non-capturing groups (?:...).
#
# It will return either "cron" (without quotes) or cron, depending on input.
QUOTED_OR_BARE_VALUE_PATTERN = r'(?:\"[^"]*\"|\S+)'


def strip_quotes(values: Dict[str, List[str]]) -> Dict[str, List[str]]:
    def _strip(v: str) -> str:
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            return v[1:-1]
        return v

    return {k: [_strip(x) for x in vs] for k, vs in values.items()}


def value_frequencies(d: Dict[str, List[str]]) -> Dict[str, Counter]:
    """Count frequencies per placeholder."""
    return {k: Counter(v) for k, v in d.items()}


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Only consider auditd SYSCALL lines
    # Example line prefix:
    #   type=SYSCALL msg=audit(1765984141.124:2367):

    #file_1 = "/home/lorenz/Documents/llm-admin-shell/Evaluation/marvin_big_log1.log"
    #file_2 = "/home/lorenz/Documents/llm-admin-shell/Evaluation/marvin_big_log2.log"

    file_1 = "/home/lorenz/Documents/llm-admin-shell/ExperimentResult/combine/AllAggregated/AI/audit.log"
    file_2 = "/home/lorenz/Documents/llm-admin-shell/ExperimentResult/combine/AllAggregated/Human/audit.log"

    if False:
        prefix = re.compile(r"^type=EXECVE\s+msg=audit\(")

        keys = [
            "a0",  # argv[0] (binary)
            "a1",  # first argument
            "a2",  # second argument
        ]
    elif False:
        # PATH: filesystem targets
        prefix = re.compile(r"^type=PATH\s+msg=audit\(")

        keys = [
            "name",      # file or directory
            "nametype",  # NORMAL / DELETE / PARENT
        ]
    elif False:
        prefix = re.compile(r"^type=SYSCALL\s+msg=audit\(")

        keys = [
            "syscall",   # numeric syscall id
            "success",   # yes / no
            "exit",      # return code
            "comm",      # short command name
            "exe",       # full executable path
            "auid",      # original login user
            "uid",       # effective user
            "tty",       # interactive vs daemon
            "key",       # MITRE / rule tag if present
        ]
    elif True:
        # SOCKADDR: networking targets
        prefix = re.compile(r"^type=SOCKADDR\s+msg=audit\(")

        keys = [
            "saddr",
        ]

    d1 = extract_kv_from_file(
        file_1,
        prefix_regex=prefix,
        keys=keys,
        value_pattern=r"\S+",
        require_all_keys=False,
        ignore_case=False,
    )

    d2 = extract_kv_from_file(
        file_2,
        prefix_regex=prefix,
        keys=keys,
        value_pattern=r"\S+",
        require_all_keys=False,
        ignore_case=False,
    )

    d1 = strip_quotes(d1)
    d2 = strip_quotes(d2)

    #print("FILE 1 extracted:")
    #print({k: (len(v), v[:10]) for k, v in d1.items()})  # show counts + first 10
    #print()

    #print("FILE 2 extracted:")
    #print({k: (len(v), v[:10]) for k, v in d2.items()})
    #print()

    for key in keys:
        
        TOP_K = 30

        # counts for this field in each file
        c1 = Counter(d1.get(key, []))
        c2 = Counter(d2.get(key, []))

        # unique values
        only_1 = set(c1) - set(c2)
        only_2 = set(c2) - set(c1)

        # rank by frequency (highest first)
        top1 = sorted(only_1, key=lambda v: c1[v], reverse=True)[:TOP_K]
        top2 = sorted(only_2, key=lambda v: c2[v], reverse=True)[:TOP_K]

        print(f"Top 10 unique {key} only in file 1 (value, count):")
        for v in top1:
            print(f"  {v}, {c1[v]}")

        print(f"\nTop 10 unique {key} only in file 2 (value, count):")
        for v in top2:
            print(f"  {v}, {c2[v]}")

        print()
        
        # Compare frequencies
        c1 = Counter(d1.get(key, []))
        c2 = Counter(d2.get(key, []))

        n1 = sum(c1.values())
        n2 = sum(c2.values())

        # Avoid division by zero
        if n1 == 0 or n2 == 0:
            print(f"Skipping {key}: no data (n1={n1}, n2={n2})")
            continue

        # rate difference: (count/n) in file1 minus (count/n) in file2
        rate_diff = {
            v: (c1[v] / n1) - (c2[v] / n2)
            for v in (set(c1) | set(c2))
            if c1[v] != 0 or c2[v] != 0
        }

        # Sort by absolute rate difference
        top_signed = sorted(rate_diff.items(), key=lambda x: abs(x[1]), reverse=True)[:TOP_K]

        #print(f"\nTop {TOP_K} {key} RATE differences (file1 - file2), ranked by |difference|:")
        '''
        for val, d in top_signed:
            print(
                f"  {val}: {d:+.4f}  "
                f"(file1={c1[val]}/{n1}={c1[val]/n1:.4f}, file2={c2[val]}/{n2}={c2[val]/n2:.4f})"
            )
        '''
        
        # ---- Plot ----
        if top_signed:
            labels = [val for val, _ in top_signed][::-1]
            values = [d for _, d in top_signed][::-1]

            plt.figure(figsize=(12, 6))
            bars = plt.barh(labels, values)
            plt.axvline(0)

            plt.xlabel("Rate difference (file1 − file2)")
            plt.title(f"Top {TOP_K} {key} frequency differences (normalized)")

            for bar in bars:
                width = bar.get_width()
                y = bar.get_y() + bar.get_height() / 2

                offset = 0.0005  # tweak if needed

                if width >= 0:
                    x = width + offset
                    ha = "left"
                else:
                    x = width - offset
                    ha = "right"

                plt.text(
                    x,
                    y,
                    f"{width:+.3f}",
                    va="center",
                    ha=ha,
                    fontsize=9,
                )

            plt.tight_layout()
            plt.show()
