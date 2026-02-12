import re
from pathlib import Path
from typing import Pattern, Sequence, List, Dict, Any


def strip_outer_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def extract_rows_from_file(
    file_path: str | Path,
    *,
    prefix_regex: Pattern[str],
    keys: Sequence[str],
    value_pattern: str = r"\S+",
    require_all_keys: bool = False,
    ignore_case: bool = False,
    keep_line: bool = True,
    strip_quotes: bool = True,
) -> List[Dict[str, Any]]:
    """
    Return row-aligned extracted data: one dict per matched log line.

    Each row contains:
      - "_lineno": 1-based line number in the file
      - "_line": original line (if keep_line=True)
      - extracted keys found in that line: {key: value, ...}

    If require_all_keys=True, only rows containing ALL keys are kept.
    """
    path = Path(file_path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    flags = re.IGNORECASE if ignore_case else 0

    key_regexes = {
        k: re.compile(rf"\b{re.escape(k)}=(?P<val>{value_pattern})(?=\s|$)", flags=flags)
        for k in keys
    }

    rows: List[Dict[str, Any]] = []

    for lineno, line in enumerate(lines, start=1):
        if not prefix_regex.search(line):
            continue

        found: Dict[str, str] = {}
        for k, rx in key_regexes.items():
            m = rx.search(line)
            if m:
                val = m.group("val")
                if strip_quotes:
                    val = strip_outer_quotes(val)
                found[k] = val

        if require_all_keys and any(k not in found for k in keys):
            continue

        row: Dict[str, Any] = {"_lineno": lineno}
        if keep_line:
            row["_line"] = line
        row.update(found)

        rows.append(row)

    return rows


if __name__ == "__main__":
    # ---- configure here ----

    file_1 = "/home/lorenz/Documents/llm-admin-shell/ExperimentResult/combine/AllAggregated/AI/audit.log"
    file_2 = "/home/lorenz/Documents/llm-admin-shell/ExperimentResult/combine/AllAggregated/Human/audit.log"

    # Choose one prefix + keys set (toggle as you like)

    # EXECVE
    prefix = re.compile(r"^type=EXECVE\s+msg=audit\(")
    keys = ["a0", "a1", "a2"]

    # PATH
    # prefix = re.compile(r"^type=PATH\s+msg=audit\(")
    # keys = ["name", "nametype"]

    # SYSCALL
    # prefix = re.compile(r"^type=SYSCALL\s+msg=audit\(")
    # keys = ["syscall", "success", "exit", "comm", "exe", "auid", "uid", "tty", "key"]

    # SOCKADDR
    # prefix = re.compile(r"^type=SOCKADDR\s+msg=audit\(")
    # keys = ["saddr"]

    # ---- extract rows ----
    rows1 = extract_rows_from_file(
        file_1,
        prefix_regex=prefix,
        keys=keys,
        require_all_keys=False,  # keep rows even if some keys are missing
        keep_line=True,
        strip_quotes=True,
    )

    rows2 = extract_rows_from_file(
        file_2,
        prefix_regex=prefix,
        keys=keys,
        require_all_keys=False,
        keep_line=True,
        strip_quotes=True,
    )

    # ---- demo: print a specific row (change index as needed) ----
    i = 23
    if i < len(rows1):
        print("FILE 1 row", i)
        print(rows1[i])
        print()

    if i < len(rows2):
        print("FILE 2 row", i)
        print(rows2[i])
        print()

    # ---- demo: show first few rows ----
    # for r in rows1[:5]:
    #     print(r)
