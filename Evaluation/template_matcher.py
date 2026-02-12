import re
from pathlib import Path
from typing import Dict, List
from collections import defaultdict, Counter


# ------------------------------------------------------------
# 1. Template → regex compiler (Drain-style <*> wildcards)
# ------------------------------------------------------------

def compile_template_named(template: str, field_names: List[str]) -> re.Pattern:
    escaped = re.escape(template)

    # Non-capturing wildcard (audit metadata)
    escaped = escaped.replace(r"<AUDIT>", r"\S+")

    # Named wildcards
    for name in field_names:
        escaped = escaped.replace(r"<\*>", rf"(?P<{name}>\S+)", 1)

    return re.compile(rf"^{escaped}(?:\s+.*)?$")



# ------------------------------------------------------------
# 2. Extract placeholder values from a single file
# ------------------------------------------------------------

def extract_placeholders_from_file(
    file_path: str | Path,
    template: str,
    field_names: List[str],
) -> Dict[str, List[str]]:
    """
    Reads a log file, applies the template regex,
    and collects placeholder values.

    Returns:
        {
            field_1: [v1, v2, ...],
            field_2: [v1, v2, ...],
            ...
        }
    """
    rx = compile_template_named(template, field_names)

    values: Dict[str, List[str]] = defaultdict(list)

    lines = Path(file_path).read_text(encoding="utf-8").splitlines()

    for line in lines:
        m = rx.match(line)
        if not m:
            continue

        for name in field_names:
            values[name].append(m.group(name))

    return dict(values)


# ------------------------------------------------------------
# 3. Convenience: extract for TWO files
# ------------------------------------------------------------

def extract_from_two_files(
    file_1: str | Path,
    file_2: str | Path,
    template: str,
    field_names: List[str],
) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Applies the same template to two files.
    """
    d1 = extract_placeholders_from_file(file_1, template, field_names)
    d2 = extract_placeholders_from_file(file_2, template, field_names)
    return d1, d2


# ------------------------------------------------------------
# 4. Optional helpers for comparison
# ------------------------------------------------------------

def unique_values(d: Dict[str, List[str]]) -> Dict[str, set]:
    """Convert lists to sets (unique values per placeholder)."""
    return {k: set(v) for k, v in d.items()}


def value_frequencies(d: Dict[str, List[str]]) -> Dict[str, Counter]:
    """Count frequencies per placeholder."""
    return {k: Counter(v) for k, v in d.items()}


# ------------------------------------------------------------
# 5. Example usage (can be removed if importing as library)
# ------------------------------------------------------------

if __name__ == "__main__":

    template = (
        'type=SYSCALL msg=audit(<AUDIT>): '
        'arch=<*> syscall=<*> success=<*>'
    )

    fields = ["arch", "syscall", "success"]

    file_1 = "/home/lorenz/Documents/llm-admin-shell/ExperimentResult/LOGS_Result_Armin/exp1/audit.log"
    file_2 = "/home/lorenz/Documents/llm-admin-shell/Evaluation/marvin_big_log2.log"

    d1, d2 = extract_from_two_files(file_1, file_2, template, fields)

    print("FILE 1:")
    print(d1)
    print()

    print("FILE 2:")
    print(d2)
    print()

    # Example comparisons
    print("Unique syscalls only in file 1:")
    print(set(d1.get("syscall", [])) - set(d2.get("syscall", [])))

    print("\nSyscall frequency difference (file1 - file2):")
    print(Counter(d1.get("syscall", [])) - Counter(d2.get("syscall", [])))
