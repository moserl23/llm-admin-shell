"""Load log data into `Example` objects for downstream ML experiments.

The loader centralizes dataset resolution, optional line normalization, and
alternative windowing schemes including template- and timing-based views.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Literal

import json
import re
from datetime import datetime

import numpy as np

from src.core.ml.data import Example
from src.core.ml.env import load_project_env
from src.core.shared.actor_catalog import (
    DatasetInput,
    discover_actor_groups,
    experiment_aggregated_dir,
)

from itertools import combinations


# -----------------------------
# Config
# -----------------------------
PreprocessMode = Literal["raw", "soft", "aggressive", "template"]
WindowMode = Literal["none", "lines", "cids", "inter_times"]  # NEW
PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_project_env()

@dataclass(frozen=True)
class LoadConfig:
    """Configuration for dataset resolution, preprocessing, and windowing.

    The same loader supports raw-text, template-based, and timing-based views
    so experiments can switch representations without changing call sites.
    """
    # named dataset under PROJECT_ROOT; ignored when root is set explicitly
    dataset: DatasetInput = "Nextcloud"

    # explicit root wins over dataset/env resolution
    root: Optional[str] = None

    # which groups belong to which label; None => use dataset defaults
    human_groups: Optional[Tuple[str, ...]] = None
    ai_groups: Optional[Tuple[str, ...]] = None

    # which log files to load inside each group directory
    log_files: Tuple[str, ...] = ("audit.log", "nextcloud.log", "syslog.log")

    # hygiene / scaling knobs
    drop_empty: bool = True
    strip: bool = True
    max_lines_per_file: Optional[int] = None  # None = no limit
    prefix_with_log_type: bool = True
    encoding: str = "utf-8"
    errors: str = "replace"

    # preprocessing control
    preprocess_mode: PreprocessMode = "raw"  # raw | soft | aggressive | template

    # Drain3 template mining (only used when preprocess_mode="template" or window_mode="cids")
    drain_ini_path: Optional[str] = None  # None -> core/drain3.ini next to this file

    # windowing control
    window_mode: WindowMode = "none"       # none | lines | cids | inter_times
    window_size: int = 1
    window_stride: Optional[int] = None
    window_drop_last: bool = True

    # line windows
    join_token: str = " <EOL> "

    # cid windows
    cid_prefix: str = "CID"

    # inter-event time windows (NEW)
    inter_time_unit: str = "seconds"                 # seconds | log10_seconds
    inter_time_clip_max: Optional[float] = None      # e.g. 3600 to clip huge gaps
    inter_time_add_epsilon: float = 0.0              # e.g. 1e-6 if using log10
    inter_time_join_token: str = " "                 # serialize diffs into Example.text

    # if True, windows are formed *within each file* (recommended)
    # if False, lines are concatenated across log_files order per group
    window_within_each_file: bool = True

    # random / enumerated actor labeling for null hypothesis
    randomize_actor_labels: bool = False
    assignment_idx: Optional[int] = None

def _default_data_root(dataset: DatasetInput | str) -> Path:
    """Return the canonical aggregated-data directory for a dataset."""
    return experiment_aggregated_dir(dataset)

def get_num_actor_label_assignments(dataset: DatasetInput | str) -> int:
    """Return the number of non-observed human/AI label assignments.

    This excludes the true assignment because the null setting only enumerates
    alternative relabelings of the actor groups.
    """
    human_groups, ai_groups = _default_groups(dataset)
    n_total = len(human_groups) + len(ai_groups)
    n_human = len(human_groups)

    from math import comb
    return comb(n_total, n_human) - 1 # -1 because we don't want the actual correct assignment


def _default_groups(dataset: DatasetInput | str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return the dataset-defined human and AI actor groups."""
    return discover_actor_groups(dataset)


def _resolve_data_root(cfg: LoadConfig) -> Path:
    """Resolve the data root from explicit config, environment, or dataset defaults."""
    if cfg.root is not None:
        return Path(cfg.root)

    env_root = os.environ.get("DATAANALYSIS_DATA_ROOT")
    if env_root:
        return Path(env_root)

    return _default_data_root(cfg.dataset)


def _resolve_groups(cfg: LoadConfig) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Resolve human/AI groups, optionally under a null relabeling.

    When randomization is enabled, the true human assignment is excluded so
    `assignment_idx` indexes only alternative human-vs-AI splits.
    """
    default_human_groups, default_ai_groups = _default_groups(cfg.dataset)

    human_groups = cfg.human_groups if cfg.human_groups is not None else default_human_groups
    ai_groups = cfg.ai_groups if cfg.ai_groups is not None else default_ai_groups

    if not cfg.randomize_actor_labels:
        return human_groups, ai_groups

    all_groups = tuple(sorted(set(human_groups) | set(ai_groups)))
    n_human = len(human_groups)

    original_human = tuple(sorted(human_groups))

    # Enumerate only null assignments so repeated evaluation never reuses the
    # observed human/AI split as a baseline condition.
    all_assignments = [
        comb for comb in combinations(all_groups, n_human)
        if tuple(sorted(comb)) != original_human
    ]
    n_assignments = len(all_assignments)

    if cfg.assignment_idx is None:
        raise ValueError(
            "randomize_actor_labels=True requires assignment_idx to be set."
        )

    if not (0 <= cfg.assignment_idx < n_assignments):
        raise ValueError(
            f"assignment_idx={cfg.assignment_idx} is out of range for dataset={cfg.dataset!r}. "
            f"Valid null range: 0..{n_assignments - 1}"
        )

    randomized_human = tuple(all_assignments[cfg.assignment_idx])
    randomized_ai = tuple(g for g in all_groups if g not in randomized_human)

    return randomized_human, randomized_ai


def resolve_human_ai_groups(cfg: LoadConfig) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Expose the resolved human/AI group split for external callers."""
    return _resolve_groups(cfg)

# -----------------------------
# Log type detection (filename-level)
# -----------------------------
def _infer_log_type(filename: str) -> str:
    """Infer the coarse log type from the filename."""
    fn = filename.lower()
    if "audit" in fn:
        return "audit"
    if "nextcloud" in fn:
        return "nextcloud"
    if "syslog" in fn:
        return "syslog"
    return "unknown"


# -----------------------------
# Reading
# -----------------------------
def _read_lines(path: Path, *, encoding: str, errors: str, max_lines: Optional[int]) -> Iterable[Tuple[int, str]]:
    """Yield `(line_number, text)` pairs from a file up to an optional limit."""
    with path.open("r", encoding=encoding, errors=errors) as f:
        for i, line in enumerate(f, start=1):
            if max_lines is not None and i > max_lines:
                break
            yield i, line


# -----------------------------
# Preprocessing regexes (shared)
# -----------------------------
_UNIX_PATH_RE = re.compile(r"(?:/[^ \t\n\r\"']+)+")
_IP_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")

_LONG_HEX_RE = re.compile(r"\b[0-9a-fA-F]{16,}\b")
_HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")

_NUM_RE = re.compile(r"\b\d+\b")

# audit: normalize msg=audit(....):
_AUDIT_MSG_RE = re.compile(r"msg=audit\([^\)]*\):")

# syslog-ish (not perfect, but practical)
_SYSLOG_RE = re.compile(
    r"^(?P<ts>"
    r"(?:[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})"
    r"|"
    r"(?:\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)"
    r")\s+"
    r"(?P<host>\S+)\s+"
    r"(?P<proc>[^:]+):\s*"
    r"(?P<msg>.*)$"
)

_PROC_PID_RE = re.compile(r"^(.+?)\[\d+\]$")


# -----------------------------
# Type detection from content (optional)
# -----------------------------
def _detect_type_from_line(line: str) -> str:
    """Infer a log type from line content when filename-based typing is unavailable."""
    s = line.lstrip()
    if s.startswith("type=") and "msg=audit(" in s:
        return "audit"
    if s.startswith("{") and '"reqId"' in s and '"app"' in s:
        return "nextcloud"
    if _SYSLOG_RE.match(s):
        return "syslog"
    return "generic"


# -----------------------------
# Preprocess variants
# -----------------------------
def _preprocess_generic(line: str, *, aggressive: bool) -> str:
    """Normalize common volatile tokens in unstructured log lines.

    Aggressive mode additionally masks standalone numbers to reduce variance
    from IDs and counters.
    """
    s = line.strip()
    if not s:
        return s
    s = _UNIX_PATH_RE.sub("/PATH", s)
    s = _IP_RE.sub("<IP>", s)
    s = _LONG_HEX_RE.sub("<HEX>", s)
    s = _HEX_RE.sub("<HEX>", s)
    if aggressive:
        s = _NUM_RE.sub("<NUM>", s)
    return s


def _preprocess_audit(line: str, *, aggressive: bool) -> str:
    """Normalize audit log lines while preserving their event structure.

    Audit logs are highly number-volatile, so numeric masking is applied even
    in the softer preprocessing modes.
    """
    s = line.strip()
    if not s:
        return s

    s = _AUDIT_MSG_RE.sub("msg=audit(<AUDIT_META>):", s)
    s = _UNIX_PATH_RE.sub("/PATH", s)
    s = _LONG_HEX_RE.sub("<HEX>", s)
    s = _HEX_RE.sub("<HEX>", s)
    s = _IP_RE.sub("<IP>", s)

    # audit is extremely number-volatile; even "soft" should replace many numbers
    s = _NUM_RE.sub("<NUM>", s)
    return s


def _preprocess_syslog(line: str, *, aggressive: bool) -> str:
    """Normalize syslog lines into a stable `<TS> <HOST> proc: msg` shape.

    The parser falls back to generic preprocessing when the line does not
    match the expected syslog structure.
    """
    s = line.strip()
    if not s:
        return s

    m = _SYSLOG_RE.match(s)
    if not m:
        print("Generic Fallback!")
        print("LINE:", repr(s))
        return _preprocess_generic(s, aggressive=aggressive)

    proc = m.group("proc")
    msg = m.group("msg")

    # Normalize process pid part: CRON[1576] -> CRON[<PID>]
    proc2 = _PROC_PID_RE.match(proc.strip())
    if proc2:
        proc = f"{proc2.group(1)}[<PID>]"

    msg = _UNIX_PATH_RE.sub("/PATH", msg)
    msg = _IP_RE.sub("<IP>", msg)
    msg = _LONG_HEX_RE.sub("<HEX>", msg)
    msg = _HEX_RE.sub("<HEX>", msg)

    # normalize durations
    msg = re.sub(r"\b\d+\.\d+s\b", "<DUR>", msg)
    msg = re.sub(r"\b\d+us\b", "<DUR>", msg)
    msg = re.sub(r"\b\d+ms\b", "<DUR>", msg)

    if aggressive:
        msg = re.sub(r"\b\d+\b", "<NUM>", msg)

    return f"<TS> <HOST> {proc}: {msg}"


def _preprocess_nextcloud(line: str, *, aggressive: bool) -> str:
    """Normalize structured Nextcloud JSON logs into a compact text view.

    Important fields are retained, while volatile tokens in URLs, messages,
    and exceptions are masked to improve cross-run comparability.
    """
    s = line.strip()
    if not s:
        return s

    try:
        obj = json.loads(s)
    except Exception:
        return _preprocess_generic(s, aggressive=aggressive)

    app = obj.get("app", "<APP>")
    level = obj.get("level", "<LEVEL>")
    method = obj.get("method", "<METHOD>")
    url = obj.get("url", "<URL>")
    msg = obj.get("message", "")

    # normalize URL query string
    url = re.sub(r"\?.*$", "?<QS>", str(url))

    msg = _IP_RE.sub("<IP>", str(msg))
    msg = _UNIX_PATH_RE.sub("/PATH", msg)
    msg = _LONG_HEX_RE.sub("<HEX>", msg)
    msg = _HEX_RE.sub("<HEX>", msg)
    msg = re.sub(r"'[^']+'@'[^']+'", "'<USER>'@'<HOST>'", msg)
    msg = re.sub(r"\bSQLSTATE\[[^\]]+\]\s*\[\d+\]", "SQLSTATE[<STATE>][<CODE>]", msg)

    exc = obj.get("exception")
    if isinstance(exc, dict):
        exc_name = exc.get("Exception", "<EXC>")
        if aggressive:
            exc_part = f"{exc_name}(<CODE>)"
        else:
            exc_code = exc.get("Code")
            exc_part = f"{exc_name}({exc_code})" if exc_code is not None else f"{exc_name}(<CODE>)"
    else:
        exc_part = "<NOEXC>"

    base = f"nextcloud app={app} level={level} {method} {url} exc={exc_part} msg={msg}"
    if aggressive:
        base = _NUM_RE.sub("<NUM>", base)
    return base


def _preprocess_line(line: str, *, mode: PreprocessMode, assumed_type: Optional[str] = None) -> str:
    """Dispatch a line to the log-type-specific preprocessing routine.

    Template mode intentionally reuses the soft normalization path so Drain3
    clusters on lightly normalized messages rather than raw text.
    """
    if mode == "raw":
        return line

    aggressive = (mode == "aggressive")  # soft => aggressive=False

    kind = assumed_type or _detect_type_from_line(line)
    if kind == "audit":
        return _preprocess_audit(line, aggressive=aggressive)
    if kind == "nextcloud":
        return _preprocess_nextcloud(line, aggressive=aggressive)
    if kind == "syslog":
        return _preprocess_syslog(line, aggressive=aggressive)
    return _preprocess_generic(line, aggressive=aggressive)


# -----------------------------
# Windowing helpers
# -----------------------------
def _make_windows_from_lines(
    lines: List[str],
    *,
    window_size: int,
    stride: Optional[int] = None,
    join_token: str = " <EOL> ",
    drop_last: bool = True,
) -> List[str]:
    """Group consecutive text lines into fixed-size serialized windows."""
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if stride is None:
        stride = window_size
    if stride <= 0:
        raise ValueError("stride must be > 0")

    out: List[str] = []
    n = len(lines)
    for start in range(0, n, stride):
        chunk = lines[start:start + window_size]
        if len(chunk) < window_size and drop_last:
            break
        if not chunk:
            continue
        out.append(join_token.join(chunk))
    return out


def _make_windows_from_cids(
    cids: List[int],
    *,
    window_size: int,
    stride: Optional[int] = None,
    prefix: str = "CID",
    drop_last: bool = True,
) -> List[str]:
    """Serialize fixed-size windows of Drain3 cluster IDs."""
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if stride is None:
        stride = window_size
    if stride <= 0:
        raise ValueError("stride must be > 0")

    out: List[str] = []
    n = len(cids)
    for start in range(0, n, stride):
        chunk = cids[start:start + window_size]
        if len(chunk) < window_size and drop_last:
            break
        if not chunk:
            continue
        out.append(" ".join(f"{prefix}{cid}" for cid in chunk))
    return out


# -----------------------------
# Timestamp extraction + inter-event diffs (NEW)
# -----------------------------
_NEXTCLOUD_TIME_RE = re.compile(r'"time"\s*:\s*"([^"]+)"')
_AUDIT_EVENT_RE = re.compile(r"audit\((\d+(?:\.\d+)?):(\d+)\)")

def _extract_nextcloud_timestamps(lines: List[str]) -> List[datetime]:
    """Extract and sort ISO timestamps from Nextcloud JSON log lines."""
    ts: List[datetime] = []
    for line in lines:
        m = _NEXTCLOUD_TIME_RE.search(line)
        if not m:
            continue
        try:
            ts.append(datetime.fromisoformat(m.group(1)))
        except Exception:
            continue
    ts.sort()
    return ts

def _extract_auditlog_timestamps(lines: List[str]) -> List[datetime]:
    """Return one timestamp per unique audit event bundle.

    Audit logs often emit multiple lines per event serial, so timing analysis
    uses the first timestamp observed for each serial only.
    """
    seen: set[int] = set()
    ts: List[datetime] = []
    for line in lines:
        m = _AUDIT_EVENT_RE.search(line)
        if not m:
            continue
        epoch_str, serial_str = m.group(1), m.group(2)
        try:
            serial = int(serial_str)
        except Exception:
            continue
        if serial in seen:
            continue
        try:
            t = datetime.fromtimestamp(float(epoch_str))
        except Exception:
            continue
        seen.add(serial)
        ts.append(t)
    ts.sort()
    return ts

def _extract_syslog_timestamps(lines: List[str]) -> List[datetime]:
    """Extract and sort syslog timestamps when the first token is ISO-formatted.

    Non-ISO syslog formats are intentionally ignored here rather than guessed
    to avoid mixing parsing assumptions into timing-based evaluations.
    """
    ts: List[datetime] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        first = s.split(" ", 1)[0]
        try:
            ts.append(datetime.fromisoformat(first))
        except Exception:
            continue
    ts.sort()
    return ts

def _extract_generic_timestamps(lines: List[str]) -> List[datetime]:
    """Extract sortable timestamps from generic logs using the first token."""
    ts: List[datetime] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        first = s.split(" ", 1)[0]
        try:
            ts.append(datetime.fromisoformat(first))
        except Exception:
            continue
    ts.sort()
    return ts

def _extract_timestamps(lines: List[str], *, assumed_type: str) -> List[datetime]:
    """Route timestamp extraction to the parser for the expected log type."""
    if assumed_type == "audit":
        return _extract_auditlog_timestamps(lines)
    if assumed_type == "nextcloud":
        return _extract_nextcloud_timestamps(lines)
    if assumed_type == "syslog":
        return _extract_syslog_timestamps(lines)
    return _extract_generic_timestamps(lines)

def _inter_event_diffs_seconds(timestamps: List[datetime]) -> np.ndarray:
    """Convert ordered timestamps into non-negative inter-event gaps in seconds."""
    if len(timestamps) < 2:
        return np.array([], dtype=np.float32)

    diffs: List[float] = []
    prev = timestamps[0]
    for cur in timestamps[1:]:
        dt = (cur - prev).total_seconds()
        if dt >= 0:
            diffs.append(float(dt))
        prev = cur

    return np.asarray(diffs, dtype=np.float32)

def _transform_diffs(
    diffs: np.ndarray,
    *,
    unit: str,
    clip_max: Optional[float],
    add_epsilon: float,
) -> np.ndarray:
    """Apply clipping and scaling to inter-event gaps for text serialization."""
    x = diffs.astype(np.float32)
    if clip_max is not None:
        x = np.clip(x, 0.0, float(clip_max))
    if add_epsilon and add_epsilon > 0:
        x = x + float(add_epsilon)

    if unit == "seconds":
        return x
    if unit == "log10_seconds":
        return np.log10(x)
    raise ValueError(f"Unknown inter_time_unit={unit!r}")


# -----------------------------
# Drain3 integration
# -----------------------------
def _get_drain_ini(cfg: LoadConfig) -> Path:
    """Resolve the Drain3 configuration file path and validate its presence."""
    if cfg.drain_ini_path is not None:
        p = Path(cfg.drain_ini_path)
    else:
        p = Path(__file__).resolve().parent / "drain3.ini"
    if not p.is_file():
        raise FileNotFoundError(f"Missing Drain3 config: {p}")
    return p


def _create_template_miner(*, ini_path: Path):
    """Create a non-persistent Drain3 miner from the configured ini file."""
    try:
        from drain3 import TemplateMiner
        from drain3.template_miner_config import TemplateMinerConfig
    except Exception as e:
        raise ImportError(
            "Drain3 is not installed. Install it with: pip install drain3"
        ) from e

    cfg = TemplateMinerConfig()
    cfg.load(str(ini_path))
    return TemplateMiner(config=cfg, persistence_handler=None)


def _assign_templates_and_cids_global(
    miner,
    preprocessed_lines: List[str],
) -> Tuple[List[str], List[int]]:
    """Assign Drain3 cluster IDs and recover the current template per line.

    Template strings can evolve as additional logs are seen, so the templates
    returned here reflect the miner state after the full sequence is processed.
    """
    cluster_ids: List[int] = []
    for line in preprocessed_lines:
        result = miner.add_log_message(line)
        cluster_ids.append(int(result["cluster_id"]))

    cid_to_template = {c.cluster_id: c.get_template() for c in miner.drain.clusters}
    templates = [cid_to_template[cid] for cid in cluster_ids]
    return templates, cluster_ids


# -----------------------------
# Main loader
# -----------------------------
def load_examples(cfg: LoadConfig = LoadConfig()) -> List[Example]:
    """Load log files and convert them into labeled `Example` instances.

    Depending on configuration, examples can represent raw lines, normalized
    text, template IDs, or inter-event-time windows.
    """
    root = _resolve_data_root(cfg)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    human_groups, ai_groups = _resolve_groups(cfg)

    label_by_group: Dict[str, str] = {}
    for g in human_groups:
        label_by_group[g] = "human"
    for g in ai_groups:
        label_by_group[g] = "ai"

    examples: List[Example] = []

    need_drain = (cfg.preprocess_mode == "template") or (cfg.window_mode == "cids")
    ini_path = _get_drain_ini(cfg) if need_drain else None

    miners_by_log_type: Dict[str, object] = {}
    if need_drain:
        assert ini_path is not None
        miners_by_log_type = {}

    for group, label in label_by_group.items():
        group_dir = root / group
        if not group_dir.exists():
            raise FileNotFoundError(f"Missing group directory: {group_dir}")

        # placeholders (not used unless you implement cross-file windows)
        cross_file_lines: List[Tuple[str, str, str, int]] = []
        cross_file_cids: List[Tuple[int, str, str, int]] = []

        for log_name in cfg.log_files:
            p = group_dir / log_name
            if not p.exists():
                raise FileNotFoundError(f"Missing log file: {p}")

            log_type = _infer_log_type(log_name)

            raw_lines: List[Tuple[int, str]] = []
            for line_no, line in _read_lines(
                p, encoding=cfg.encoding, errors=cfg.errors, max_lines=cfg.max_lines_per_file
            ):
                if cfg.strip:
                    line = line.strip("\n").strip("\r")
                if cfg.drop_empty and (not line or not line.strip()):
                    continue
                raw_lines.append((line_no, line))

            if not raw_lines:
                continue

            # Template mode mines on softly normalized text to reduce spurious
            # clusters while keeping the message structure informative.
            if cfg.preprocess_mode == "raw":
                pre_lines = [ln for _, ln in raw_lines]
            elif cfg.preprocess_mode == "soft":
                pre_lines = [_preprocess_line(ln, mode="soft", assumed_type=log_type) for _, ln in raw_lines]
            elif cfg.preprocess_mode == "aggressive":
                pre_lines = [_preprocess_line(ln, mode="aggressive", assumed_type=log_type) for _, ln in raw_lines]
            elif cfg.preprocess_mode == "template":
                pre_lines = [_preprocess_line(ln, mode="soft", assumed_type=log_type) for _, ln in raw_lines]
            else:
                raise ValueError(f"Unknown preprocess_mode={cfg.preprocess_mode}")

            # Keep one miner per log type so templates are shared across actors
            # without forcing unrelated log formats into the same cluster space.
            templates: Optional[List[str]] = None
            cids: Optional[List[int]] = None
            if need_drain:
                assert ini_path is not None
                miner = miners_by_log_type.get(log_type)
                if miner is None:
                    miner = _create_template_miner(ini_path=ini_path)
                    miners_by_log_type[log_type] = miner
                templates, cids = _assign_templates_and_cids_global(miner, pre_lines)

            # Downstream windowing works on either normalized text or templates.
            if cfg.preprocess_mode == "template":
                assert templates is not None
                base_texts = templates
            else:
                base_texts = pre_lines

            def maybe_prefix(s: str) -> str:
                return f"[{log_type}] {s}" if cfg.prefix_with_log_type else s

            # ---- Windowing within each file ----
            if cfg.window_mode == "none":
                for (line_no, _raw), txt in zip(raw_lines, base_texts):
                    examples.append(
                        Example(
                            text=maybe_prefix(txt),
                            label=label,
                            group=group,
                            log_type=log_type,
                            path=str(p),
                            line_no=line_no,
                        )
                    )

            elif cfg.window_mode == "lines":
                win_texts = _make_windows_from_lines(
                    [maybe_prefix(t) for t in base_texts],
                    window_size=cfg.window_size,
                    stride=cfg.window_stride,
                    join_token=cfg.join_token,
                    drop_last=cfg.window_drop_last,
                )

                stride = cfg.window_stride if cfg.window_stride is not None else cfg.window_size
                start_line_nos = [raw_lines[i][0] for i in range(0, len(raw_lines), stride)]
                start_line_nos = start_line_nos[:len(win_texts)]

                for wtxt, start_no in zip(win_texts, start_line_nos):
                    examples.append(
                        Example(
                            text=wtxt,
                            label=label,
                            group=group,
                            log_type=log_type,
                            path=str(p),
                            line_no=start_no,
                        )
                    )

            elif cfg.window_mode == "cids":
                assert cids is not None
                win_texts = _make_windows_from_cids(
                    cids,
                    window_size=cfg.window_size,
                    stride=cfg.window_stride,
                    prefix=cfg.cid_prefix,
                    drop_last=cfg.window_drop_last,
                )

                stride = cfg.window_stride if cfg.window_stride is not None else cfg.window_size
                start_line_nos = [raw_lines[i][0] for i in range(0, len(raw_lines), stride)]
                start_line_nos = start_line_nos[:len(win_texts)]

                for wtxt, start_no in zip(win_texts, start_line_nos):
                    examples.append(
                        Example(
                            text=maybe_prefix(wtxt),
                            label=label,
                            group=group,
                            log_type=log_type,
                            path=str(p),
                            line_no=start_no,
                        )
                    )

            elif cfg.window_mode == "inter_times":
                # Timing features must be extracted from raw log text because
                # preprocessing may remove or rewrite the timestamp field.
                raw_texts = [ln for _, ln in raw_lines]
                ts = _extract_timestamps(raw_texts, assumed_type=log_type)

                diffs = _inter_event_diffs_seconds(ts)
                diffs = _transform_diffs(
                    diffs,
                    unit=cfg.inter_time_unit,
                    clip_max=cfg.inter_time_clip_max,
                    add_epsilon=cfg.inter_time_add_epsilon,
                )

                stride = cfg.window_stride if cfg.window_stride is not None else cfg.window_size
                if stride <= 0:
                    raise ValueError("window_stride must be > 0 (or None)")

                win_texts: List[str] = []
                for start in range(0, len(diffs), stride):
                    chunk = diffs[start:start + cfg.window_size]
                    if len(chunk) < cfg.window_size and cfg.window_drop_last:
                        break
                    if chunk.size == 0:
                        continue
                    win_texts.append(cfg.inter_time_join_token.join(f"{v:.6g}" for v in chunk))

                # A timing window spans multiple events, so `line_no` is only a
                # coarse anchor to the source file rather than an exact mapping.
                start_no = raw_lines[0][0] if raw_lines else None
                for wtxt in win_texts:
                    examples.append(
                        Example(
                            text=wtxt,
                            label=label,
                            group=group,
                            log_type=log_type,
                            path=str(p),
                            line_no=start_no,
                        )
                    )

            else:
                raise ValueError(f"Unknown window_mode={cfg.window_mode}")

            # ---- Optional cross-file windows (not implemented) ----
            if not cfg.window_within_each_file:
                for (line_no, _raw), txt in zip(raw_lines, base_texts):
                    cross_file_lines.append((maybe_prefix(txt), log_type, str(p), line_no))
                if cids is not None:
                    for (line_no, _raw), cid in zip(raw_lines, cids):
                        cross_file_cids.append((cid, log_type, str(p), line_no))

        if not cfg.window_within_each_file and cfg.window_mode != "none":
            raise NotImplementedError(
                "window_within_each_file=False is not implemented in this version to avoid mixing semantics. "
                "Keep it True (recommended)."
            )

    return examples
