from pathlib import Path
from typing import Any, Optional, List, Sequence, Literal

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
import re

import json
from dataclasses import dataclass
import subprocess
from collections import Counter
from datetime import datetime, timezone

# mathematics
import math
from statistics import NormalDist

# mashine learning
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

# deep learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

class Evaluation:

    # Clustering Degree
    SYSLOG_STRONG_CLUSTERING = False
    NEXTCLOUD_STRONG_CLUSTERING = False

    # ---------------- Common regex ----------------
    _LONG_HEX_RE  = re.compile(r"\b[0-9a-fA-F]{12,}\b")
    _HEX_RE       = re.compile(r"\b0x[0-9a-fA-F]+\b")
    _IP_RE        = re.compile(r"\b(\d{1,3}\.){3}\d{1,3}\b")
    _NUM_RE       = re.compile(r"\b\d+\b")

    # More general path match (not only quoted)
    _UNIX_PATH_RE = re.compile(r"(?<![\w.-])/(?:[\w.-]+/)*[\w.-]+")  # /var/run/... /etc/shadow etc.

    # ---------------- Auditd-specific ----------------
    _AUDIT_PREFIX_RE = re.compile(r"^type=\w+\s+msg=audit\([^)]+\):\s*")
    _AUDIT_MSG_RE    = re.compile(r"msg=audit\([^)]+\):")

    # ---------------- Syslog-specific ----------------
    # Example: 2025-12-17T15:09:01.133248+00:00 finalArenaServer CRON[1576]: ...
    _SYSLOG_RE = re.compile(
        r"^(?P<ts>\d{4}-\d{2}-\d{2}T[^ ]+)\s+(?P<host>\S+)\s+(?P<proc>[^:]+):\s*(?P<msg>.*)$"
    )
    _PROC_PID_RE = re.compile(r"\b([A-Za-z0-9_.-]+)\[(\d+)\]")  # CRON[1576], systemd[1], kernel: no pid

    # ---------------- Timestamp extraction ----------------
    _NEXTCLOUD_TIME_RE = re.compile(r'"time"\s*:\s*"([^"]+)"')
    _AUDIT_EPOCH_RE    = re.compile(r"audit\((\d+\.\d+):") # no longer used
    _AUDIT_EVENT_RE = re.compile(r"audit\((\d+(?:\.\d+)?):(\d+)\)")

    
    # -----------------------------------------------------------------------------
    # -------------------------------- Constructor --------------------------------
    # -----------------------------------------------------------------------------

    def __init__(self) -> None:
        # file paths
        self.file_path_1: str | None = None
        self.file_path_2: str | None = None

        # raw logs
        self.lines_file_1: list[str] | None = None
        self.lines_file_2: list[str] | None = None

        # Drain templates
        self.templated_file_1: list[str] | None = None
        self.templated_file_2: list[str] | None = None

        # cluster IDs
        self.cid_file_1: list[int] | None = None
        self.cid_file_2: list[int] | None = None

        # Detect-Mate
        self.detectmate_dir = Path("/home/lorenz/Documents/DetectMate/DetectMateLibrary")
        self.detectmate_entry = "MA_lorenz.py"
        self.combo_log_path = Path("/home/lorenz/Documents/DetectMate/Logs/IDS_audit.log")

    # --------------------------------------------------------------------------------
    # -------------------------------- Static-Methods --------------------------------
    # --------------------------------------------------------------------------------


    @staticmethod
    def read_file(file_path: str|Path) -> list[str]:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        # splitlines() removes trailing "\n" cleanly and handles last-line-no-newline well
        return path.read_text(encoding="utf-8").splitlines()

    ################ Preprocessing ################
    @staticmethod
    def _detect_type(line: str) -> str:
        s = line.lstrip()
        if s.startswith("type=") and "msg=audit(" in s:
            return "audit"
        if s.startswith("{") and '"reqId"' in s and '"app"' in s:
            return "nextcloud"
        if Evaluation._SYSLOG_RE.match(s):
            return "syslog"
        return "generic"

    @staticmethod
    def _preprocess_audit(line: str) -> str:
        s = line.strip()
        if not s:
            return s

        # normalize audit(...) meta
        s = Evaluation._AUDIT_MSG_RE.sub("msg=audit(<AUDIT_META>):", s)

        # normalize paths, hex, ips
        s = Evaluation._UNIX_PATH_RE.sub("/PATH", s)
        s = Evaluation._LONG_HEX_RE.sub("<HEX>", s)
        s = Evaluation._HEX_RE.sub("<HEX>", s)
        s = Evaluation._IP_RE.sub("<IP>", s)

        # numbers in audit are extremely volatile (pids, inode, ids, etc.)
        s = Evaluation._NUM_RE.sub("<NUM>", s)
        return s

    @staticmethod
    def _preprocess_syslog(line: str) -> str:
        s = line.strip()
        if not s:
            return s

        m = Evaluation._SYSLOG_RE.match(s)
        if not m:
            # fallback
            return Evaluation._preprocess_generic(s)

        proc = m.group("proc")
        msg  = m.group("msg")

        # Normalize process pid part: CRON[1576] -> CRON[<PID>]
        proc = Evaluation._PROC_PID_RE.sub(r"\1[<PID>]", proc)

        # Normalize message content
        msg = Evaluation._UNIX_PATH_RE.sub("/PATH", msg)
        msg = Evaluation._IP_RE.sub("<IP>", msg)
        msg = Evaluation._LONG_HEX_RE.sub("<HEX>", msg)
        msg = Evaluation._HEX_RE.sub("<HEX>", msg)

        # For syslog, do NOT blanket-replace all numbers.
        # Replace only “obviously volatile” standalone numbers (pids, counts, durations).
        # Keep things like service names and fixed tokens.
        msg = re.sub(r"\b\d+\.\d+s\b", "<DUR>", msg)     # 10.301s
        msg = re.sub(r"\b\d+us\b", "<DUR>", msg)        # 10000us
        msg = re.sub(r"\b\d+ms\b", "<DUR>", msg)        # if present

        if Evaluation.SYSLOG_STRONG_CLUSTERING:                                        # leads to strong clustering
            msg = re.sub(r"\b\d+\b", "<NUM>", msg)

        return f"<TS> <HOST> {proc}: {msg}"


    @staticmethod
    def _preprocess_nextcloud(line: str) -> str:
        s = line.strip()
        if not s:
            return s

        try:
            obj = json.loads(s)
        except Exception:
            return Evaluation._preprocess_generic(s)

        # Pull out the “shape” fields (these are usually stable)
        app = obj.get("app", "<APP>")
        level = obj.get("level", "<LEVEL>")
        method = obj.get("method", "<METHOD>")
        url = obj.get("url", "<URL>")
        msg = obj.get("message", "")

        # Normalize URL: replace query params (often contain user names etc.)
        # /index.php/login?user=admin&direct=1... -> /index.php/login?<QS>
        url = re.sub(r"\?.*$", "?<QS>", url)

        # Normalize message content
        msg = Evaluation._IP_RE.sub("<IP>", msg)
        msg = Evaluation._UNIX_PATH_RE.sub("/PATH", msg)
        msg = Evaluation._LONG_HEX_RE.sub("<HEX>", msg)
        msg = Evaluation._HEX_RE.sub("<HEX>", msg)
        msg = re.sub(r"'[^']+'@'[^']+'", "'<USER>'@'<HOST>'", msg)  # SQL user@host
        msg = re.sub(r"\bSQLSTATE\[[^\]]+\]\s*\[\d+\]", "SQLSTATE[<STATE>][<CODE>]", msg)

        # If exception exists, keep only high-level class/code (drop trace)

        exc = obj.get("exception")
        if isinstance(exc, dict):
            exc_name = exc.get("Exception", "<EXC>")
            if Evaluation.NEXTCLOUD_STRONG_CLUSTERING:
                exc_part = f"{exc_name}(<CODE>)"
            else:
                exc_code = exc.get("Code")
                exc_part = f"{exc_name}({exc_code})" if exc_code is not None else f"{exc_name}(<CODE>)"

        else:
            exc_part = "<NOEXC>"

        # Build a normalized string for Drain
        return f"nextcloud app={app} level={level} {method} {url} exc={exc_part} msg={msg}"

    @staticmethod
    def _preprocess_generic(line: str) -> str:
        s = line.strip()
        s = Evaluation._UNIX_PATH_RE.sub("/PATH", s)
        s = Evaluation._IP_RE.sub("<IP>", s)
        s = Evaluation._LONG_HEX_RE.sub("<HEX>", s)
        s = Evaluation._HEX_RE.sub("<HEX>", s)
        # avoid too aggressive num replace in generic; but keep if you want
        return s

    @staticmethod
    def _preprocess(line: str) -> str:
        kind = Evaluation._detect_type(line)
        if kind == "audit":
            return Evaluation._preprocess_audit(line)
        if kind == "nextcloud":
            return Evaluation._preprocess_nextcloud(line)
        if kind == "syslog":
            return Evaluation._preprocess_syslog(line)
        return Evaluation._preprocess_generic(line)


    # -------------------------- sequence window helpers --------------------------

    @staticmethod
    def _make_windows_from_lines(
        lines: list[str],
        *,
        window_size: int,
        stride: int | None = None,
        join_token: str = " <EOL> ",
        drop_last: bool = True,
    ) -> list[str]:
        """
        Turn a list of lines into window-documents by concatenating consecutive lines.

        Example window (size=3):
            "L1 <EOL> L2 <EOL> L3"

        Args:
            window_size: number of consecutive lines per window
            stride: step between window starts. If None -> stride=window_size (non-overlapping)
            join_token: delimiter between lines
            drop_last: if True, drop incomplete last window; if False, keep it

        Returns:
            list of window strings
        """
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if stride is None:
            stride = window_size
        if stride <= 0:
            raise ValueError("stride must be > 0")

        out: list[str] = []
        n = len(lines)
        for start in range(0, n, stride):
            chunk = lines[start:start + window_size]
            if len(chunk) < window_size and drop_last:
                break
            if not chunk:
                continue
            out.append(join_token.join(chunk))
        return out

    @staticmethod
    def _make_windows_from_cids(
        cids: list[int],
        *,
        window_size: int,
        stride: int | None = None,
        prefix: str = "CID",
        drop_last: bool = True,
    ) -> list[str]:
        """
        Turn a list of integer CIDs into CID window-documents.

        Example window (size=3):
            "CID5 CID7 CID9"

        Args:
            window_size: number of consecutive CIDs per window
            stride: step between window starts. If None -> stride=window_size (non-overlapping)
            prefix: text prefix for each cid token (keeps it as text for your models)
            drop_last: if True, drop incomplete last window; if False, keep it

        Returns:
            list of window strings
        """
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if stride is None:
            stride = window_size
        if stride <= 0:
            raise ValueError("stride must be > 0")

        out: list[str] = []
        n = len(cids)
        for start in range(0, n, stride):
            chunk = cids[start:start + window_size]
            if len(chunk) < window_size and drop_last:
                break
            if not chunk:
                continue
            out.append(" ".join(f"{prefix}{cid}" for cid in chunk))
        return out



    ################ Template Generation ################
    @staticmethod
    def logs_to_templates(log_lines: list[str], ini_path: str | None = None) -> tuple[list[str], list[int]]:
        cfg = TemplateMinerConfig()

        if ini_path is None:
            ini = Path(__file__).resolve().parent / "drain3.ini"
        else:
            ini = Path(ini_path)

        if not ini.is_file():
            raise FileNotFoundError(f"Missing Drain3 config: {ini}")

        cfg.load(str(ini))
            
        miner = TemplateMiner(config=cfg, persistence_handler=None)

        cluster_ids: list[int] = []
        for line in log_lines:
            msg = Evaluation._preprocess(line)
            result = miner.add_log_message(msg)
            cluster_ids.append(result["cluster_id"])

        # Final templates after all logs are processed
        final_templates = {c.cluster_id: c.get_template() for c in miner.drain.clusters}

        # Map each log to its final template
        templates = [final_templates[cid] for cid in cluster_ids]

        return templates, cluster_ids
    
    ################ Timestamp Extraction ################
    
    @staticmethod
    def _extract_nextcloud_timestamps(lines: list[str]) -> list[datetime]:
        ts: list[datetime] = []
        for line in lines:
            m = Evaluation._NEXTCLOUD_TIME_RE.search(line)
            if not m:
                continue
            try:
                ts.append(datetime.fromisoformat(m.group(1)))
            except Exception:
                continue
        return ts

    '''
    @staticmethod
    def _extract_auditlog_timestamps(lines: list[str]) -> list[datetime]:
        ts: list[datetime] = []
        for line in lines:
            m = Evaluation._AUDIT_EPOCH_RE.search(line)
            if not m:
                continue
            try:
                ts.append(datetime.fromtimestamp(float(m.group(1))))
            except Exception:
                continue
        return ts
    '''

    @staticmethod
    def _extract_auditlog_timestamps(lines: list[str]) -> list[datetime]:
        """
        Return ONE timestamp per audit event bundle (unique serial).
        Uses the first-seen timestamp for each serial.
        """
        seen: set[int] = set()
        ts: list[datetime] = []
        for line in lines:
            m = Evaluation._AUDIT_EVENT_RE.search(line)
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


    @staticmethod
    def _extract_syslog_timestamps(lines: list[str]) -> list[datetime]:
        ts: list[datetime] = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            try:
                first = s.split(" ", 1)[0]
                ts.append(datetime.fromisoformat(first))
            except Exception:
                continue
        return ts

    @staticmethod
    def _extract_generic_timestamps(lines: list[str]) -> list[datetime]:
        # fallback: try "first token is iso timestamp"
        ts: list[datetime] = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            first = s.split(" ", 1)[0]
            try:
                ts.append(datetime.fromisoformat(first))
            except Exception:
                continue
        return ts

    @staticmethod
    def extract_timestamps(lines: list[str]) -> list[datetime]:
        """
        Prefer a single extraction strategy based on the dominant detected type.
        """
        kinds = [Evaluation._detect_type(ln) for ln in lines[:200] if ln.strip()]
        if not kinds:
            return []

        dominant = Counter(kinds).most_common(1)[0][0]

        if dominant == "audit":
            ts = Evaluation._extract_auditlog_timestamps(lines)
        elif dominant == "nextcloud":
            ts = Evaluation._extract_nextcloud_timestamps(lines)
        elif dominant == "syslog":
            ts = Evaluation._extract_syslog_timestamps(lines)
        else:
            ts = Evaluation._extract_generic_timestamps(lines)

        ts.sort()
        return ts

    ################ inter-event functionality ################

    @staticmethod
    def inter_event_diffs_seconds(timestamps: list[datetime]) -> np.ndarray:
        if len(timestamps) < 2:
            return np.array([], dtype=float)

        diffs: list[float] = []
        prev = timestamps[0]
        for cur in timestamps[1:]:
            dt = (cur - prev).total_seconds()
            if dt >= 0:
                diffs.append(dt)
            prev = cur

        return np.asarray(diffs, dtype=float)


    @staticmethod
    def make_inter_event_bin_edges(
        diffs1: np.ndarray,
        diffs2: np.ndarray,
        *,
        use_log_bins: bool,
        n_bins: int,
    ) -> np.ndarray | None:
        all_diffs = np.concatenate([diffs1, diffs2])
        all_diffs = all_diffs[np.isfinite(all_diffs)]
        all_diffs = all_diffs[all_diffs >= 0]

        if all_diffs.size == 0:
            return None

        dmin = float(np.min(all_diffs))
        dmax = float(np.max(all_diffs))
        if dmax == dmin:
            return None

        if use_log_bins:
            lo = max(dmin, 1e-6)
            hi = max(dmax, lo * 1.000001)
            edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
            if dmin <= 0.0:
                edges = np.concatenate(([0.0], edges))
            return edges
        else:
            return np.linspace(dmin, dmax, n_bins + 1)


    ################ index adjustment ################


    @staticmethod
    def _idx_to_ranges(idxs: Sequence[int]) -> list[tuple[int, int]]:
        """[3,4,5, 10,11] -> [(3,6),(10,12)] (half-open)."""
        if not idxs:
            return []
        idxs = sorted(set(int(i) for i in idxs))
        out: list[tuple[int, int]] = []
        start = prev = idxs[0]
        for i in idxs[1:]:
            if i == prev + 1:
                prev = i
            else:
                out.append((start, prev + 1))
                start = prev = i
        out.append((start, prev + 1))
        return out

    @staticmethod
    def _line_ranges_to_window_mask(
        *,
        n_lines: int,
        test_ranges: Sequence[tuple[int, int]],
        window_size: int,
        stride: int,
        drop_last: bool,
    ) -> np.ndarray:
        starts = list(range(0, n_lines, stride))
        if drop_last:
            starts = [s for s in starts if s + window_size <= n_lines]
        else:
            starts = [s for s in starts if s < n_lines]

        mask = np.zeros(len(starts), dtype=bool)
        for k, s in enumerate(starts):
            e = min(s + window_size, n_lines)
            for (t0, t1) in test_ranges:
                if (s < t1) and (e > t0):  # overlap
                    mask[k] = True
                    break
        return mask

    @staticmethod
    def adjust_split_indices_for_windows(
        *,
        n_human_lines: int,
        n_ai_lines: int,
        train_idx: Sequence[int],
        test_idx: Sequence[int],
        window_size: int,
        stride: int | None = None,
        drop_last: bool = True,
    ) -> tuple[list[int], list[int]]:
        """
        Convert LINE-level indices (into human_lines + ai_lines)
        into WINDOW-level indices (into human_windows + ai_windows),
        marking any window that overlaps ANY test line as test.

        Returns: (train_window_idx, test_window_idx)
        """
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if stride is None:
            stride = window_size
        if stride <= 0:
            raise ValueError("stride must be > 0")

        n_total_lines = n_human_lines + n_ai_lines

        train_idx = np.asarray(list(train_idx), dtype=int)
        test_idx  = np.asarray(list(test_idx), dtype=int)

        if train_idx.size == 0 or test_idx.size == 0:
            raise ValueError("train_idx/test_idx must be non-empty.")

        if np.any(train_idx < 0) or np.any(train_idx >= n_total_lines) or np.any(test_idx < 0) or np.any(test_idx >= n_total_lines):
            raise ValueError(f"indices out of range for n_total_lines={n_total_lines}.")

        # split test indices into per-side LOCAL indices
        test_h = [int(i) for i in test_idx if i < n_human_lines]
        test_a = [int(i - n_human_lines) for i in test_idx if i >= n_human_lines]

        test_h_ranges = Evaluation._idx_to_ranges(test_h)
        test_a_ranges = Evaluation._idx_to_ranges(test_a)

        mask_h = Evaluation._line_ranges_to_window_mask(
            n_lines=n_human_lines,
            test_ranges=test_h_ranges,
            window_size=window_size,
            stride=stride,
            drop_last=drop_last,
        )
        mask_a = Evaluation._line_ranges_to_window_mask(
            n_lines=n_ai_lines,
            test_ranges=test_a_ranges,
            window_size=window_size,
            stride=stride,
            drop_last=drop_last,
        )

        n_h_windows = int(mask_h.size)

        test_w = np.where(mask_h)[0].tolist() + (np.where(mask_a)[0] + n_h_windows).tolist()
        train_w = np.where(~mask_h)[0].tolist() + (np.where(~mask_a)[0] + n_h_windows).tolist()

        return train_w, test_w


    ################ Mathematics ################

    @staticmethod
    def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        eps = 1e-12
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
    
    @staticmethod
    def hist_prob(diffs: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
        h, _ = np.histogram(diffs, bins=bin_edges)
        h = h.astype(float)
        s = h.sum()
        if s <= 0:
            return np.ones(len(h), dtype=float) / len(h)
        return h / s
    
    @staticmethod
    def wilson_ci(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
        """
        Wilson score CI for a binomial proportion.
        Returns (low, high).
        """
        if n <= 0:
            raise ValueError("n must be > 0")
        if x < 0 or x > n:
            raise ValueError("x must be in [0, n]")

        z = NormalDist().inv_cdf(1 - alpha / 2)
        phat = x / n

        den = 1.0 + (z * z) / n
        center = (phat + (z * z) / (2 * n)) / den
        half = (z / den) * math.sqrt((phat * (1 - phat)) / n + (z * z) / (4 * n * n))

        return max(0.0, center - half), min(1.0, center + half)

    @staticmethod
    def newcombe_diff_ci(x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05) -> tuple[float, float]:
        """
        Newcombe CI for difference of proportions: (p1 - p2),
        combining Wilson CIs.
        """
        L1, U1 = Evaluation.wilson_ci(x1, n1, alpha=alpha)
        L2, U2 = Evaluation.wilson_ci(x2, n2, alpha=alpha)
        return (L1 - U2, U1 - L2)
    
    @staticmethod
    def _gini_from_counts(counter: Counter) -> float:
        if not counter:
            return float("nan")
        x = np.array(sorted(counter.values()), dtype=float)
        n = x.size
        s = x.sum()
        if s <= 0 or n == 0:
            return float("nan")
        return float(2.0 * (np.arange(1, n + 1) * x).sum() / (n * s) - (n + 1) / n)

    @staticmethod
    def _kurtosis_from_counts(counter: Counter, convexify: bool = True) -> float:
        freq = list(counter.values())
        if len(freq) == 0:
            return float("nan")

        if convexify:
            asc = sorted(freq)
            desc = sorted(freq, reverse=True)
            x = np.asarray(asc + desc, dtype=float)
        else:
            x = np.asarray(freq, dtype=float)

        n = x.size
        if n < 4:
            return float("nan")

        mean = x.mean()
        d = x - mean
        s2 = (d @ d) / (n - 1)
        if s2 == 0:
            return float("nan")
        s4 = s2 ** 2
        m4 = np.sum(d ** 4)

        g2 = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * (m4 / s4) \
            - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
        return float(g2)

    @staticmethod
    def _entropy_from_counts(counter: Counter, base: float = 2.0) -> float:
        if not counter:
            return float("nan")
        x = np.array(list(counter.values()), dtype=float)
        total = x.sum()
        if total <= 0:
            return float("nan")
        p = x / total
        p = p[p > 0]
        if p.size == 0:
            return float("nan")

        if base == 2.0:
            return float(-np.sum(p * np.log2(p)))
        elif base == np.e:
            return float(-np.sum(p * np.log(p)))
        else:
            return float(-np.sum(p * (np.log(p) / np.log(base))))

    @staticmethod
    def _mad_from_counts(counter: Counter) -> float:
        if not counter:
            return float("nan")
        x = np.array(list(counter.values()), dtype=float)
        if x.size == 0:
            return float("nan")
        mean = x.mean()
        return float(np.mean(np.abs(x - mean)))

    @staticmethod
    def _sliding_windows(seq: list[int], size: int, stride: int):
        for i in range(0, len(seq) - size + 1, stride):
            yield tuple(seq[i:i + size])

    @staticmethod
    def _stats_from_ids(ids: list[int]) -> dict[str, float]:
        cnt = Counter(ids)
        return {
            "gini": Evaluation._gini_from_counts(cnt),
            "kurtosis": Evaluation._kurtosis_from_counts(cnt, convexify=True),
            "entropy": Evaluation._entropy_from_counts(cnt, base=2.0),
            "mad": Evaluation._mad_from_counts(cnt),
        }

    @staticmethod
    def _stats_from_windows(ids: list[int], window_size: int, stride: int) -> dict[str, float]:
        wins = list(Evaluation._sliding_windows(ids, window_size, stride))
        cnt = Counter(wins)
        return {
            "gini_seq": Evaluation._gini_from_counts(cnt),
            "kurtosis_seq": Evaluation._kurtosis_from_counts(cnt, convexify=True),
            "entropy_seq": Evaluation._entropy_from_counts(cnt, base=2.0),
            "mad_seq": Evaluation._mad_from_counts(cnt),
        }

    @staticmethod
    def extract_linear_features(clf, vectorizer: TfidfVectorizer, top_k: int = 30) -> list[tuple[str, float]]:
        names = np.array(vectorizer.get_feature_names_out())

        if hasattr(clf, "coef_"):
            w = clf.coef_[0]
        elif hasattr(clf, "feature_log_prob_"):
            # MultinomialNB: use log-likelihood ratio for class 1 vs class 0
            w = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        else:
            raise TypeError("Model is not interpretable (no coef_ or feature_log_prob_)")

        idx = np.argsort(np.abs(w))[::-1][:top_k]
        return [(str(names[i]), float(w[i])) for i in idx]


    # --------------------------------------------------------------------------------
    # -------------------------------- Instance-Methods --------------------------------
    # --------------------------------------------------------------------------------

    def set_files(self, file_path_1: str | Path, file_path_2: str | Path) -> None:
        self.file_path_1 = file_path_1
        self.file_path_2 = file_path_2
        self.lines_file_1 = self.read_file(self.file_path_1)
        self.lines_file_2 = self.read_file(self.file_path_2)
        # Invalidate cached derived data
        self.templated_file_1 = None
        self.templated_file_2 = None
        self.cid_file_1 = None
        self.cid_file_2 = None

    def build_templates(self) -> None:
        if (
            self.templated_file_1 is not None and self.templated_file_2 is not None
            and self.cid_file_1 is not None and self.cid_file_2 is not None
        ):
            return
        if self.lines_file_1 is None or self.lines_file_2 is None:
            raise RuntimeError("lines_file_1/2 not set")

        all_logs = self.lines_file_1 + self.lines_file_2
        templates, cluster_ids = Evaluation.logs_to_templates(all_logs)
        n1 = len(self.lines_file_1)

        self.templated_file_1 = templates[:n1]
        self.templated_file_2 = templates[n1:]
        self.cid_file_1 = cluster_ids[:n1]
        self.cid_file_2 = cluster_ids[n1:]

    def peek_logs(self, n: int = 5) -> None:
        """
        Print the first `n` log lines from both files.
        """
        print("=== File 1 (first", n, "lines) ===")
        for line in self.lines_file_1[:n]:
            print(line)

        print("\n=== File 2 (first", n, "lines) ===")
        for line in self.lines_file_2[:n]:
            print(line)

    def build_line_windows(
        self,
        *,
        window_size: int,
        stride: int | None = None,
        max_lines: int = 5000,
        preprocessing_flag: bool = True,
        template_flag: bool = False,
        join_token: str = " <EOL> ",
        drop_last: bool = True,
    ) -> tuple[list[str], list[str]]:
        """
        Builds window-documents from file_1 and file_2 by concatenating consecutive lines.

        Returns:
            (windows_file_1, windows_file_2)
        """
        if self.lines_file_1 is None or self.lines_file_2 is None:
            raise RuntimeError("lines_file_1 / lines_file_2 not set. Call set_files() first.")

        # Choose base lines
        if template_flag:
            self.build_templates()
            base1 = self.templated_file_1[:max_lines]
            base2 = self.templated_file_2[:max_lines]
        else:
            base1 = self.lines_file_1[:max_lines]
            base2 = self.lines_file_2[:max_lines]
            if preprocessing_flag:
                base1 = [self._preprocess(x) for x in base1]
                base2 = [self._preprocess(x) for x in base2]

        w1 = self._make_windows_from_lines(
            base1,
            window_size=window_size,
            stride=stride,
            join_token=join_token,
            drop_last=drop_last,
        )
        w2 = self._make_windows_from_lines(
            base2,
            window_size=window_size,
            stride=stride,
            join_token=join_token,
            drop_last=drop_last,
        )
        return w1, w2


    def build_cid_windows(
        self,
        *,
        window_size: int,
        stride: int | None = None,
        max_lines: int = 5000,
        prefix: str = "CID",
        drop_last: bool = True,
    ) -> tuple[list[str], list[str]]:
        """
        Builds CID window-documents from file_1 and file_2.

        Returns:
            (cid_windows_file_1, cid_windows_file_2)
        """
        self.build_templates()

        c1 = self.cid_file_1[:max_lines]
        c2 = self.cid_file_2[:max_lines]

        w1 = self._make_windows_from_cids(
            c1,
            window_size=window_size,
            stride=stride,
            prefix=prefix,
            drop_last=drop_last,
        )
        w2 = self._make_windows_from_cids(
            c2,
            window_size=window_size,
            stride=stride,
            prefix=prefix,
            drop_last=drop_last,
        )
        return w1, w2

    def _effective_line_counts_for_windowing(
        self,
        *,
        max_lines: int,
        preprocessing_flag: bool,
        template_flag: bool,
        window_mode: Literal["raw", "cid"],
    ) -> tuple[int, int]:
        """
        Return the exact line counts that will be used to build windows
        for the given (window_mode, max_lines, preprocessing_flag, template_flag).
        """
        if self.lines_file_1 is None or self.lines_file_2 is None:
            raise RuntimeError("lines_file_1 / lines_file_2 not set. Call set_files() first.")

        if window_mode == "cid":
            self.build_templates()
            n1 = len(self.cid_file_1[:max_lines])
            n2 = len(self.cid_file_2[:max_lines])
            return n1, n2

        # window_mode == "raw"
        if template_flag:
            self.build_templates()
            n1 = len(self.templated_file_1[:max_lines])
            n2 = len(self.templated_file_2[:max_lines])
            return n1, n2

        # raw lines (optionally preprocessed, but preprocessing does NOT change count in your code)
        n1 = len(self.lines_file_1[:max_lines])
        n2 = len(self.lines_file_2[:max_lines])
        return n1, n2


    #-------------------------- helpers --------------------------


    def _complexity_metrics_from_lines(
        self,
        lines1: list[str],
        lines2: list[str],
        window_size: int,
        stride: int,
    ) -> tuple[dict[str, float], dict[str, float], list[int], list[int]]:
        """
        Computes template-based + sequence-window-based metrics for both samples.
        Returns:
        (metrics1, metrics2, cids1, cids2)

        metrics keys:
        gini, kurtosis, entropy, mad, gini_seq, kurtosis_seq, entropy_seq, mad_seq
        """
        # shared clustering space
        _, cluster_ids = Evaluation.logs_to_templates(lines1 + lines2)
        n1 = len(lines1)
        cids1 = cluster_ids[:n1]
        cids2 = cluster_ids[n1:]

        m1 = Evaluation._stats_from_ids(cids1)
        m2 = Evaluation._stats_from_ids(cids2)

        if len(cids1) >= window_size:
            m1.update(Evaluation._stats_from_windows(cids1, window_size, stride))
        else:
            m1.update({"gini_seq": float("nan"), "kurtosis_seq": float("nan"), "entropy_seq": float("nan"), "mad_seq": float("nan")})

        if len(cids2) >= window_size:
            m2.update(Evaluation._stats_from_windows(cids2, window_size, stride))
        else:
            m2.update({"gini_seq": float("nan"), "kurtosis_seq": float("nan"), "entropy_seq": float("nan"), "mad_seq": float("nan")})

        return m1, m2, cids1, cids2


    def _run_combo_detector(self, train_logs: list[str], test_logs: list[str]) -> dict[str, Any]:
        """
        Internal helper:
        - writes TRAIN then TEST to COMBO_LOG_PATH
        - runs DetectMate
        - parses JSON stdout
        Returns parsed result dict.
        """

        train_size = len(train_logs)
        test_size = len(test_logs)

        if train_size == 0 or test_size == 0:
            raise ValueError(f"Need non-empty train/test. Got train={train_size} test={test_size}")

        # 1) Write combined file: TRAIN then TEST
        self.combo_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.combo_log_path.open("w", encoding="utf-8") as f:
            for line in train_logs:
                f.write(line.rstrip("\n") + "\n")
            for line in test_logs:
                f.write(line.rstrip("\n") + "\n")

        # 2) Run DetectMate
        if not self.detectmate_dir.is_dir():
            raise FileNotFoundError(f"DetectMate dir not found: {self.detectmate_dir}")

        cmd = [
            "uv", "run", "python", self.detectmate_entry,
            "--train-size", str(train_size),
            "--test-size", str(test_size),
        ]

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.detectmate_dir),
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "Combo detector subprocess failed.\n"
                f"cmd: {e.cmd}\n"
                f"returncode: {e.returncode}\n"
                f"stdout:\n{e.stdout}\n"
                f"stderr:\n{e.stderr}\n"
            ) from e

        stdout = (proc.stdout or "").strip()
        if not stdout:
            raise RuntimeError(
                "Combo detector returned empty stdout.\n"
                f"stderr was:\n{proc.stderr}"
            )

        try:
            result = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                "Failed to parse combo detector output as JSON.\n"
                f"stdout was:\n{stdout}\n"
                f"stderr was:\n{proc.stderr}"
            ) from e

        if not isinstance(result, dict):
            raise RuntimeError(f"Combo detector returned non-dict JSON: {type(result)}")

        return result


    #------------------------------- Metric Methods -----------------------------------



    def inter_event_result(
        self,
        *,
        max_lines: int = 5000,
        min_events: int = 20,
    ) -> dict[str, Any]:
        """
        Returns inter-event time differences for both files so plotting can use them.

        Output:
        {
          "diffs_1": np.ndarray,
          "diffs_2": np.ndarray,
          "n_timestamps_1": int,
          "n_timestamps_2": int,
          "dominant_type_1": str,
          "dominant_type_2": str,
        }
        """
        if self.lines_file_1 is None or self.lines_file_2 is None:
            raise RuntimeError("lines_file_1 / lines_file_2 not set. Call set_files() first.")

        lines1 = self.lines_file_1[:max_lines]
        lines2 = self.lines_file_2[:max_lines]

        ts1 = Evaluation.extract_timestamps(lines1)
        ts2 = Evaluation.extract_timestamps(lines2)

        diffs1 = Evaluation.inter_event_diffs_seconds(ts1)
        diffs2 = Evaluation.inter_event_diffs_seconds(ts2)

        # Helpful context: dominant type (same logic as extract_timestamps, but cheap to re-check)
        kinds1 = [Evaluation._detect_type(ln) for ln in lines1[:200] if ln.strip()]
        kinds2 = [Evaluation._detect_type(ln) for ln in lines2[:200] if ln.strip()]
        dom1 = Counter(kinds1).most_common(1)[0][0] if kinds1 else "unknown"
        dom2 = Counter(kinds2).most_common(1)[0][0] if kinds2 else "unknown"

        # If you want the same gating rule as event_time_evaluate:
        if len(ts1) < min_events or len(ts2) < min_events:
            # return empty arrays so caller can handle gracefully
            return {
                "diffs_1": np.array([], dtype=float),
                "diffs_2": np.array([], dtype=float),
                "n_timestamps_1": len(ts1),
                "n_timestamps_2": len(ts2),
                "dominant_type_1": dom1,
                "dominant_type_2": dom2,
            }

        return {
            "diffs_1": diffs1,
            "diffs_2": diffs2,
            "n_timestamps_1": len(ts1),
            "n_timestamps_2": len(ts2),
            "dominant_type_1": dom1,
            "dominant_type_2": dom2,
        }


    def n_gram_report(
        self,
        max_lines: int = 5000,
        test_size: float = 0.30,
        random_state: int = 42,
        preprocessing_flag: bool = True,
        template_flag: bool = False,
        use_char_tfidf: bool = False,
        top_k_features: int = 30,
        train_idx: Optional[Sequence[int]] = None,
        test_idx: Optional[Sequence[int]] = None,
        window_mode: Literal["none", "raw", "cid"] = "none",
        window_size: int = 5,
    ) -> dict[str, Any]:
        """
        Train linear models on TF-IDF n-grams and report:
        - balanced accuracy on a held-out test split
        - top-K most influential features per model (signed weights)
        Returns:
        {
        "settings": {...},
        "data": {"n_human": ..., "n_ai": ...},
        "models": {
            "Linear SVM": {"balanced_accuracy": ..., "top_features": [...]},
            "Logistic Regression": {...},
            "Naive Bayes": {...}
        }
        }
        """

        # --------- choose dataset ----------
        if window_mode == "raw":
            human, ai = self.build_line_windows(
                window_size=window_size,
                max_lines=max_lines,
                preprocessing_flag=preprocessing_flag,
                template_flag=template_flag,
                stride=None,           # non-overlapping
                drop_last=True,
            )
        elif window_mode == "cid":
            if self.cid_file_1 is None or self.cid_file_2 is None:
                self.build_templates()
            human, ai = self.build_cid_windows(
                window_size=window_size,
                max_lines=max_lines,
                stride=None,           # non-overlapping
                drop_last=True,
            )
        else:
            if template_flag:
                self.build_templates()
                human_raw = self.templated_file_1[:max_lines]
                ai_raw = self.templated_file_2[:max_lines]
                human = list(human_raw)
                ai = list(ai_raw)
            else:
                if self.lines_file_1 is None or self.lines_file_2 is None:
                    raise RuntimeError("lines_file_1 / lines_file_2 not set. Call set_files() first.")
                human_raw = self.lines_file_1[:max_lines]
                ai_raw = self.lines_file_2[:max_lines]
                if preprocessing_flag:
                    human = [self._preprocess(x) for x in human_raw]
                    ai = [self._preprocess(x) for x in ai_raw]
                else:
                    human = list(human_raw)
                    ai = list(ai_raw)

        if len(human) == 0 or len(ai) == 0:
            raise ValueError(f"Need non-empty human/ai logs. Got human={len(human)} ai={len(ai)}")

        X_text = human + ai
        y = np.array([0] * len(human) + [1] * len(ai), dtype=int)

        # --------- vectorizer ----------
        if use_char_tfidf:
            vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 6),
                min_df=2,
                sublinear_tf=True,
            )
        else:
            max_n = 3
            if window_mode != "none":
                max_n = min(window_size, 10)
            vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, max_n),
                min_df=2,
                sublinear_tf=True,
            )

        X = vectorizer.fit_transform(X_text)

        # --------- split ----------
        if (train_idx is None) and (test_idx is None):
            # default: random split (existing behavior)
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        else:
            # manual split: caller specifies indices
            n = X.shape[0]
            if train_idx is None or test_idx is None:
                raise ValueError("Provide BOTH train_idx and test_idx, or neither.")

            # --- NEW: if windowing, interpret indices as LINE indices and convert to WINDOW indices ---
            if window_mode in ("raw", "cid"):
                n1_lines, n2_lines = self._effective_line_counts_for_windowing(
                    max_lines=max_lines,
                    preprocessing_flag=preprocessing_flag,
                    template_flag=template_flag,
                    window_mode=window_mode,
                )

                train_idx, test_idx = Evaluation.adjust_split_indices_for_windows(
                    n_human_lines=n1_lines,
                    n_ai_lines=n2_lines,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    window_size=window_size,
                    stride=None,      # must match your builders (None => window_size)
                    drop_last=True,   # must match your builders
                )

            # now indices must be WINDOW indices into X_text
            train_idx = np.asarray(list(train_idx), dtype=int)
            test_idx  = np.asarray(list(test_idx), dtype=int)

            # basic validation
            if train_idx.size == 0 or test_idx.size == 0:
                raise ValueError("train_idx/test_idx must be non-empty.")
            if np.any(train_idx < 0) or np.any(train_idx >= n) or np.any(test_idx < 0) or np.any(test_idx >= n):
                raise ValueError(f"indices out of range (n_samples={n}).")
            if np.intersect1d(train_idx, test_idx).size != 0:
                raise ValueError("train_idx and test_idx must be disjoint.")

            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test  = X[test_idx]
            y_test  = y[test_idx]

        # --------- models (only the ones you asked for) ----------
        models = {
            "Linear SVM": LinearSVC(),
            "Logistic Regression": LogisticRegression(max_iter=500),
            #"Naive Bayes": MultinomialNB(),
        }

        report: dict[str, Any] = {
            "settings": {
                "max_lines": max_lines,
                "test_size": test_size,
                "random_state": random_state,
                "preprocessing_flag": preprocessing_flag,
                "template_flag": template_flag,
                "use_char_tfidf": use_char_tfidf,
                "top_k_features": top_k_features,
                "ngram_range": (3, 6) if use_char_tfidf else (1, max_n),
                "analyzer": "char" if use_char_tfidf else "word",
                "min_df": 2,
                "sublinear_tf": True,
            },
            "data": {"n_human": int(len(human)), "n_ai": int(len(ai)), "n_features": int(X.shape[1])},
            "models": {},
        }

        for name, clf in models.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # --- metrics ---
            bac = float(balanced_accuracy_score(y_test, y_pred))
            acc = float(accuracy_score(y_test, y_pred))

            # treat label=1 ("ai") as positive class
            prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
            rec  = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
            f1   = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

            top_features = self.extract_linear_features(clf, vectorizer, top_k=top_k_features)

            report["models"][name] = {
                "balanced_accuracy": bac,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "confusion_matrix": {
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                },
                "top_features": top_features,
            }


        return report


    def deep_learning_report(
        self,
        *,
        max_lines: int = 5000,
        test_size: float = 0.30,
        random_state: int = 42,
        preprocessing_flag: bool = True,
        template_flag: bool = False,
        train_idx: Optional[Sequence[int]] = None,
        test_idx: Optional[Sequence[int]] = None,
        window_mode: Literal["none", "raw", "cid"] = "none",
        window_size: int = 5,
        # --- model/training hyperparams ---
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        embed_dim: int = 64,
        num_filters: int = 64,
        max_len_cap: int = 512,
        len_percentile: float = 95.0,
        grad_clip: float = 5.0,
        use_pos_weight: bool = True,
    ) -> dict[str, Any]:
        """
        Train the MultiKernelCharCNN on character sequences and report metrics
        on a held-out test split (or on explicit indices).

        Labels: 0=human, 1=ai
        """

        # -------------------- 1) choose dataset --------------------
        
        # --------- choose dataset ----------
        if window_mode == "raw":
            human, ai = self.build_line_windows(
                window_size=window_size,
                max_lines=max_lines,
                preprocessing_flag=preprocessing_flag,
                template_flag=template_flag,
                stride=None,           # non-overlapping
                drop_last=True,
            )
        elif window_mode == "cid":
            if self.cid_file_1 is None or self.cid_file_2 is None:
                self.build_templates()
            human, ai = self.build_cid_windows(
                window_size=window_size,
                max_lines=max_lines,
                stride=None,           # non-overlapping
                drop_last=True,
            )
        else:        
            if template_flag:
                self.build_templates()
                human_raw = self.templated_file_1[:max_lines]
                ai_raw = self.templated_file_2[:max_lines]
                human = list(human_raw)
                ai = list(ai_raw)
            else:
                if self.lines_file_1 is None or self.lines_file_2 is None:
                    raise RuntimeError("lines_file_1 / lines_file_2 not set. Call set_files() first.")
                human_raw = self.lines_file_1[:max_lines]
                ai_raw = self.lines_file_2[:max_lines]
                if preprocessing_flag:
                    human = [self._preprocess(x) for x in human_raw]
                    ai = [self._preprocess(x) for x in ai_raw]
                else:
                    human = list(human_raw)
                    ai = list(ai_raw)



        if len(human) == 0 or len(ai) == 0:
            raise ValueError(f"Need non-empty human/ai logs. Got human={len(human)} ai={len(ai)}")

        X_text = human + ai
        y = np.array([0] * len(human) + [1] * len(ai), dtype=np.int64)

        # -------------------- 2) split --------------------
        n = len(X_text)
        if (train_idx is None) and (test_idx is None):
            idx = np.arange(n)
            train_idx, test_idx = train_test_split(
                idx,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        else:
            if train_idx is None or test_idx is None:
                raise ValueError("Provide BOTH train_idx and test_idx, or neither.")

            # If we're windowing, user might be providing LINE-level indices from the original logs.
            # Convert them to WINDOW-level indices automatically.
            if window_mode in ("raw", "cid"):
                n1_lines, n2_lines = self._effective_line_counts_for_windowing(
                    max_lines=max_lines,
                    preprocessing_flag=preprocessing_flag,
                    template_flag=template_flag,
                    window_mode=window_mode,
                )

                train_idx, test_idx = Evaluation.adjust_split_indices_for_windows(
                    n_human_lines=n1_lines,
                    n_ai_lines=n2_lines,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    window_size=window_size,
                    stride=None,  # IMPORTANT if you add stride later
                    drop_last=True,  # must match builder
                )
                train_idx = np.asarray(list(train_idx), dtype=int)
                test_idx  = np.asarray(list(test_idx), dtype=int)
            if train_idx.size == 0 or test_idx.size == 0:
                raise ValueError("train_idx/test_idx must be non-empty.")
            if np.any(train_idx < 0) or np.any(train_idx >= n) or np.any(test_idx < 0) or np.any(test_idx >= n):
                raise ValueError(f"indices out of range (n_samples={n}).")
            if np.intersect1d(train_idx, test_idx).size != 0:
                raise ValueError("train_idx and test_idx must be disjoint.")

        X_train_text = [X_text[i] for i in train_idx]
        y_train = y[train_idx]
        X_test_text = [X_text[i] for i in test_idx]
        y_test = y[test_idx]

        # -------------------- choose max_len from TRAIN only --------------------
        lengths = np.array([len(t) for t in X_train_text], dtype=float)
        max_len = int(min(np.percentile(lengths, len_percentile), max_len_cap))
        max_len = max(max_len, 8)

        # -------------------- 3) build char vocab on TRAIN only --------------------
        PAD_ID = 0
        UNK_ID = 1  # reserve 1 for unknown chars
        EOL_ID = 2

        def build_char_vocab(texts: list[str]) -> dict[str, int]:
            chars = sorted({ch for t in texts for ch in t})
            # start at 3 because 0=PAD, 1=UNK, 2=EOL
            return {ch: i + 3 for i, ch in enumerate(chars)}

        char2idx = build_char_vocab(X_train_text)
        vocab_size = max(char2idx.values(), default=1) + 1  # +1 because ids are explicit

        def encode(text: str, mapping: dict[str, int]) -> list[int]:
            return [mapping.get(ch, UNK_ID) for ch in text]
        
        def trunc_head_tail(seq: list[int], L: int) -> list[int]:
            if len(seq) <= L:
                return seq
            half = L // 2
            return seq[:half] + seq[-(L - half):]
        
        def pad_to_len(seq: list[int], L: int) -> list[int]:
            if len(seq) >= L:
                return seq[:L]
            return seq + [PAD_ID] * (L - len(seq))
        
        def pad_segmented_text(
            text: str,
            L: int,
            join_token: str,
            mapping: dict[str, int],
            *,
            per_segment_head_tail: bool = True,
        ) -> list[int]:
            parts = text.split(join_token)

            # fallback if no split
            if len(parts) <= 1:
                seq = encode(text, mapping)
                seq = trunc_head_tail(seq, L)
                return pad_to_len(seq, L)

            n = len(parts)
            sep = n - 1

            # if there are too many segments, we can’t even fit separators + >=1 char each
            if sep >= L:
                # keep first L items as alternating [EOL,...] is meaningless, so just head-tail
                seq = encode(text, mapping)
                seq = trunc_head_tail(seq, L)
                return pad_to_len(seq, L)

            available = L - sep  # chars we can spend on actual content
            base = available // n
            rem = available % n

            out: list[int] = []
            for i, p in enumerate(parts):
                alloc = base + (1 if i < rem else 0)
                if alloc > 0:
                    seg = encode(p, mapping)
                    if len(seg) > alloc:
                        seg = trunc_head_tail(seg, alloc) if per_segment_head_tail else seg[:alloc]
                    else:
                        # optionally: you could pad inside segment, but you already pad at end
                        pass
                    out.extend(seg)

                if i != n - 1:
                    out.append(EOL_ID)

            # enforce exact length
            if len(out) > L:
                out = out[:L]
            elif len(out) < L:
                out += [PAD_ID] * (L - len(out))
            return out


        join_token = " <EOL> "  # must match build_line_windows default

        has_eol = (window_mode == "raw") and any((join_token in t) for t in X_train_text[:50])
        if has_eol:
            X_train = np.array(
                [pad_segmented_text(t, max_len, join_token, char2idx) for t in X_train_text],
                dtype=np.int64
            )
            X_test = np.array(
                [pad_segmented_text(t, max_len, join_token, char2idx) for t in X_test_text],
                dtype=np.int64
            )
        else:
            train_encoded = [encode(t, char2idx) for t in X_train_text]
            test_encoded = [encode(t, char2idx) for t in X_test_text]
            X_train = np.array([pad_to_len(trunc_head_tail(s, max_len), max_len) for s in train_encoded], dtype=np.int64)
            X_test = np.array([pad_to_len(trunc_head_tail(s, max_len), max_len) for s in test_encoded], dtype=np.int64)


        # -------------------- 4) Dataset / DataLoader --------------------
        class _LogDataset(Dataset):
            def __init__(self, X_arr: np.ndarray, y_arr: np.ndarray):
                self.X = torch.tensor(X_arr, dtype=torch.long)
                self.y = torch.tensor(y_arr, dtype=torch.float32)

            def __len__(self) -> int:
                return self.X.shape[0]

            def __getitem__(self, i: int):
                return self.X[i], self.y[i]


        train_loader = DataLoader(_LogDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(_LogDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

        # -------------------- 5) Model definition --------------------
        class MultiKernelCharCNN(nn.Module):
            def __init__(self, vocab_size: int, embed_dim: int, num_filters: int):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)

                self.conv3 = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
                self.conv5 = nn.Conv1d(embed_dim, num_filters, kernel_size=5, padding=2)
                self.conv7 = nn.Conv1d(embed_dim, num_filters, kernel_size=7, padding=3)

                self.bn3 = nn.BatchNorm1d(num_filters)
                self.bn5 = nn.BatchNorm1d(num_filters)
                self.bn7 = nn.BatchNorm1d(num_filters)

                self.pool = nn.AdaptiveMaxPool1d(1)

                self.fc1 = nn.Linear(num_filters * 3, 128)
                self.fc2 = nn.Linear(128, 1)
                self.dropout = nn.Dropout(0.5)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.embedding(x)      # (B, L, E)
                x = x.permute(0, 2, 1)     # (B, E, L)

                x3 = torch.relu(self.bn3(self.conv3(x)))
                x5 = torch.relu(self.bn5(self.conv5(x)))
                x7 = torch.relu(self.bn7(self.conv7(x)))

                x3 = self.pool(x3).squeeze(-1)  # (B, F)
                x5 = self.pool(x5).squeeze(-1)
                x7 = self.pool(x7).squeeze(-1)

                x = torch.cat([x3, x5, x7], dim=1)  # (B, 3F)
                x = self.dropout(torch.relu(self.fc1(x)))
                x = self.fc2(x)  # (B, 1)
                return x

        # -------------------- 6) Training setup --------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiKernelCharCNN(vocab_size, embed_dim, num_filters).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if use_pos_weight:
            n_pos = int((y_train == 1).sum())
            n_neg = int((y_train == 0).sum())
            pos_weight_value = n_neg / max(n_pos, 1)
            pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            n_pos = int((y_train == 1).sum())
            n_neg = int((y_train == 0).sum())
            pos_weight_value = 1.0
            criterion = nn.BCEWithLogitsLoss()

        # -------------------- 7) Train --------------------
        print("Start Train Section")
        for ep in range(epochs):
            model.train()
            total_loss = 0.0

            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)

                optimizer.zero_grad()
                logits = model(Xb).squeeze(-1)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                total_loss += float(loss.item())
                print("Another batch is complete!")

            avg_loss = total_loss / max(len(train_loader), 1)
            # You can comment this out if you don’t want per-epoch printing
            print(f"Epoch {ep+1}/{epochs} - loss={avg_loss:.4f}")

        # -------------------- 8) Evaluate --------------------
        print("Starting Evaluation Section")
        model.eval()
        all_probs: list[float] = []
        all_preds: list[int] = []
        all_true: list[int] = []

        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb = Xb.to(device)
                logits = model(Xb).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                all_probs.extend(probs.tolist())
                all_preds.extend(preds.tolist())
                all_true.extend(yb.long().cpu().numpy().tolist())

        y_true = np.array(all_true, dtype=int)
        y_pred = np.array(all_preds, dtype=int)

        bac = float(balanced_accuracy_score(y_true, y_pred))
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        return {
            "settings": {
                "max_lines": max_lines,
                "test_size": test_size,
                "random_state": random_state,
                "preprocessing_flag": preprocessing_flag,
                "template_flag": template_flag,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "embed_dim": embed_dim,
                "num_filters": num_filters,
                "len_percentile": len_percentile,
                "max_len": max_len,
                "max_len_cap": max_len_cap,
                "use_pos_weight": use_pos_weight,
                "pos_weight": float(pos_weight_value),
            },
            "data": {
                "n_human": int(len(human)),
                "n_ai": int(len(ai)),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_pos_train": int((y_train == 1).sum()),
                "n_neg_train": int((y_train == 0).sum()),
                "vocab_size": int(vocab_size),
            },
            "models": {
                "MultiKernelCharCNN": {
                    "balanced_accuracy": bac,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "confusion_matrix": {
                        "tn": int(tn),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tp": int(tp),
                    },
                }
            },
        }





    def combo_detector_anomaly_count(self, *, max_lines: int = 5000) -> int:        
        """
        Train combo detector on 100% of file_1 logs and test on 100% of file_2 logs.
        Returns the absolute number of anomalies detected in the test set.
        """

        if self.lines_file_1 is None or self.lines_file_2 is None:
            raise RuntimeError("lines_file_1 / lines_file_2 are not set. Call set_files() first.")

        train_logs = self.lines_file_1[:max_lines]
        test_logs  = self.lines_file_2[:max_lines]

        if len(train_logs) == 0 or len(test_logs) == 0:
            raise ValueError(f"Need non-empty train/test. Got train={len(train_logs)} test={len(test_logs)}")

        res = self._run_combo_detector(train_logs, test_logs)

        if "anomalie_count" not in res:
            raise KeyError(f"DetectMate output missing 'anomalie_count'. Keys: {list(res.keys())}")

        return int(res["anomalie_count"])


    def complexity_indices_result(
        self,
        max_lines: int = 5000,
        window_size: int = 10,
        stride: int = 10
    ) -> dict[str, dict[str, float]]:
        if self.lines_file_1 is None or self.lines_file_2 is None:
            raise RuntimeError("lines_file_1 / lines_file_2 not set. Call set_files() first.")

        lines1 = self.lines_file_1[:max_lines]
        lines2 = self.lines_file_2[:max_lines]

        m1, m2, cids1, cids2 = self._complexity_metrics_from_lines(lines1, lines2, window_size, stride)

        # enrich sample dicts with helpful context
        sample_1 = dict(m1)
        sample_2 = dict(m2)
        sample_1.update({"n_lines": float(len(lines1)), "n_templates": float(len(set(cids1)))})
        sample_2.update({"n_lines": float(len(lines2)), "n_templates": float(len(set(cids2)))})

        # absolute deltas for all shared numeric keys (skip metadata like n_lines, n_templates)
        delta: dict[str, float] = {}
        ignore = {"n_lines", "n_templates"}

        shared_keys = (set(sample_1.keys()) & set(sample_2.keys())) - ignore
        for k in shared_keys:
            v1 = sample_1.get(k)
            v2 = sample_2.get(k)

            # only compute delta for finite numeric values
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                if np.isfinite(v1) and np.isfinite(v2):
                    delta[k] = float(abs(v1 - v2))
                else:
                    delta[k] = float("nan")

        return {
            "delta": delta,
            "sample_1": sample_1,
            "sample_2": sample_2,
        }

    def one_gram_diff_report(
        self,
        max_lines: int = 5000,
        preprocessing_flag: bool = True,
        template_flag: bool = False,
        mode: str = "word",          # "word" or "char"
        top_k: int = 20,
        min_count: int = 2,          # ignore very rare tokens
        use_prob: bool = True,       # True: Δ on probabilities, False: Δ on raw counts
    ) -> dict[str, Any]:
        """
        Compare 1-gram distributions between file_1 (A) and file_2 (B) and return
        the top_k tokens by |Δ| where Δ = A - B.

        Returns a dict with:
        - settings
        - totals
        - top_differences: list of rows:
            { "token", "count_A", "count_B", "p_A", "p_B", "delta" }

        Notes:
        - If use_prob=True, p_A and p_B are token probabilities (normalized counts),
            and delta = p_A - p_B.
        - If use_prob=False, delta = count_A - count_B.
        """
        if template_flag:
            self.build_templates()
            A_lines = self.templated_file_1[:max_lines]
            B_lines = self.templated_file_2[:max_lines]
        else:
            if self.lines_file_1 is None or self.lines_file_2 is None:
                raise RuntimeError("lines_file_1 / lines_file_2 not set. Call set_files() first.")
            A_raw = self.lines_file_1[:max_lines]
            B_raw = self.lines_file_2[:max_lines]
            if preprocessing_flag:
                A_lines = [self._preprocess(x) for x in A_raw]
                B_lines = [self._preprocess(x) for x in B_raw]
            else:
                A_lines = list(A_raw)
                B_lines = list(B_raw)

        if not A_lines or not B_lines:
            raise ValueError(f"Need non-empty logs. Got A={len(A_lines)} B={len(B_lines)}")

        if mode not in ("word", "char"):
            raise ValueError("mode must be 'word' or 'char'")

        # --- tokenize into 1-grams ---
        def tokens_from_lines(lines: list[str]) -> list[str]:
            if mode == "char":
                return list("\n".join(lines))
            toks: list[str] = []
            for ln in lines:
                toks.extend(ln.split())
            return toks

        toks_A = tokens_from_lines(A_lines)
        toks_B = tokens_from_lines(B_lines)

        cnt_A = Counter(toks_A)
        cnt_B = Counter(toks_B)

        total_A = sum(cnt_A.values())
        total_B = sum(cnt_B.values())
        if total_A == 0 or total_B == 0:
            raise RuntimeError("Token totals are zero (maybe preprocessing removed everything).")

        # --- candidate vocab: union, then filter by pooled min_count ---
        pooled = cnt_A + cnt_B
        vocab = [tok for tok, c in pooled.items() if c >= min_count]

        # --- compute delta per token ---
        rows = []
        for tok in vocab:
            a = cnt_A.get(tok, 0)
            b = cnt_B.get(tok, 0)

            pA = a / total_A
            pB = b / total_B

            if use_prob:
                delta = pA - pB
            else:
                delta = float(a - b)

            rows.append((tok, a, b, pA, pB, delta))

        # sort by absolute delta
        rows.sort(key=lambda x: abs(x[5]), reverse=True)
        top = rows[:top_k]

        # format output rows
        top_rows = [
            {
                "token": tok,
                "count_A": int(a),
                "count_B": int(b),
                "p_A": float(pA),
                "p_B": float(pB),
                "delta": float(d),
            }
            for (tok, a, b, pA, pB, d) in top
        ]

        return {
            "settings": {
                "max_lines": max_lines,
                "preprocessing_flag": preprocessing_flag,
                "template_flag": template_flag,
                "mode": mode,
                "top_k": top_k,
                "min_count": min_count,
                "use_prob": use_prob,
                "delta_definition": "Δ = p_A - p_B" if use_prob else "Δ = count_A - count_B",
            },
            "totals": {
                "n_lines_A": len(A_lines),
                "n_lines_B": len(B_lines),
                "n_tokens_A": int(total_A),
                "n_tokens_B": int(total_B),
                "vocab_size_filtered": int(len(vocab)),
            },
            "top_differences": top_rows,
        }



    #------------------------------- Evaluation Methods -------------------------------

    def human_like_evaluate(
        self,
        *,
        max_lines: int = 5000,
        preprocessing_flag: bool = False,
    ) -> int:
        """
        Human-like heuristic:
        checks whether interactive commands (e.g. `tail -f`) appear in exactly
        one of the two files (symmetric difference).

        Returns:
            1 -> mismatch detected
            0 -> otherwise
        """

        if self.lines_file_1 is None or self.lines_file_2 is None:
            raise RuntimeError("lines_file_1 / lines_file_2 not set. Call set_files() first.")

        # Select lines
        lines1 = self.lines_file_1[:max_lines]
        lines2 = self.lines_file_2[:max_lines]

        if preprocessing_flag:
            lines1 = [self._preprocess(x) for x in lines1]
            lines2 = [self._preprocess(x) for x in lines2]

        # --- rule: tail -f ---
        tail_f_re = re.compile(
            r'(?<!\S)a0="tail"(?=\s|$)(?=.*(?<!\S)a\d+="-(?:f|F)"(?=\s|$))',
            re.IGNORECASE
        )

        has1 = any(tail_f_re.search(ln) for ln in lines1)
        has2 = any(tail_f_re.search(ln) for ln in lines2)
        if (has1 and not has2) or (not has1 and has2):
            return int(True)

        ### DEBUG
        '''
        if has1 or has2:
            return int(True)
        '''
            
        return int(False)

    # this evaluate function is not very meaningful right now!
    def combo_detector_evaluate(
        self,
        *,
        max_lines: int = 5000,
        test_size: float = 0.30,
        alpha: float = 0.05,
    ) -> int:


        # handle not existing files
        if self.lines_file_1 is None or self.lines_file_2 is None:
            raise RuntimeError(
                "lines_file_1 / lines_file_2 are not set. Call set_files() first."
            )

        capped_train_logs = self.lines_file_1[:max_lines]
        capped_test_logs  = self.lines_file_2[:max_lines]

        n = len(capped_train_logs)
        split = int(n * (1.0-test_size))

        train_logs  = capped_train_logs[:split]      # first 70%
        test_logs_1 = capped_train_logs[split:]      # remaining 30% holdout
        test_logs_2 = capped_test_logs               # "AI logs"

        fpr_res = self._run_combo_detector(train_logs, test_logs_1)
        fpr_events = fpr_res["event_list"]
        fpr_anomalies = fpr_res["anomalie_count"]
        fpr_total = fpr_res["total_count"]


        tpr_res = self._run_combo_detector(train_logs, test_logs_2)
        tpr_events = tpr_res["event_list"]
        tpr_anomalies = tpr_res["anomalie_count"]
        tpr_total = tpr_res["total_count"]



        ci_low, ci_high = Evaluation.newcombe_diff_ci(tpr_anomalies, tpr_total, fpr_anomalies, fpr_total, alpha=alpha)


        # require AI anomaly rate to be higher AND statistically above 0 difference
        return int(ci_low > 0)



    def n_gram_evaluate(
        self,
        *,
        max_lines: int = 5000,
        test_size: float = 0.30,
        random_state: int = 42,
        use_char_tfidf: bool = False,
        permutations: int = 200,
        alpha: float = 0.05,
        preprocessing_flag: bool = True,
        template_flag: bool = False,
        use_bonferroni: bool = False,
        window_mode: Literal["none", "raw", "cid"] = "none",
        window_size: int = 5,
    ) -> int:
        # ---------------- choose dataset ----------------
        if window_mode == "raw":
            human, ai = self.build_line_windows(
                window_size=window_size,
                max_lines=max_lines,
                preprocessing_flag=preprocessing_flag,
                template_flag=template_flag,
                stride=None,           # non-overlapping
                drop_last=True,
            )
        elif window_mode == "cid":
            if self.cid_file_1 is None or self.cid_file_2 is None:
                self.build_templates()
            human, ai = self.build_cid_windows(
                window_size=window_size,
                max_lines=max_lines,
                stride=None,           # non-overlapping
                drop_last=True,
            )
        else:
            if template_flag:
                self.build_templates()
                human_raw = self.templated_file_1[:max_lines]
                ai_raw    = self.templated_file_2[:max_lines]
                # no preprocessing needed
                human = list(human_raw)
                ai    = list(ai_raw)
            else:
                if self.lines_file_1 is None or self.lines_file_2 is None:
                    raise RuntimeError("lines_file_1 / lines_file_2 not set. Call set_files() first.")
                human_raw = self.lines_file_1[:max_lines]
                ai_raw    = self.lines_file_2[:max_lines]
                if preprocessing_flag:
                    human = [self._preprocess(x) for x in human_raw]
                    ai    = [self._preprocess(x) for x in ai_raw]
                else:
                    # no preprocessing needed
                    human = list(human_raw)
                    ai    = list(ai_raw)

        if len(human) == 0 or len(ai) == 0:
            raise ValueError(f"Need non-empty human/ai logs. Got human={len(human)} ai={len(ai)}")


        X_text = human + ai
        y = np.array([0] * len(human) + [1] * len(ai), dtype=int)

        # ---------------- vectorizer ----------------
        if use_char_tfidf:
            vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 6),
                min_df=2,              # helps reduce overfitting on tiny quirks
                sublinear_tf=True,
            )
        else:
            max_n = 3
            if window_mode != "none":
                max_n = min(window_size, 10)
            vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, max_n),
                min_df=2,
                sublinear_tf=True,
            )

        X = vectorizer.fit_transform(X_text)

        # ---------------- split ----------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # ---------------- models ----------------
        models = {
            "Linear SVM": lambda: LinearSVC(),
            "Logistic Regression": lambda: LogisticRegression(max_iter=500),
            "Naive Bayes": lambda: MultinomialNB(),
            #"Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE), # takes too long to compute for every permutation!
        }

        # multiple-testing correction (optional but recommended)
        alpha_eff = (alpha / len(models)) if use_bonferroni else alpha

        rng = np.random.default_rng(random_state)
        hits = 0
        # ---------------- evaluate each model ----------------
        for name, make_clf in models.items():
            # Fit on true labels
            clf = make_clf()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            observed = balanced_accuracy_score(y_test, y_pred)

            # Permutation null distribution (shuffle y_train only)
            null_scores = np.empty(permutations, dtype=float)
            y_train_perm = y_train.copy()

            for i in range(permutations):
                y_train_perm = rng.permutation(y_train)
                clf_i = make_clf()
                clf_i.fit(X_train, y_train_perm)
                pred_i = clf_i.predict(X_test)
                null_scores[i] = balanced_accuracy_score(y_test, pred_i)

            p_value = (np.sum(null_scores >= observed) + 1.0) / (permutations + 1.0)

            # If ANY model is significant -> True
            if p_value < alpha_eff:
                hits += 1

        return hits


    def event_time_evaluate(
        self,
        *,
        max_lines: int = 5000,
        min_events: int = 20,
        permutations: int = 500,
        alpha: float = 0.05,
        random_state: int = 42,
        use_log_bins: bool = True,
        n_bins: int = 50,
    ) -> int:

        if self.lines_file_1 is None or self.lines_file_2 is None:
            raise RuntimeError("lines_file_1 / lines_file_2 not set. Call set_files() first.")

        lines1 = self.lines_file_1[:max_lines]
        lines2 = self.lines_file_2[:max_lines]

        ts1 = Evaluation.extract_timestamps(lines1)
        ts2 = Evaluation.extract_timestamps(lines2)

        if len(ts1) < min_events or len(ts2) < min_events:
            return int(False)

        diffs1 = Evaluation.inter_event_diffs_seconds(ts1)
        diffs2 = Evaluation.inter_event_diffs_seconds(ts2)

        if len(diffs1) < (min_events - 1) or len(diffs2) < (min_events - 1):
            return int(False)


        bin_edges = Evaluation.make_inter_event_bin_edges(
            diffs1, diffs2, use_log_bins=use_log_bins, n_bins=n_bins
        )
                
        if bin_edges is None:
            return int(False)

        p1 = Evaluation.hist_prob(diffs1, bin_edges)
        p2 = Evaluation.hist_prob(diffs2, bin_edges)
        observed = Evaluation.js_divergence(p1, p2)

        # ---------------- permutation test ----------------
        rng = np.random.default_rng(random_state)

        n1 = len(diffs1)
        n2 = len(diffs2)
        combined = np.concatenate([diffs1, diffs2])

        null_stats = np.empty(permutations, dtype=float)
        for i in range(permutations):
            perm_idx = rng.permutation(n1 + n2)
            a = combined[perm_idx[:n1]]
            b = combined[perm_idx[n1:]]
            pa = Evaluation.hist_prob(a, bin_edges)
            pb = Evaluation.hist_prob(b, bin_edges)
            null_stats[i] = Evaluation.js_divergence(pa, pb)

        p_value = (np.sum(null_stats >= observed) + 1.0) / (permutations + 1.0)

        return int(p_value < alpha)


    def one_gram_evaluate(
        self,
        *,
        max_lines: int = 5000,
        preprocessing_flag: bool = True,
        template_flag: bool = False,
        mode: str = "word",          # "word" or "char"
        top_k: int = 2000,
        permutations: int = 500,
        alpha: float = 0.05,
        random_state: int = 42,
    ) -> int:

        if template_flag:
            self.build_templates()
            human_lines = self.templated_file_1[:max_lines]
            ai_lines    = self.templated_file_2[:max_lines]
        else:
            if self.lines_file_1 is None or self.lines_file_2 is None:
                raise RuntimeError("lines_file_1 / lines_file_2 not set. Call set_files() first.")
            human_raw = self.lines_file_1[:max_lines]
            ai_raw    = self.lines_file_2[:max_lines]
            if preprocessing_flag:
                human_lines = [self._preprocess(x) for x in human_raw]
                ai_lines    = [self._preprocess(x) for x in ai_raw]
            else:
                human_lines = list(human_raw)
                ai_lines    = list(ai_raw)

        if not human_lines or not ai_lines:
            raise ValueError(f"Need non-empty human/ai logs. Got human={len(human_lines)} ai={len(ai_lines)}")

        # --------- helper: extract 1-grams from a list of lines ----------
        def tokens_from_lines(lines: list[str]) -> list[str]:
            if mode == "char":
                # treat each character as a token
                return list("\n".join(lines))
            # MODE == "word"
            toks: list[str] = []
            for ln in lines:
                toks.extend(ln.split())
            return toks

        # --------- helper: make a probability vector over a fixed vocab ----------
        def prob_vector(lines: list[str], vocab: list[str]) -> np.ndarray:
            counts = Counter(tokens_from_lines(lines))
            vec = np.array([counts.get(tok, 0) for tok in vocab], dtype=float)
            s = vec.sum()
            if s == 0:
                # avoid divide-by-zero; should be rare unless lines are empty after preprocessing
                return np.ones(len(vocab), dtype=float) / len(vocab)
            return vec / s

        # --------- Jensen–Shannon divergence (bounded, symmetric) ----------
        def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
            eps = 1e-12
            p = np.clip(p, eps, 1.0)
            q = np.clip(q, eps, 1.0)
            p = p / p.sum()
            q = q / q.sum()
            m = 0.5 * (p + q)
            return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))

        # ---------------- build shared vocabulary ----------------
        # Use pooled top-K tokens to define the support for both distributions
        pooled_counts = Counter(tokens_from_lines(human_lines)) + Counter(tokens_from_lines(ai_lines))
        vocab = [tok for tok, _ in pooled_counts.most_common(top_k)]
        if len(vocab) == 0:
            raise RuntimeError("Vocabulary is empty; cannot compute 1-gram distributions.")

        # ---------------- observed statistic ----------------
        p_h = prob_vector(human_lines, vocab)
        p_ai = prob_vector(ai_lines, vocab)
        observed = js_divergence(p_h, p_ai)

        # ---------------- permutation test ----------------
        rng = np.random.default_rng(random_state)

        all_lines = human_lines + ai_lines
        labels = np.array([0] * len(human_lines) + [1] * len(ai_lines), dtype=int)

        null_stats = np.empty(permutations, dtype=float)
        for i in range(permutations):
            perm = rng.permutation(labels)
            perm_h = [all_lines[j] for j in range(len(all_lines)) if perm[j] == 0]
            perm_ai = [all_lines[j] for j in range(len(all_lines)) if perm[j] == 1]

            # keep group sizes constant (permutation does)
            p0 = prob_vector(perm_h, vocab)
            p1 = prob_vector(perm_ai, vocab)
            null_stats[i] = js_divergence(p0, p1)

        # one-sided p-value: "distance is large"
        p_value = (np.sum(null_stats >= observed) + 1.0) / (permutations + 1.0)

        return int(p_value < alpha)



    def complexity_index_evaluate(
        self,
        *,
        max_lines: int = 5000,
        permutations: int = 300,
        alpha: float = 0.05,
        random_state: int = 42,
        window_size: int = 10,
        stride: int = 10,
        use_templates: bool = True,
        use_sequences: bool = True,
        use_bonferroni: bool = False,
    ) -> int:

        if self.lines_file_1 is None or self.lines_file_2 is None:
            raise RuntimeError("lines_file_1 / lines_file_2 not set. Call set_files() first.")

        lines1 = self.lines_file_1[:max_lines]
        lines2 = self.lines_file_2[:max_lines]

        if len(lines1) == 0 or len(lines2) == 0:
            raise ValueError(f"Need non-empty logs. Got file1={len(lines1)} file2={len(lines2)}")

        m1, m2, cids1, cids2 = self._complexity_metrics_from_lines(lines1, lines2, window_size, stride)

        observed_metrics: dict[str, float] = {}

        if use_templates:
            for k in ("gini", "kurtosis", "entropy", "mad"):
                if np.isfinite(m1[k]) and np.isfinite(m2[k]):
                    observed_metrics[k] = m1[k] - m2[k]

        if use_sequences:
            for k in ("gini_seq", "kurtosis_seq", "entropy_seq", "mad_seq"):
                if np.isfinite(m1[k]) and np.isfinite(m2[k]):
                    observed_metrics[k] = m1[k] - m2[k]

        observed_metrics = {k: v for k, v in observed_metrics.items() if np.isfinite(v)}
        if not observed_metrics:
            return 0

        metric_names = list(observed_metrics.keys())
        alpha_eff = (alpha / len(metric_names)) if use_bonferroni else alpha

        rng = np.random.default_rng(random_state)

        def pvalue_template(metric_key: str) -> float:
            obs = abs(observed_metrics[metric_key])
            combined = np.array(cids1 + cids2, dtype=int)
            nA = len(cids1)

            null_vals = np.empty(permutations, dtype=float)
            for i in range(permutations):
                perm = rng.permutation(combined.size)
                A = combined[perm[:nA]].tolist()
                B = combined[perm[nA:]].tolist()
                sa = Evaluation._stats_from_ids(A)
                sb = Evaluation._stats_from_ids(B)
                null_vals[i] = abs(sa[metric_key] - sb[metric_key])

            return float((np.sum(null_vals >= obs) + 1.0) / (permutations + 1.0))

        def pvalue_seq(metric_key: str) -> float:
            obs = abs(observed_metrics[metric_key])

            win1 = list(Evaluation._sliding_windows(cids1, window_size, stride))
            win2 = list(Evaluation._sliding_windows(cids2, window_size, stride))
            nA = len(win1)
            nB = len(win2)
            if nA == 0 or nB == 0:
                return 1.0

            combined = win1 + win2

            null_vals = np.empty(permutations, dtype=float)
            for i in range(permutations):
                perm = rng.permutation(nA + nB)
                A = [combined[j] for j in perm[:nA]]
                B = [combined[j] for j in perm[nA:]]

                ca = Counter(A)
                cb = Counter(B)

                if metric_key == "gini_seq":
                    null_vals[i] = abs(Evaluation._gini_from_counts(ca) - Evaluation._gini_from_counts(cb))
                elif metric_key == "kurtosis_seq":
                    null_vals[i] = abs(Evaluation._kurtosis_from_counts(ca, True) - Evaluation._kurtosis_from_counts(cb, True))
                elif metric_key == "entropy_seq":
                    null_vals[i] = abs(Evaluation._entropy_from_counts(ca, 2.0) - Evaluation._entropy_from_counts(cb, 2.0))
                elif metric_key == "mad_seq":
                    null_vals[i] = abs(Evaluation._mad_from_counts(ca) - Evaluation._mad_from_counts(cb))
                else:
                    null_vals[i] = 0.0

            return float((np.sum(null_vals >= obs) + 1.0) / (permutations + 1.0))

        hits = 0
        for key in metric_names:
            p = pvalue_seq(key) if key.endswith("_seq") else pvalue_template(key)
            if p < alpha_eff:
                hits += 1

        return hits


    def n_gram_sequence_evaluate(self) -> bool:
        pass

    def cnn_evaluate(self) -> bool:
        pass

    def cnn_sequence_evaluate(self) -> bool:
        pass

        

if __name__ == "__main__":

    #path1 = "/home/lorenz/Documents/llm-admin-shell/Evaluation/marvin_big_log1.log"
    #path2 = "/home/lorenz/Documents/llm-admin-shell/Evaluation/marvin_big_log2.log"

    path1 = "/home/lorenz/Documents/llm-admin-shell/ExperimentResult/combine/ExperimentAggregated/Benni/audit.log"
    path2 = "/home/lorenz/Documents/llm-admin-shell/ExperimentResult/combine/ExperimentAggregated/GPT4.1/audit.log"

    evaluator = Evaluation()
    evaluator.set_files(path1, path2)


    result = evaluator.deep_learning_report(epochs=2, window_mode='raw')    


    print(result)