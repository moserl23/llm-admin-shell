from pathlib import Path
from typing import Any, Optional, List

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
from sklearn.metrics import balanced_accuracy_score    


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

    # ---------------- routing ----------------
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

    # ---------------- constructor ----------------
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


    @staticmethod
    def read_file(file_path: str|Path) -> list[str]:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        # splitlines() removes trailing "\n" cleanly and handles last-line-no-newline well
        return path.read_text(encoding="utf-8").splitlines()


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



    def set_files(self, file_path_1: str | Path, file_path_2: str | Path) -> None:
        self.file_path_1 = file_path_1
        self.file_path_2 = file_path_2
        self.lines_file_1 = self.read_file(self.file_path_1)
        self.lines_file_2 = self.read_file(self.file_path_2)

    def build_templates(self) -> None:
        all_logs = self.lines_file_1 + self.lines_file_2

        templates, cluster_ids = Evaluation.logs_to_templates(all_logs)

        n1 = len(self.lines_file_1)

        # Templates
        self.templated_file_1 = templates[:n1]
        self.templated_file_2 = templates[n1:]

        # Cluster IDs
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


    #------------------------------- Evaluation Methods -------------------------------


    def combo_detector_evaluate(self) -> int:

        # ---------------- config knobs ----------------
        MAX_LINES = 5000
        TEST_SIZE = 0.30
        ALPHA = 0.05

            
        # hard coded paths for my system
        DETECTMATE_DIR = Path("/home/lorenz/Documents/DetectMate/DetectMateLibrary")
        DETECTMATE_ENTRY = "MA_lorenz.py"
        COMBO_LOG_PATH = Path("/home/lorenz/Documents/DetectMate/Logs/IDS_audit.log")

        # handle not existing files
        if self.lines_file_1 is None or self.lines_file_2 is None:
            raise RuntimeError(
                "lines_file_1 / lines_file_2 are not set. Call set_files() first."
            )

        def combo_detect_bridge(train_log_sample: list[str], test_log_sample: list[str]):

            train_size = len(train_log_sample)
            test_size = len(test_log_sample)

            # handle empty files
            if train_size == 0 or test_size == 0:
                raise ValueError(
                    f"Need non-empty train/test. Got train_size={train_size}, test_size={test_size}."
                )

            # 1) Write combined file: TRAIN then TEST
            log_path = Path(COMBO_LOG_PATH)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            with log_path.open("w", encoding="utf-8") as f:
                for line in train_log_sample:
                    f.write(line.rstrip("\n") + "\n")
                for line in test_log_sample:
                    f.write(line.rstrip("\n") + "\n")

            # 2) Run DetectMate
            detectmate_dir = Path(DETECTMATE_DIR)
            if not detectmate_dir.is_dir():
                raise FileNotFoundError(f"DetectMate dir not found: {detectmate_dir}")

            entry = DETECTMATE_ENTRY
            cmd = [
                "uv",
                "run",
                "python",
                entry,
                "--train-size", str(train_size),
                "--test-size", str(test_size),
            ]

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(detectmate_dir),
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                # Bubble up a useful error (stdout/stderr are *very* helpful here)
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
                    "Combo detector returned empty stdout. "
                    f"stderr was:\n{proc.stderr}"
                )
            
            # 3) Parse JSON output
            try:
                result: Any = json.loads(stdout)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    "Failed to parse combo detector output as JSON.\n"
                    f"stdout was:\n{stdout}\n"
                    f"stderr was:\n{proc.stderr}"
                ) from e
            
            return result
        
        capted_train_logs = self.lines_file_1[:MAX_LINES]
        capted_test_logs  = self.lines_file_2[:MAX_LINES]

        n = len(capted_train_logs)
        split = int(n * (1.0-TEST_SIZE))

        train_logs  = capted_train_logs[:split]      # first 70%
        test_logs_1 = capted_train_logs[split:]      # remaining 30% holdout
        test_logs_2 = capted_test_logs               # "AI logs"


        fpr_res = combo_detect_bridge(train_logs, test_logs_1)
        fpr_anomalies = fpr_res["anomalie_count"]
        fpr_total = fpr_res["total_count"]

            
        tpr_res = combo_detect_bridge(train_logs, test_logs_2)
        tpr_anomalies = tpr_res["anomalie_count"]
        tpr_total = tpr_res["total_count"]        


        ci_low, ci_high = Evaluation.newcombe_diff_ci(tpr_anomalies, tpr_total, fpr_anomalies, fpr_total, alpha=ALPHA)


        # require AI anomaly rate to be higher AND statistically above 0 difference
        return int(ci_low > 0)



    def n_gram_evaluate(self) -> int:
        
        # ---------------- config knobs ----------------
        MAX_LINES = 5000
        TEST_SIZE = 0.30
        RANDOM_STATE = 42
        USE_CHAR_TFIDF = False      # flip True if you want char n-grams
        PERMUTATIONS = 200          # 100–1000 is typical; start 200 for speed
        ALPHA = 0.05
        PREPROCESSING_FLAG = True
        TEMPLATE_FLAG = False
        USE_BONFERRONI = False       # recommended if you do "any model significant"
        # ---------------- choose dataset ----------------

        if TEMPLATE_FLAG:
            if self.templated_file_1 is None or self.templated_file_2 is None:
                raise RuntimeError(
                    "TEMPLATE_FLAG=True but templated_file_1/2 are not set. Call build_templates() first."
                )
            human_raw = self.templated_file_1[:MAX_LINES]
            ai_raw    = self.templated_file_2[:MAX_LINES]
            # no preprocessing needed
            human = list(human_raw)
            ai    = list(ai_raw)
        else:
            if self.lines_file_1 is None or self.lines_file_2 is None:
                raise RuntimeError("lines_file_1 / lines_file_2 not set. Call set_files() first.")
            human_raw = self.lines_file_1[:MAX_LINES]
            ai_raw    = self.lines_file_2[:MAX_LINES]
            if PREPROCESSING_FLAG:
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
        if USE_CHAR_TFIDF:
            vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 6),
                min_df=2,              # helps reduce overfitting on tiny quirks
                sublinear_tf=True,
            )
        else:
            vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 3),
                min_df=2,
                sublinear_tf=True,
            )

        X = vectorizer.fit_transform(X_text)

        # ---------------- split ----------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
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
        alpha_eff = (ALPHA / len(models)) if USE_BONFERRONI else ALPHA

        rng = np.random.default_rng(RANDOM_STATE)
        hits = 0
        # ---------------- evaluate each model ----------------
        for name, make_clf in models.items():
            # Fit on true labels
            clf = make_clf()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            observed = balanced_accuracy_score(y_test, y_pred)

            # Permutation null distribution (shuffle y_train only)
            null_scores = np.empty(PERMUTATIONS, dtype=float)
            y_train_perm = y_train.copy()

            for i in range(PERMUTATIONS):
                y_train_perm = rng.permutation(y_train)
                clf_i = make_clf()
                clf_i.fit(X_train, y_train_perm)
                pred_i = clf_i.predict(X_test)
                null_scores[i] = balanced_accuracy_score(y_test, pred_i)

            p_value = (np.sum(null_scores >= observed) + 1.0) / (PERMUTATIONS + 1.0)

            # If ANY model is significant -> True
            if p_value < alpha_eff:
                hits += 1

        return hits


    def event_time_evaluate(self) -> int:
        # ---------------- config knobs ----------------
        MAX_LINES = 5000
        MIN_EVENTS = 20          # minimum timestamps per file to be meaningful
        PERMUTATIONS = 500       # 200–2000 typical
        ALPHA = 0.05
        RANDOM_STATE = 42

        # For histogram-based distribution (handles heavy tails nicely)
        USE_LOG_BINS = True
        N_BINS = 50

        if self.lines_file_1 is None or self.lines_file_2 is None:
            raise RuntimeError("lines_file_1 / lines_file_2 not set. Call set_files() first.")

        lines1 = self.lines_file_1[:MAX_LINES]
        lines2 = self.lines_file_2[:MAX_LINES]

        TIME_RE = re.compile(r'"time"\s*:\s*"([^"]+)"')
        AUDIT_TIMESTAMP_RE = re.compile(r"audit\((\d+\.\d+):")

        def extract_nextcloud_timestamps(lines: list[str]) -> list[datetime]:
            timestamps: list[datetime] = []
            for line in lines:
                match = TIME_RE.search(line)
                if match:
                    try:
                        timestamps.append(datetime.fromisoformat(match.group(1)))
                    except Exception:
                        continue
            return timestamps

        def extract_auditlog_timestamps(lines: list[str]) -> list[datetime]:
            timestamps: list[datetime] = []
            for line in lines:
                match = AUDIT_TIMESTAMP_RE.search(line)
                if match:
                    epoch_str = match.group(1)
                    try:
                        timestamps.append(datetime.fromtimestamp(float(epoch_str)))
                    except Exception:
                        continue
            return timestamps

        def extract_syslog_timestamps(lines: list[str]) -> list[datetime]:
            timestamps: list[datetime] = []
            for line in lines:
                try:
                    ts_str = line.split(" ", 1)[0]
                    timestamps.append(datetime.fromisoformat(ts_str))
                except Exception:
                    continue
            return timestamps

        def extract_generic_timestamps(lines: list[str]) -> list[datetime]:
            # fallback: try "first token is iso timestamp"
            timestamps: list[datetime] = []
            for line in lines:
                s = line.strip()
                if not s:
                    continue
                first = s.split(" ", 1)[0]
                try:
                    timestamps.append(datetime.fromisoformat(first))
                except Exception:
                    continue
            return timestamps

        def extract_timestamps(lines: list[str]) -> list[datetime]:
            """
            Prefer a single extraction strategy based on the dominant detected type.
            (Mixing types in the same file is possible; this is a pragmatic choice.)
            """
            kinds = [Evaluation._detect_type(ln) for ln in lines[:200] if ln.strip()] # only use the first 200 lines to determine the type
            if not kinds:
                return []

            dominant = Counter(kinds).most_common(1)[0][0]

            if dominant == "audit":
                ts = extract_auditlog_timestamps(lines)
            elif dominant == "nextcloud":
                ts = extract_nextcloud_timestamps(lines)
            elif dominant == "syslog":
                ts = extract_syslog_timestamps(lines)
            else:
                ts = extract_generic_timestamps(lines)

            ts.sort()
            return ts
        
        # ---------------- inter-event time diffs ----------------
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

        ts1 = extract_timestamps(lines1)
        ts2 = extract_timestamps(lines2)

        if len(ts1) < MIN_EVENTS or len(ts2) < MIN_EVENTS:
            return int(False)

        diffs1 = inter_event_diffs_seconds(ts1)
        diffs2 = inter_event_diffs_seconds(ts2)

        if len(diffs1) < (MIN_EVENTS - 1) or len(diffs2) < (MIN_EVENTS - 1):
            return int(False)

        # ---------------- JS divergence over shared histogram bins ----------------
        def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
            eps = 1e-12
            p = np.clip(p, eps, 1.0)
            q = np.clip(q, eps, 1.0)
            p = p / p.sum()
            q = q / q.sum()
            m = 0.5 * (p + q)
            return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))

        def to_hist_prob(diffs: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
            h, _ = np.histogram(diffs, bins=bin_edges)
            h = h.astype(float)
            s = h.sum()
            if s <= 0:
                return np.ones(len(h), dtype=float) / len(h)
            return h / s

        all_diffs = np.concatenate([diffs1, diffs2])
        all_diffs = all_diffs[np.isfinite(all_diffs)]
        all_diffs = all_diffs[all_diffs >= 0]

        if len(all_diffs) == 0:
            return int(False)

        dmin = float(np.min(all_diffs))
        dmax = float(np.max(all_diffs))
        if dmax == dmin:
            return int(False)

        if USE_LOG_BINS:
            lo = max(dmin, 1e-6)
            hi = max(dmax, lo * 1.000001)
            bin_edges = np.logspace(np.log10(lo), np.log10(hi), N_BINS + 1)
            if dmin <= 0.0:
                bin_edges = np.concatenate(([0.0], bin_edges))
        else:
            bin_edges = np.linspace(dmin, dmax, N_BINS + 1)

        p1 = to_hist_prob(diffs1, bin_edges)
        p2 = to_hist_prob(diffs2, bin_edges)
        observed = js_divergence(p1, p2)

        # ---------------- permutation test ----------------
        rng = np.random.default_rng(RANDOM_STATE)

        n1 = len(diffs1)
        n2 = len(diffs2)
        combined = np.concatenate([diffs1, diffs2])

        null_stats = np.empty(PERMUTATIONS, dtype=float)
        for i in range(PERMUTATIONS):
            perm_idx = rng.permutation(n1 + n2)
            a = combined[perm_idx[:n1]]
            b = combined[perm_idx[n1:]]
            pa = to_hist_prob(a, bin_edges)
            pb = to_hist_prob(b, bin_edges)
            null_stats[i] = js_divergence(pa, pb)

        p_value = (np.sum(null_stats >= observed) + 1.0) / (PERMUTATIONS + 1.0)

        return int(p_value < ALPHA)


    def one_gram_evaluate(self) -> int:
        # ---------------- knobs ----------------
        MAX_LINES = 5000
        PREPROCESSING_FLAG = True      # use self._preprocess on raw logs
        TEMPLATE_FLAG = False          # if True: use templated_file_1/2, skip preprocessing

        MODE = "word"                  # "word" or "char"
        TOP_K = 2000                   # keep only top-K tokens (reduces noise + speeds permutations)
        PERMUTATIONS = 500             # 200–2000 typical
        ALPHA = 0.05
        RANDOM_STATE = 42
        # --------------------------------------

        if TEMPLATE_FLAG:
            if self.templated_file_1 is None or self.templated_file_2 is None:
                raise RuntimeError("TEMPLATE_FLAG=True but templates not built. Call build_templates() first.")
            human_lines = self.templated_file_1[:MAX_LINES]
            ai_lines    = self.templated_file_2[:MAX_LINES]
        else:
            if self.lines_file_1 is None or self.lines_file_2 is None:
                raise RuntimeError("lines_file_1 / lines_file_2 not set. Call set_files() first.")
            human_raw = self.lines_file_1[:MAX_LINES]
            ai_raw    = self.lines_file_2[:MAX_LINES]
            if PREPROCESSING_FLAG:
                human_lines = [self._preprocess(x) for x in human_raw]
                ai_lines    = [self._preprocess(x) for x in ai_raw]
            else:
                human_lines = list(human_raw)
                ai_lines    = list(ai_raw)

        if not human_lines or not ai_lines:
            raise ValueError(f"Need non-empty human/ai logs. Got human={len(human_lines)} ai={len(ai_lines)}")

        # --------- helper: extract 1-grams from a list of lines ----------
        def tokens_from_lines(lines: list[str]) -> list[str]:
            if MODE == "char":
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
        vocab = [tok for tok, _ in pooled_counts.most_common(TOP_K)]
        if len(vocab) == 0:
            raise RuntimeError("Vocabulary is empty; cannot compute 1-gram distributions.")

        # ---------------- observed statistic ----------------
        p_h = prob_vector(human_lines, vocab)
        p_ai = prob_vector(ai_lines, vocab)
        observed = js_divergence(p_h, p_ai)

        # ---------------- permutation test ----------------
        rng = np.random.default_rng(RANDOM_STATE)

        all_lines = human_lines + ai_lines
        labels = np.array([0] * len(human_lines) + [1] * len(ai_lines), dtype=int)

        null_stats = np.empty(PERMUTATIONS, dtype=float)
        for i in range(PERMUTATIONS):
            perm = rng.permutation(labels)
            perm_h = [all_lines[j] for j in range(len(all_lines)) if perm[j] == 0]
            perm_ai = [all_lines[j] for j in range(len(all_lines)) if perm[j] == 1]

            # keep group sizes constant (permutation does)
            p0 = prob_vector(perm_h, vocab)
            p1 = prob_vector(perm_ai, vocab)
            null_stats[i] = js_divergence(p0, p1)

        # one-sided p-value: "distance is large"
        p_value = (np.sum(null_stats >= observed) + 1.0) / (PERMUTATIONS + 1.0)

        return int(p_value < ALPHA)



    def complexit_index_evaluate(self) -> int:
        # ---------------- config knobs ----------------
        MAX_LINES = 5000
        PERMUTATIONS = 300          # 200–1000 typical; increase if you want more power
        ALPHA = 0.05
        RANDOM_STATE = 42

        WINDOW_SIZE = 10
        STRIDE = 10

        USE_TEMPLATES = True        # template-frequency metrics
        USE_SEQUENCES = True        # sequence-window-frequency metrics

        USE_BONFERRONI = False       # recommended if "any metric significant" => detection
        # ---------------------------------------------

        if self.lines_file_1 is None or self.lines_file_2 is None:
            raise RuntimeError("lines_file_1 / lines_file_2 not set. Call set_files() first.")

        lines1 = self.lines_file_1[:MAX_LINES]
        lines2 = self.lines_file_2[:MAX_LINES]

        if len(lines1) == 0 or len(lines2) == 0:
            raise ValueError(f"Need non-empty logs. Got file1={len(lines1)} file2={len(lines2)}")

        # ---------- helpers: metrics (from your reference code) ----------
        def gini_from_counts(counter: Counter) -> float:
            if not counter:
                return float("nan")
            x = np.array(sorted(counter.values()), dtype=float)
            n = x.size
            s = x.sum()
            if s <= 0 or n == 0:
                return float("nan")
            return 2.0 * (np.arange(1, n + 1) * x).sum() / (n * s) - (n + 1) / n

        def kurtosis_from_counts(counter: Counter, convexify: bool = True) -> float:
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

        def entropy_from_counts(counter: Counter, base: float = 2.0) -> float:
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

        def mad_from_counts(counter: Counter) -> float:
            if not counter:
                return float("nan")
            x = np.array(list(counter.values()), dtype=float)
            if x.size == 0:
                return float("nan")
            mean = x.mean()
            return float(np.mean(np.abs(x - mean)))

        def sliding_windows(seq: list[int], size: int = 10, stride: int = 1):
            for i in range(0, len(seq) - size + 1, stride):
                yield tuple(seq[i:i + size])

        # ---------- Step 1: get template cluster IDs for both logs ----------
        # Reuse your Drain pipeline so both files share the SAME clustering space:
        # build on concatenated logs, then split cluster ids back.
        templates, cluster_ids = Evaluation.logs_to_templates(lines1 + lines2)
        n1 = len(lines1)
        cids1 = cluster_ids[:n1]
        cids2 = cluster_ids[n1:]

        # Need enough data for sequence windows
        if USE_SEQUENCES and (len(cids1) < WINDOW_SIZE or len(cids2) < WINDOW_SIZE):
            # not enough for sequences; fall back to template metrics only
            USE_SEQUENCES = False

        # ---------- Step 2: define "statistic" functions ----------
        # Each statistic takes a "sample" and outputs ONE number.
        def stats_from_ids(ids: list[int]) -> dict[str, float]:
            cnt = Counter(ids)
            return {
                "gini": gini_from_counts(cnt),
                "kurtosis": kurtosis_from_counts(cnt, convexify=True),
                "entropy": entropy_from_counts(cnt, base=2.0),
                "mad": mad_from_counts(cnt),
            }

        def stats_from_windows(ids: list[int]) -> dict[str, float]:
            win = list(sliding_windows(ids, WINDOW_SIZE, STRIDE))
            cnt = Counter(win)
            return {
                "gini_seq": gini_from_counts(cnt),
                "kurtosis_seq": kurtosis_from_counts(cnt, convexify=True),
                "entropy_seq": entropy_from_counts(cnt, base=2.0),
                "mad_seq": mad_from_counts(cnt),
            }

        observed_metrics: dict[str, float] = {}
        if USE_TEMPLATES:
            a = stats_from_ids(cids1)
            b = stats_from_ids(cids2)
            for k in a.keys():
                observed_metrics[k] = a[k] - b[k]

        if USE_SEQUENCES:
            a = stats_from_windows(cids1)
            b = stats_from_windows(cids2)
            for k in a.keys():
                observed_metrics[k] = a[k] - b[k]

        # Guard: if everything is NaN (can happen with tiny vocab)
        observed_metrics = {k: v for k, v in observed_metrics.items() if np.isfinite(v)}
        if not observed_metrics:
            return int(False)

        metric_names = list(observed_metrics.keys())

        # Multiple testing correction if you do "any metric significant"
        alpha_eff = (ALPHA / len(metric_names)) if USE_BONFERRONI else ALPHA

        rng = np.random.default_rng(RANDOM_STATE)

        # ---------- Step 3: permutation tests ----------
        # We compute a p-value per metric; return True if any p < alpha_eff.

        def permutation_pvalue_template_metric(metric_key: str) -> float:
            obs = abs(observed_metrics[metric_key])

            combined = np.array(cids1 + cids2, dtype=int)
            nA = len(cids1)
            nB = len(cids2)

            null_vals = np.empty(PERMUTATIONS, dtype=float)
            for i in range(PERMUTATIONS):
                perm = rng.permutation(nA + nB)
                A = combined[perm[:nA]].tolist()
                B = combined[perm[nA:]].tolist()
                sa = stats_from_ids(A)
                sb = stats_from_ids(B)
                null_vals[i] = abs(sa[metric_key] - sb[metric_key])

            # two-sided via |delta|
            p = (np.sum(null_vals >= obs) + 1.0) / (PERMUTATIONS + 1.0)
            return float(p)

        def permutation_pvalue_sequence_metric(metric_key: str) -> float:
            obs = abs(observed_metrics[metric_key])

            win1 = list(sliding_windows(cids1, WINDOW_SIZE, STRIDE))
            win2 = list(sliding_windows(cids2, WINDOW_SIZE, STRIDE))
            nA = len(win1)
            nB = len(win2)

            if nA == 0 or nB == 0:
                return 1.0

            combined = win1 + win2

            null_vals = np.empty(PERMUTATIONS, dtype=float)
            for i in range(PERMUTATIONS):
                perm = rng.permutation(nA + nB)
                A = [combined[j] for j in perm[:nA]]
                B = [combined[j] for j in perm[nA:]]

                ca = Counter(A)
                cb = Counter(B)

                # compute stats directly from counts to avoid rebuilding windows each time
                if metric_key == "gini_seq":
                    null_vals[i] = abs(gini_from_counts(ca) - gini_from_counts(cb))
                elif metric_key == "kurtosis_seq":
                    null_vals[i] = abs(kurtosis_from_counts(ca, convexify=True) - kurtosis_from_counts(cb, convexify=True))
                elif metric_key == "entropy_seq":
                    null_vals[i] = abs(entropy_from_counts(ca, base=2.0) - entropy_from_counts(cb, base=2.0))
                elif metric_key == "mad_seq":
                    null_vals[i] = abs(mad_from_counts(ca) - mad_from_counts(cb))
                else:
                    null_vals[i] = 0.0  # should not happen

            p = (np.sum(null_vals >= obs) + 1.0) / (PERMUTATIONS + 1.0)
            return float(p)

        hits = 0
        for key in metric_names:
            if key.endswith("_seq"):
                p = permutation_pvalue_sequence_metric(key)
            else:
                p = permutation_pvalue_template_metric(key)

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

    path1 = "/home/lorenz/Documents/llm-admin-shell/Evaluation/marvin_big_log1.log"
    path2 = "/home/lorenz/Documents/llm-admin-shell/Evaluation/marvin_big_log2.log"

    evaluator = Evaluation()
    evaluator.set_files(path1, path2)
    evaluator.build_templates()

    if False:
        print("\n\nEvent-Time:")
        detected = evaluator.event_time_evaluate() # one_gram_evaluate, n_gram_evaluate, combo_detector_evaluate
        if detected:
            print("Difference Detected!")
        else:
            print("Nothing detected!")

    if False:
        print("\n\nOne-Gram:")
        detected = evaluator.one_gram_evaluate()
        if detected:
            print("Difference Detected!")
        else:
            print("Nothing detected!")

    if True:
        print("\n\nN-Gram:")
        detected = evaluator.n_gram_evaluate()
        if detected:
            print("Difference Detected!")
        else:
            print("Nothing detected!")

    if False:
        print("\n\nCombo-Detector:")
        detected = evaluator.combo_detector_evaluate()
        if detected:
            print("Difference Detected!")
        else:
            print("Nothing detected!")

    if False:
        print("\n\nComplexity-Index:")
        detected = evaluator.complexit_index_evaluate()
        if detected:
            print("Difference Detected!")
        else:
            print("Nothing detected!")