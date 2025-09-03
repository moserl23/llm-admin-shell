# audit_drain_gini.py
import re
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

# ---------- CONFIG ----------
LOG_PATH = Path("logs_audit.txt")
OUTPUT_DIR = Path("drain_out")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sequence-Length and Stride
WINDOW_SIZE = 10     # sequence window size as in the paper
STRIDE = 1           # slide = 1 as in the paper
# ----------------------------

AUDIT_MSG_RE = re.compile(r"msg=audit\([^)]+\):")  # mask 'msg=audit(....):'
LONG_HEX_RE  = re.compile(r"\b[0-9a-fA-F]{12,}\b")  # long hex-ish tokens
HEX_RE       = re.compile(r"\b0x[0-9a-fA-F]+\b")
PATH_RE      = re.compile(r'(?P<q>["\'])(/[^"\']+)(?P=q)')  # quoted absolute paths
IP_RE        = re.compile(r"\b(\d{1,3}\.){3}\d{1,3}\b")
NUM_RE       = re.compile(r"\b\d+\b")  # fallback number mask

def preprocess_audit_line(line: str) -> str:
    """Light, lossless-ish masking to help Drain generalize. Adjust as needed."""
    s = line.strip()
    if not s:
        return s
    s = AUDIT_MSG_RE.sub("msg=audit(<AUDIT_META>):", s)
    # mask quoted absolute paths to reduce path variability, but keep shape:
    s = PATH_RE.sub('"</PATH>"', s)
    s = LONG_HEX_RE.sub("<HEX>", s)
    s = HEX_RE.sub("<HEX>", s)
    s = IP_RE.sub("<IP>", s)
    # Optional: mask standalone large integers. You may want to keep small ints (e.g., type codes).
    s = NUM_RE.sub("<NUM>", s)
    return s

def gini_from_counts(counter: Counter) -> float:
    """
    Gini (simplified) as used in the paper (Eq. 2):
      G = 2 * sum(i * x_i) / (n * sum x) - (n + 1) / n
    where x_i are sorted ascending, n is number of categories.
    """
    if not counter:
        return float("nan")
    x = np.array(sorted(counter.values()))
    n = x.size
    return 2.0 * (np.arange(1, n + 1) * x).sum() / (n * x.sum()) - (n + 1) / n


def kurtosis_from_counts(counter: Counter, convexify: bool = True) -> float:
    """
    Kurtosis (Fisher's definition: unbiased excess kurtosis) on frequency-of-occurrence
    as described in the paper. Preprocessing: make a 'convex' sequence by concatenating
    ascending- and descending-sorted frequency lists. Returns np.nan if undefined.

    Reference: uses frequency values -> make convex graph (asc + desc) -> Fisher kurtosis
    where normal distribution has 0.  See paper's Evaluation Metric 4. 
    """
    import numpy as np

    # get raw frequencies
    freq = list(counter.values())
    if len(freq) == 0:
        return float("nan")

    # paper's preprocessing step (convex graph)
    if convexify:
        asc = sorted(freq)
        desc = sorted(freq, reverse=True)
        x = np.asarray(asc + desc, dtype=float)
    else:
        x = np.asarray(freq, dtype=float)

    n = x.size
    # Fisher's definition requires n >= 4
    if n < 4:
        return float("nan")

    mean = x.mean()
    d = x - mean
    # sample variance with (n-1) in denominator
    s2 = (d @ d) / (n - 1)
    if s2 == 0:
        return float("nan")
    s4 = s2 ** 2
    m4 = np.sum(d ** 4)  # raw 4th central moment (sum, not averaged)

    # Fisher's unbiased excess kurtosis (normal -> 0)
    # g2 = [n(n+1)/((n-1)(n-2)(n-3))] * (m4 / s4) - [3(n-1)^2/((n-2)(n-3))]
    g2 = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * (m4 / s4) \
         - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return float(g2)

def entropy_from_counts(counter: Counter, base: float = 2.0) -> float:
    """
    Evaluation Metric 6 (Entropy), exactly as in the paper:
      1) Convert frequencies to probabilities p_i = x_i / sum(x)
      2) H = - sum_i p_i * log_base(p_i), with base = 2 by default (log2)
    High entropy => frequencies are evenly spread => higher complexity.
    Ref: Eq. (3) & text describing probability preprocessing. 
    """
    if not counter:
        return float("nan")
    x = np.array(list(counter.values()), dtype=float)
    total = x.sum()
    if total <= 0:
        return float("nan")
    p = x / total
    # guard against any numerical zeros
    p = p[p > 0]

    if base == 2.0:
        h = -np.sum(p * np.log2(p))
    elif base == np.e:
        h = -np.sum(p * np.log(p))
    else:
        h = -np.sum(p * (np.log(p) / np.log(base)))
    return float(h)


def mad_from_counts(counter: Counter) -> float:
    """
    Evaluation Metric 7 (Mean Absolute Deviation), exactly as in the paper:
      MAD = (1/N) * sum_i |x_i - mean(x)|
    Lower MAD => frequencies closer to the mean => higher complexity.
    Ref: Eq. (4) and accompanying explanation.
    """
    if not counter:
        return float("nan")
    x = np.array(list(counter.values()), dtype=float)
    mean = x.mean()
    return float(np.mean(np.abs(x - mean)))

def sliding_windows(seq, size=10, stride=1):
    for i in range(0, len(seq) - size + 1, stride):
        yield tuple(seq[i:i + size])

def main():
    if not LOG_PATH.exists():
        raise SystemExit(f"File not found: {LOG_PATH}")

    # Initialize Drain3 with our parameters
    cfg = TemplateMinerConfig()
    cfg.load("drain3.ini") 

    miner = TemplateMiner(config=cfg)

    template_ids_seq = []           # ordered sequence of template ids per line
    template_id_to_template = {}    # map to the mined template text
    line_count = 0

    with LOG_PATH.open("r", errors="ignore") as f:
        for raw in tqdm(f, desc="Parsing logs"):
            line_count += 1
            msg = preprocess_audit_line(raw)
            if not msg:
                continue
            result = miner.add_log_message(msg)
            # Drain3 returns dict with cluster_id & template_mined
            cid = result.get("cluster_id")
            tpl = result.get("template_mined")
            if cid is None:
                # Safety: if Drain didn't return an id, skip
                continue
            template_ids_seq.append(cid)
            if tpl:
                template_id_to_template[cid] = tpl


    # Histogram of templates
    tpl_counts = Counter(template_ids_seq)
    gini_templates = gini_from_counts(tpl_counts)
    kurtosis_templates = kurtosis_from_counts(tpl_counts, convexify=True)
    entropy_templates = entropy_from_counts(tpl_counts)
    mad_templates = mad_from_counts(tpl_counts)

    # Sequence of template-IDs (size=10, stride=1) -> histogram
    seq_counts = Counter(sliding_windows(template_ids_seq, WINDOW_SIZE, STRIDE))
    gini_sequences = gini_from_counts(seq_counts)
    kurtosis_sequences = kurtosis_from_counts(seq_counts, convexify=True)
    entropy_sequences = entropy_from_counts(seq_counts)
    mad_sequences = mad_from_counts(seq_counts)

    

    # Output gini
    print("Gini(templates):", gini_templates)
    print("Gini(seq win=10):", gini_sequences)
    print("Kurt(templates):", kurtosis_templates)
    print("Kurt(seq win=10):", kurtosis_sequences)
    print("Entropy(templates):", entropy_templates)
    print("Entropy(seq win=10):", entropy_sequences)
    print("MAD(templates):", mad_templates)
    print("MAD(seq win=10):", mad_sequences)



if __name__ == "__main__":
    main()
