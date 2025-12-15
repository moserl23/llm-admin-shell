from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import re

# -----------------------------------------
# High-quality preprocessing for audit logs
# -----------------------------------------

AUDIT_MSG_RE = re.compile(r"msg=audit\([^)]+\):")
LONG_HEX_RE  = re.compile(r"\b[0-9a-fA-F]{12,}\b")
HEX_RE       = re.compile(r"\b0x[0-9a-fA-F]+\b")
PATH_RE      = re.compile(r'(?P<q>["\'])(/[^"\']+)(?P=q)')
IP_RE        = re.compile(r"\b(\d{1,3}\.){3}\d{1,3}\b")
NUM_RE       = re.compile(r"\b\d+\b")

def preprocess(line: str) -> str:
    """Lossless-ish masking to help Drain generalize."""
    s = line.strip()
    if not s:
        return s
    s = AUDIT_MSG_RE.sub("msg=audit(<AUDIT_META>):", s)
    s = PATH_RE.sub('"/PATH"', s)
    s = LONG_HEX_RE.sub("<HEX>", s)
    s = HEX_RE.sub("<HEX>", s)
    s = IP_RE.sub("<IP>", s)
    s = NUM_RE.sub("<NUM>", s)
    return s

# -----------------------------------------
# Sliding window of template IDs
# -----------------------------------------

def sliding_windows(seq, size=10):
    return [seq[i:i + size] for i in range(len(seq) - size + 1)]

# -----------------------------------------
# Drain3 template extraction for log list
# -----------------------------------------

def template_id_stream(log_lines):
    cfg = TemplateMinerConfig()
    cfg.load("drain3.ini")
    miner = TemplateMiner(config=cfg, persistence_handler=None)

    ids = []
    for line in log_lines:
        msg = preprocess(line)
        if not msg:
            continue
        result = miner.add_log_message(msg)
        cid = result.get("cluster_id")
        if cid:
            ids.append(cid)

    return ids

# -----------------------------------------
# Build dataset from human + AI logs
# -----------------------------------------

def build_dataset(human_logs, ai_logs, window=10):
    human_ids = template_id_stream(human_logs)
    ai_ids    = template_id_stream(ai_logs)

    human_seq = sliding_windows(human_ids, window)
    ai_seq    = sliding_windows(ai_ids, window)

    X_raw = human_seq + ai_seq
    y_raw = [0]*len(human_seq) + [1]*len(ai_seq)

    # collect all unique template IDs across all sequences
    all_ids = sorted(set(t for seq in X_raw for t in seq))

    enc = LabelEncoder()
    enc.fit(all_ids)

    # integer encode each sequence
    X = np.array([enc.transform(seq) for seq in X_raw])
    y = np.array(y_raw)

    return X, y

# -----------------------------------------
# Train and evaluate classifier
# -----------------------------------------

def train_and_test(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = LinearSVC()
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xte)
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(yte, preds))
    return clf

# -----------------------------------------
# MAIN — EXAMPLE USAGE
# (replace `human_logs` and `ai_logs` with yours)
# -----------------------------------------

if __name__ == "__main__":

    # Example: user-provided logs as lists of strings
    human_logs = [
        'type=EXECVE msg=audit(1763731451.292:55975): argc=3 a0="sudo" a1="apt" a2="update"',
        'type=SYSCALL msg=audit(1763731452.111:55990): arch=c000003e syscall=2 success=yes comm="bash" exe="/usr/bin/bash"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',        
        # ...
    ]

    ai_logs = [
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        # ...
    ]

    X, y = build_dataset(human_logs, ai_logs, window=10)
    print("Dataset shape:", X.shape)

    clf = train_and_test(X, y)

    # Example: classify a single window
    print("Prediction for first sample:", clf.predict([X[0]]))
