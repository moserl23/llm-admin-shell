"""Character-level CNN pipeline for log classification experiments.

The module keeps train/validation/test separation strict by deriving the
character vocabulary and sequence length from the training split only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable, Tuple, Any

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

from src.core.ml.data import Example
from src.core.ml.splits import Split
from src.core.ml.eval import EvalResult, evaluate_classifier


# ---- Dataset ----
class EncodedLogDataset(Dataset):
    """Minimal dataset wrapper for padded character sequences and label ids."""

    def __init__(self, X: np.ndarray, y_ids: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


# ---- Model ----
class MultiKernelCharCNN(nn.Module):
    """Character CNN with parallel convolutional kernels over one embedding table.

    The architecture captures short- and medium-range character patterns and
    returns class logits for each input sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        *,
        embed_dim: int = 64,
        num_filters: int = 64,
        fc_dim: int = 128,
        dropout: float = 0.5,
        kernel_sizes: Tuple[int, int, int] = (3, 5, 7),
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        k3, k5, k7 = kernel_sizes
        self.conv3 = nn.Conv1d(embed_dim, num_filters, kernel_size=k3, padding=k3 // 2)
        self.conv5 = nn.Conv1d(embed_dim, num_filters, kernel_size=k5, padding=k5 // 2)
        self.conv7 = nn.Conv1d(embed_dim, num_filters, kernel_size=k7, padding=k7 // 2)

        self.bn3 = nn.BatchNorm1d(num_filters)
        self.bn5 = nn.BatchNorm1d(num_filters)
        self.bn7 = nn.BatchNorm1d(num_filters)

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(num_filters * 3, fc_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of encoded character sequences."""

        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        x3 = torch.relu(self.bn3(self.conv3(x)))
        x5 = torch.relu(self.bn5(self.conv5(x)))
        x7 = torch.relu(self.bn7(self.conv7(x)))

        x3 = self.pool(x3).squeeze(-1)
        x5 = self.pool(x5).squeeze(-1)
        x7 = self.pool(x7).squeeze(-1)

        x = torch.cat([x3, x5, x7], dim=1)
        x = self.dropout(torch.relu(self.fc1(x)))
        logits = self.fc2(x)
        return logits


# ---- Config ----
@dataclass(frozen=True)
class CNNConfig:
    """Configuration for encoding, optimization, and early stopping."""

    # text encoding
    max_len_cap: int = 512
    len_percentile: float = 95.0

    # dataloader
    batch_size: int = 32

    # model
    embed_dim: int = 64
    num_filters: int = 64
    fc_dim: int = 128
    dropout: float = 0.5
    kernel_sizes: Tuple[int, int, int] = (3, 5, 7)

    # training
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 10
    grad_clip_norm: float = 5.0

    # class imbalance
    use_class_weights: bool = True

    # early stopping
    early_stopping: bool = True
    early_stop_metric: str = "f1_macro"  # one of: f1_macro, f1_weighted, accuracy
    patience: int = 3                    # stop after this many non-improving evals
    min_delta: float = 0.0               # minimum improvement to count
    eval_every: int = 1                  # evaluate on val every N epochs

    # misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch to reduce run-to-run variation."""

    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---- Encoding utilities ----
def _build_char_vocab(texts: List[str]) -> Dict[str, int]:
    """Build a character vocabulary from training texts only.

    Index 0 is reserved for padding and 1 for unseen characters at inference
    time, so observed characters start at 2.
    """

    chars = sorted({ch for t in texts for ch in t})
    return {ch: i + 2 for i, ch in enumerate(chars)}


def _encode(text: str, char2idx: Dict[str, int]) -> List[int]:
    """Map a string to character ids, using 1 for out-of-vocabulary symbols."""

    return [char2idx.get(ch, 1) for ch in text]


def _choose_max_len(encoded_train: List[List[int]], *, percentile: float, cap: int) -> int:
    """Choose a training-derived sequence length for padding and truncation.

    The percentile-based heuristic keeps most examples intact while limiting
    memory use; the result is capped and never smaller than one.
    """

    lengths = np.array([len(x) for x in encoded_train], dtype=np.int64)
    if len(lengths) == 0:
        return 1
    L = int(np.percentile(lengths, percentile))
    return max(1, min(L, cap))


def _pad(seq: List[int], max_len: int) -> List[int]:
    """Pad or truncate a sequence to the fixed model input length."""

    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [0] * (max_len - len(seq))


def _encode_pad_many(texts: List[str], char2idx: Dict[str, int], max_len: int) -> np.ndarray:
    """Encode a collection of texts and return a padded integer array."""

    enc = [_encode(t, char2idx) for t in texts]
    padded = [_pad(s, max_len) for s in enc]
    return np.array(padded, dtype=np.int64)


# ---- Training and prediction ----
@torch.no_grad()
def _predict(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    *,
    desc: str = "predict",
    show_progress: bool = True,
) -> np.ndarray:
    """Run batched inference and return predicted class ids."""

    model.eval()
    preds = []
    for xb, _yb in tqdm(loader, desc=desc, leave=False, disable=not show_progress):
        xb = xb.to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=-1)
        preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.array([], dtype=np.int64)


def run_one(
    examples: List[Example],
    split: Split,
    cfg: CNNConfig,
    *,
    verbose: bool = True,
    compute_test: bool = True,
) -> Dict[str, EvalResult]:
    """Train one CNN configuration on a fixed split and evaluate it.

    Vocabulary construction and sequence-length selection use only training
    data to avoid leakage. Returns validation metrics and, optionally, test
    metrics for the same trained model.
    """
    _set_seed(cfg.seed)

    # ---- Prepare labels and split-specific views ----
    X_all = np.array([ex.text for ex in examples], dtype=object)
    y_str_all = np.array([ex.label for ex in examples], dtype=object)

    labels_sorted = sorted(set(y_str_all.tolist()))
    label2id = {lab: i for i, lab in enumerate(labels_sorted)}
    id2label = {i: lab for lab, i in label2id.items()}
    y_ids_all = np.array([label2id[v] for v in y_str_all], dtype=np.int64)

    X_train = X_all[split.train_idx].tolist()
    y_train = y_ids_all[split.train_idx]
    X_val = X_all[split.val_idx].tolist()
    y_val = y_ids_all[split.val_idx]
    X_test = X_all[split.test_idx].tolist()
    y_test = y_ids_all[split.test_idx]

    if verbose:
        print(
            f"\n[CNN] Building vocab/max_len on train only "
            f"(train={len(X_train)}, val={len(X_val)}, test={len(X_test)})"
        )

    # ---- Encode with train-only statistics ----
    # Restricting the vocabulary and max length to the training partition keeps
    # the validation and test evaluations free of representation leakage.
    char2idx = _build_char_vocab(X_train)
    vocab_size = len(char2idx) + 2

    encoded_train = [_encode(t, char2idx) for t in X_train]
    max_len = _choose_max_len(encoded_train, percentile=cfg.len_percentile, cap=cfg.max_len_cap)

    if verbose:
        print(f"[CNN] vocab_size={vocab_size} | max_len={max_len} (p{cfg.len_percentile}, cap={cfg.max_len_cap})")

    Xtr = _encode_pad_many(X_train, char2idx, max_len)
    Xva = _encode_pad_many(X_val, char2idx, max_len)
    Xte = _encode_pad_many(X_test, char2idx, max_len)

    train_ds = EncodedLogDataset(Xtr, y_train)
    val_ds = EncodedLogDataset(Xva, y_val)
    test_ds = EncodedLogDataset(Xte, y_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # ---- Build model and optimization setup ----
    model = MultiKernelCharCNN(
        vocab_size=vocab_size,
        num_classes=len(labels_sorted),
        embed_dim=cfg.embed_dim,
        num_filters=cfg.num_filters,
        fc_dim=cfg.fc_dim,
        dropout=cfg.dropout,
        kernel_sizes=cfg.kernel_sizes,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Inverse-frequency weighting helps when label counts are imbalanced.
    if cfg.use_class_weights:
        counts = np.bincount(y_train, minlength=len(labels_sorted)).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        inv = counts.sum() / counts
        class_weights = torch.tensor(inv, dtype=torch.float32, device=cfg.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Metrics are computed in label-string space because the shared evaluation
    # helper expects the original class names.
    y_val_true_str = np.array([id2label[i] for i in y_val], dtype=object)

    # ---- Early stopping state ----
    use_es = bool(cfg.early_stopping)
    if use_es and cfg.early_stop_metric not in {"f1_macro", "f1_weighted", "accuracy"}:
        raise ValueError("CNNConfig.early_stop_metric must be one of: f1_macro, f1_weighted, accuracy")

    best_score = -float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad = 0

    # ---- Model training ----
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_bar = tqdm(train_loader, desc=f"[CNN] epoch {epoch}/{cfg.epochs}", leave=False, disable=not verbose)
        running = 0.0
        seen = 0

        for xb, yb in epoch_bar:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

            optimizer.step()

            bs = int(xb.size(0))
            running += float(loss.item()) * bs
            seen += bs
            if verbose:
                epoch_bar.set_postfix(loss=f"{(running / max(1, seen)):.4f}")

        if verbose:
            print(f"[CNN] epoch {epoch}/{cfg.epochs} done | avg_loss={(running / max(1, seen)):.4f}")

        # Validation is checked only at the configured cadence to keep training
        # cost predictable during larger searches.
        if use_es and (epoch % max(1, cfg.eval_every) == 0):
            y_val_pred_ids = _predict(
                model, val_loader, cfg.device, desc=f"[CNN] val @ epoch {epoch}", show_progress=False
            )
            y_val_pred_str = np.array([id2label[i] for i in y_val_pred_ids], dtype=object)
            val_res = evaluate_classifier(y_val_true_str, y_val_pred_str, labels=labels_sorted)
            val_score = float(getattr(val_res, cfg.early_stop_metric))

            improved = val_score > (best_score + float(cfg.min_delta))
            if improved:
                best_score = val_score
                best_state = copy.deepcopy(model.state_dict())
                bad = 0
                if verbose:
                    print(f"[CNN] early-stop: new best {cfg.early_stop_metric}={best_score:.4f} at epoch {epoch}")
            else:
                bad += 1
                if verbose:
                    print(f"[CNN] early-stop: no improvement ({bad}/{cfg.patience})")

                if bad >= int(cfg.patience):
                    if verbose:
                        print(f"[CNN] early-stop: stopping at epoch {epoch} (best={best_score:.4f})")
                    break

    # Restore the best validation checkpoint rather than the last epoch.
    if use_es and best_state is not None:
        model.load_state_dict(best_state)

    # ---- Final evaluation ----
    y_val_pred = _predict(model, val_loader, cfg.device, desc="[CNN] predict val", show_progress=verbose)

    y_val_pred_str = np.array([id2label[i] for i in y_val_pred], dtype=object)

    out = {
        "val": evaluate_classifier(y_val_true_str, y_val_pred_str, labels=labels_sorted),
    }

    # Test evaluation is optional so hyperparameter search can avoid repeated
    # access to the held-out split.
    if compute_test:
        y_test_pred = _predict(model, test_loader, cfg.device, desc="[CNN] predict test", show_progress=verbose)

        y_test_true_str = np.array([id2label[i] for i in y_test], dtype=object)
        y_test_pred_str = np.array([id2label[i] for i in y_test_pred], dtype=object)

        out["test"] = evaluate_classifier(
            y_test_true_str,
            y_test_pred_str,
            labels=labels_sorted,
        )

    return out


# ---- Hyperparameter search ----
@dataclass(frozen=True)
class Candidate:
    """Wrapper for one hyperparameter configuration considered in search."""

    cfg: CNNConfig


def search(
    examples: List[Example],
    split: Split,
    candidates: Iterable[Candidate],
    *,
    metric: str = "f1_macro",
    evaluate_test_for_all: bool = False,
    verbose: bool = True,
) -> Tuple[Candidate, EvalResult, EvalResult, List[Tuple[Candidate, EvalResult]]]:
    """Select the best CNN configuration from a candidate set.

    Model selection is based on validation performance only. The test split is
    evaluated once for the selected configuration unless explicitly requested
    for every candidate.
    """

    if metric not in {"f1_macro", "f1_weighted", "accuracy"}:
        raise ValueError("metric must be one of: f1_macro, f1_weighted, accuracy")

    def score(res: EvalResult) -> float:
        return getattr(res, metric)

    candidates = list(candidates)
    if verbose:
        print(f"\n[CNN] Starting search over {len(candidates)} candidates...\n")

    best: Optional[Candidate] = None
    best_val: Optional[EvalResult] = None
    best_test: Optional[EvalResult] = None
    all_val: List[Tuple[Candidate, EvalResult]] = []

    pbar = tqdm(candidates, desc="[CNN] candidates", disable=not verbose)

    for cand in pbar:
        pbar.set_postfix(
            epochs=cand.cfg.epochs,
            bs=cand.cfg.batch_size,
            emb=cand.cfg.embed_dim,
            filt=cand.cfg.num_filters,
            lr=f"{cand.cfg.lr:g}",
        )

        out = run_one(
            examples,
            split,
            cand.cfg,
            verbose=verbose,
            # Avoid using the held-out test split during model selection.
            compute_test=False,
        )
        val_res = out["val"]
        all_val.append((cand, val_res))

        if best_val is None or score(val_res) > score(best_val):
            best = cand
            best_val = val_res
            best_test = out["test"] if evaluate_test_for_all else None

            if verbose:
                print(f"[CNN] new best {metric}={score(best_val):.4f}")

    assert best is not None and best_val is not None

    if best_test is None:
        if verbose:
            print("\n[CNN] Evaluating best candidate on TEST set...\n")
        best_test = run_one(examples, split, best.cfg, verbose=verbose)["test"]

    if verbose:
        print("\n[CNN] Search complete.\n")

    return best, best_val, best_test, all_val
