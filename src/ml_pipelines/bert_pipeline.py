"""BERT-style text classification pipeline with validation-based model selection.

The key design choice is to rank candidate configurations on the validation split
and evaluate the test split only once for the selected model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable, Tuple, Any

import copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from tqdm.auto import tqdm

from src.core.ml.data import Example
from src.core.ml.splits import Split
from src.core.ml.eval import EvalResult, evaluate_classifier


class TextDataset(Dataset):
    """Minimal dataset wrapper for tokenized text batches and label tensors."""

    def __init__(self, encodings: Dict[str, torch.Tensor], labels: torch.Tensor):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


@dataclass(frozen=True)
class TransformerConfig:
    """Training and evaluation settings for a transformer classifier.

    Includes optimization, early-stopping, and device configuration used across
    both single-run training and candidate search.
    """

    model_name: str = "roberta-base"
    max_length: int = 128
    batch_size: int = 16
    lr: float = 2e-5
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_clip_norm: float = 1.0  # Stabilizes fine-tuning on small or noisy splits.

    # ---- Early stopping ----
    early_stopping: bool = True
    early_stop_metric: str = "f1_macro"  # One of: f1_macro, f1_weighted, accuracy.
    patience: int = 1
    min_delta: float = 0.0
    eval_every: int = 2  # Evaluate on validation every N epochs.


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible training runs."""

    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _encode_texts(
    tokenizer,
    texts: List[str],
    max_length: int,
    *,
    desc: str,
    verbose: bool
) -> Dict[str, torch.Tensor]:
    """Tokenize raw texts into padded tensors compatible with the model.

    Uses chunked encoding only when progress reporting is enabled to keep large
    tokenization jobs transparent without changing the resulting tensors.
    """

    if not verbose:
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    # Chunking is only for progress visibility on large corpora.
    chunk = 2048
    enc_parts: Dict[str, List[torch.Tensor]] = {}
    for i in tqdm(range(0, len(texts), chunk), desc=desc, leave=False):
        batch = texts[i:i + chunk]
        out = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        for k, v in out.items():
            enc_parts.setdefault(k, []).append(v)

    return {k: torch.cat(vs, dim=0) for k, vs in enc_parts.items()}


@torch.no_grad()
def _predict(
    model,
    dataloader: DataLoader,
    device: str,
    *,
    desc: str = "predict",
    verbose: bool = True
) -> np.ndarray:
    """Run batched inference and return predicted class ids as a NumPy array."""

    model.eval()
    preds: List[np.ndarray] = []
    it = dataloader if not verbose else tqdm(dataloader, desc=desc, leave=False)
    for batch in it:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch.pop("labels", None)
        outputs = model(**batch)
        pred = torch.argmax(outputs.logits, dim=-1)
        preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.array([], dtype=np.int64)


def _evaluate_on_split(
    *,
    model,
    tokenizer,
    X_text: List[str],
    y_ids: np.ndarray,
    cfg: TransformerConfig,
    id2label: Dict[int, str],
    labels_sorted: List[str],
    desc_prefix: str,
    verbose: bool,
) -> EvalResult:
    """Evaluate a trained model on a labeled split and return classifier metrics.

    Labels are mapped back to their original string representation before metric
    computation so the evaluation stays aligned with the global label space.
    """

    enc = _encode_texts(tokenizer, X_text, cfg.max_length, desc=f"{desc_prefix} tokenize", verbose=verbose)
    ds = TextDataset(enc, torch.tensor(y_ids))

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=cfg.device.startswith("cuda"),
    )

    y_pred = _predict(model, loader, cfg.device, desc=f"{desc_prefix} predict", verbose=verbose)

    y_true_str = np.array([id2label[i] for i in y_ids], dtype=object)
    y_pred_str = np.array([id2label[i] for i in y_pred], dtype=object)
    return evaluate_classifier(y_true_str, y_pred_str, labels=labels_sorted)


def run_one(
    examples: List[Example],
    split: Split,
    cfg: TransformerConfig,
    *,
    verbose: bool = True,
    compute_test: bool = False,        # (NEW) Option A: don’t touch test during search
    return_state: bool = False,        # (NEW) let search keep weights to avoid retraining
) -> Dict[str, Any]:
    """Train one transformer configuration and evaluate it on the validation split.

    Test evaluation is optional because search is designed to avoid touching the
    test split until model selection is complete. Returns validation metrics and,
    when requested, test metrics plus the trained state needed for reuse.
    """
    _set_seed(cfg.seed)

    # ---- Prepare labels and fixed split views ----
    X = np.array([ex.text for ex in examples], dtype=object)
    y_str = np.array([ex.label for ex in examples], dtype=object)

    labels_sorted = sorted(set(y_str.tolist()))
    label2id = {lab: i for i, lab in enumerate(labels_sorted)}
    id2label = {i: lab for lab, i in label2id.items()}
    y = np.array([label2id[v] for v in y_str], dtype=np.int64)

    X_train, y_train = X[split.train_idx].tolist(), y[split.train_idx]
    X_val, y_val = X[split.val_idx].tolist(), y[split.val_idx]
    X_test, y_test = X[split.test_idx].tolist(), y[split.test_idx]

    if verbose:
        print(
            f"\n[BERT] model={cfg.model_name} | "
            f"train={len(X_train)} val={len(X_val)} test={len(X_test)} | "
            f"epochs={cfg.epochs} bs={cfg.batch_size} max_len={cfg.max_length} lr={cfg.lr:g}"
        )
        print(f"[BERT] device={cfg.device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # ---- Build datasets and loaders ----
    # Test tokenization stays separate so search does not leak work onto test data.
    train_enc = _encode_texts(tokenizer, X_train, cfg.max_length, desc="[BERT] tokenize train", verbose=verbose)
    val_enc = _encode_texts(tokenizer, X_val, cfg.max_length, desc="[BERT] tokenize val", verbose=verbose)

    train_ds = TextDataset(train_enc, torch.tensor(y_train))
    val_ds = TextDataset(val_enc, torch.tensor(y_val))

    pin = cfg.device.startswith("cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=pin,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(labels_sorted),
        id2label=id2label,
        label2id=label2id,
    ).to(cfg.device)

    # Optional encoder freezing can make broad searches cheaper, but is disabled here.
    '''
    for p in model.base_model.parameters():
        p.requires_grad = False
    '''
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = cfg.epochs * max(1, len(train_loader))
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ---- Loss construction ----
    # We derive class weights from the training split only to avoid leaking label
    # frequencies from validation or test into the optimization objective.
    num_classes = len(labels_sorted)
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, 1.0, None)  # Guards against degenerate empty counts.

    # Inverse-frequency weighting keeps minority classes visible during fine-tuning.
    class_weights = counts.sum() / counts

    # Optional safeguard if a very small class would dominate the loss.
    # class_weights = np.minimum(class_weights, 10.0)

    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=cfg.device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_t)

    # ---- Training with periodic validation checks ----
    use_es = bool(cfg.early_stopping)
    if use_es and cfg.early_stop_metric not in {"f1_macro", "f1_weighted", "accuracy"}:
        raise ValueError("TransformerConfig.early_stop_metric must be one of: f1_macro, f1_weighted, accuracy")

    y_val_true_str = np.array([id2label[i] for i in y_val], dtype=object)

    best_score = -float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad = 0

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_bar = tqdm(train_loader, desc=f"[BERT] epoch {epoch}/{cfg.epochs}", leave=False, disable=not verbose)
        running = 0.0
        seen = 0

        for batch in epoch_bar:
            batch = {k: v.to(cfg.device) for k, v in batch.items()}

            # Loss is computed explicitly so the weighted criterion is always used.
            labels = batch.pop("labels")
            outputs = model(**batch)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            loss.backward()

            if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            bs = int(labels.size(0))
            running += float(loss.item()) * bs
            seen += bs
            global_step += 1

            if verbose:
                epoch_bar.set_postfix(
                    loss=f"{(running / max(1, seen)):.4f}",
                    step=global_step,
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

        if verbose:
            print(f"[BERT] epoch {epoch}/{cfg.epochs} done | avg_loss={(running / max(1, seen)):.4f}")

        # Validation is intentionally periodic to reduce search cost on longer runs.
        if use_es and (epoch % max(1, cfg.eval_every) == 0):
            y_val_pred = _predict(model, val_loader, cfg.device, desc=f"[BERT] val @ epoch {epoch}", verbose=False)
            y_val_pred_str = np.array([id2label[i] for i in y_val_pred], dtype=object)
            val_res_epoch = evaluate_classifier(y_val_true_str, y_val_pred_str, labels=labels_sorted)
            val_score = float(getattr(val_res_epoch, cfg.early_stop_metric))

            if val_score > (best_score + float(cfg.min_delta)):
                best_score = val_score
                best_state = copy.deepcopy(model.state_dict())
                bad = 0
                if verbose:
                    print(f"[BERT] early-stop: new best {cfg.early_stop_metric}={best_score:.4f} at epoch {epoch}")
            else:
                bad += 1
                if verbose:
                    print(f"[BERT] early-stop: no improvement ({bad}/{cfg.patience})")
                if bad >= int(cfg.patience):
                    if verbose:
                        print(f"[BERT] early-stop: stopping at epoch {epoch} (best={best_score:.4f})")
                    break

    # Restore the best validation checkpoint before the final reported metrics.
    if use_es and best_state is not None:
        model.load_state_dict(best_state)

    # ---- Final evaluation and optional artifacts ----
    y_val_pred = _predict(model, val_loader, cfg.device, desc="[BERT] predict val", verbose=verbose)
    y_val_pred_str = np.array([id2label[i] for i in y_val_pred], dtype=object)
    val_res = evaluate_classifier(y_val_true_str, y_val_pred_str, labels=labels_sorted)

    out: Dict[str, Any] = {"val": val_res}

    # Test evaluation stays opt-in so search can remain validation-only.
    if compute_test:
        test_res = _evaluate_on_split(
            model=model,
            tokenizer=tokenizer,
            X_text=X_test,
            y_ids=y_test,
            cfg=cfg,
            id2label=id2label,
            labels_sorted=labels_sorted,
            desc_prefix="[BERT] test",
            verbose=verbose,
        )
        out["test"] = test_res

    # Returning the trained state lets search reuse the winning model directly.
    if return_state:
        out["state_dict"] = copy.deepcopy(model.state_dict())
        out["meta"] = {
            "labels_sorted": labels_sorted,
            "label2id": label2id,
            "id2label": id2label,
        }

    return out


# ---- Hyperparameter search ----
@dataclass(frozen=True)
class Candidate:
    """Container for one transformer configuration considered during search."""

    cfg: TransformerConfig


def search(
    examples: List[Example],
    split: Split,
    candidates: Iterable[Candidate],
    *,
    metric: str = "f1_macro",
    evaluate_test_for_all: bool = False,  # kept for API compatibility; ignored in Option A flow
    verbose: bool = True,
) -> Tuple[Candidate, EvalResult, EvalResult, List[Tuple[Candidate, EvalResult]]]:
    """Select the best candidate on validation and test it exactly once.

    The search loop never evaluates test metrics for intermediate candidates.
    Instead, it keeps the winning model state and reuses it for a single final
    test evaluation, matching the intended null-vs-true evaluation discipline.
    """

    if metric not in {"f1_macro", "f1_weighted", "accuracy"}:
        raise ValueError("metric must be one of: f1_macro, f1_weighted, accuracy")

    def score(res: EvalResult) -> float:
        return float(getattr(res, metric))

    candidates = list(candidates)
    if verbose:
        print(f"\n[BERT] Starting search over {len(candidates)} candidates...\n")

    # Keep the winning weights so the selected model can be tested without retraining.
    best: Optional[Candidate] = None
    best_val: Optional[EvalResult] = None
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_meta: Optional[Dict[str, Any]] = None

    all_val: List[Tuple[Candidate, EvalResult]] = []

    pbar = tqdm(candidates, desc="[BERT] candidates", disable=not verbose)

    for cand in pbar:
        pbar.set_postfix(
            model=cand.cfg.model_name.split("/")[-1],
            ep=cand.cfg.epochs,
            bs=cand.cfg.batch_size,
            lr=f"{cand.cfg.lr:g}",
            maxlen=cand.cfg.max_length,
        )

        # Candidate ranking is validation-only by design.
        out = run_one(
            examples,
            split,
            cand.cfg,
            verbose=verbose,
            compute_test=False,
            return_state=True,
        )

        val_res: EvalResult = out["val"]
        all_val.append((cand, val_res))

        if best_val is None or score(val_res) > score(best_val):
            best = cand
            best_val = val_res
            best_state = out["state_dict"]
            best_meta = out["meta"]
            if verbose:
                print(f"[BERT] new best {metric}={score(best_val):.4f}")

    assert best is not None and best_val is not None and best_state is not None and best_meta is not None

    # ---- Final test evaluation ----
    # Test is touched exactly once after selection to preserve a clean estimate.
    if verbose:
        print("\n[BERT] Evaluating best candidate on TEST set (reusing trained weights)...\n")

    # Reuse the exact label mapping from training so ids remain consistent.
    labels_sorted: List[str] = best_meta["labels_sorted"]
    id2label: Dict[int, str] = best_meta["id2label"]
    label2id: Dict[str, int] = best_meta["label2id"]

    # Build the held-out test split using the saved label mapping.
    X = np.array([ex.text for ex in examples], dtype=object)
    y_str = np.array([ex.label for ex in examples], dtype=object)
    y_ids = np.array([label2id[v] for v in y_str], dtype=np.int64)

    X_test = X[split.test_idx].tolist()
    y_test = y_ids[split.test_idx]

    tokenizer = AutoTokenizer.from_pretrained(best.cfg.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        best.cfg.model_name,
        num_labels=len(labels_sorted),
        id2label=id2label,
        label2id=label2id,
    ).to(best.cfg.device)

    model.load_state_dict(best_state)

    best_test = _evaluate_on_split(
        model=model,
        tokenizer=tokenizer,
        X_text=X_test,
        y_ids=y_test,
        cfg=best.cfg,
        id2label=id2label,
        labels_sorted=labels_sorted,
        desc_prefix="[BERT] test(best)",
        verbose=verbose,
    )

    if verbose:
        print("\n[BERT] Search complete.\n")

    # Keep the public return signature unchanged.
    return best, best_val, best_test, all_val
