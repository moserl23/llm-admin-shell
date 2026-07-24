"""Hybrid RAG-style classification pipeline for bundled text examples.

The pipeline builds class-specific retrieval indices over training bundles,
uses embedding similarity as a fast decision rule, and falls back to an LLM
only when retrieval evidence is weak or incomplete.
"""

from __future__ import annotations

import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable, Tuple, Any, Set

import json

from transformers.utils import logging
logging.set_verbosity_error()
import time

import numpy as np
from tqdm.auto import tqdm

from src.core.ml.data import Example
from src.core.ml.env import load_project_env
from src.core.ml.splits import Split
from src.core.ml.eval import EvalResult, evaluate_classifier

from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_project_env()

# Optional FAISS
try:
    import faiss  # type: ignore
    _FAISS_OK = True
except Exception:
    faiss = None  # type: ignore
    _FAISS_OK = False


# ---- Bundling ----
def _normalize_bundle(text: str) -> str:
    """Normalize line endings and trim trailing whitespace within a bundle.

    This keeps bundle text stable across platforms and reduces formatting
    variation before embedding or prompting.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def _bundle_texts(
    texts: List[str],
    *,
    bundle_size: int,
    stride: int,
    strategy: str,
    drop_last: bool,
) -> List[str]:
    """Group sequential texts into fixed or sliding bundles.

    Bundles are formed within each class split rather than across labels, so
    each bundle preserves a single ground-truth class.
    """
    if bundle_size <= 0:
        raise ValueError("bundle_size must be > 0")

    n = len(texts)
    if n == 0:
        return []

    bundles: List[str] = []

    if strategy == "fixed":
        for start in range(0, n, bundle_size):
            end = start + bundle_size
            if end > n and drop_last:
                break
            chunk = texts[start:min(end, n)]
            b = _normalize_bundle("\n".join(chunk))
            if b:
                bundles.append(b)

    elif strategy == "sliding":
        stride = max(1, stride)
        if bundle_size > n:
            if drop_last:
                return []
            b = _normalize_bundle("\n".join(texts))
            return [b] if b else []

        for start in range(0, n - bundle_size + 1, stride):
            end = start + bundle_size
            chunk = texts[start:end]
            b = _normalize_bundle("\n".join(chunk))
            if b:
                bundles.append(b)

    else:
        raise ValueError(f"Unknown strategy='{strategy}', expected 'fixed' or 'sliding'.")

    return bundles


# ---- Similarity helpers ----
def _dot_sims(query: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Return similarity scores between one query vector and many candidates.

    Both inputs are assumed to be L2-normalized, so the dot product is the
    cosine similarity used for retrieval and fast-path scoring.
    """
    q = query.astype(np.float32, copy=False)
    m = mat.astype(np.float32, copy=False)
    return (m @ q).astype(np.float32)


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """L2-normalize a 2D embedding matrix row-wise."""
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def _l2_normalize_vec(x: np.ndarray) -> np.ndarray:
    """L2-normalize a single embedding vector."""
    x = x.astype(np.float32, copy=False)
    return x / (np.linalg.norm(x) + 1e-12)


# ---- In-memory retrieval index ----
class _InMemoryPerClassIndex:
    """Store per-class embeddings and bundle texts for nearest-neighbor lookup.

    Retrieval is performed independently within each class so the prompt and
    similarity comparison remain class-balanced by construction.
    """

    def __init__(self, *, backend: str = "numpy", faiss_hnsw_m: int = 32):
        self._emb: Dict[str, np.ndarray] = {}
        self._txt: Dict[str, List[str]] = {}

        self._backend = backend
        self._faiss_hnsw_m = int(faiss_hnsw_m)

        # per-class faiss indices (if used)
        self._faiss_index: Dict[str, Any] = {}

    @property
    def backend(self) -> str:
        """Return the active retrieval backend."""
        return self._backend

    def add(self, label: str, emb: np.ndarray, txt: List[str]) -> None:
        """Register normalized embeddings and aligned bundle texts for a label.

        When FAISS is requested and available, the corresponding per-class
        index is built immediately from these embeddings.
        """
        if len(txt) != emb.shape[0]:
            raise ValueError("emb and txt must align")

        # Cosine retrieval is implemented as an inner product, so stored rows
        # must be normalized once here rather than on every lookup.
        emb = emb.astype(np.float32, copy=False)
        emb = _l2_normalize_rows(emb)

        self._emb[label] = emb
        self._txt[label] = list(txt)

        # FAISS remains optional; a missing dependency should not break the run.
        if self._backend == "faiss":
            if not _FAISS_OK:
                # Fall back to exact NumPy retrieval when ANN support is absent.
                self._backend = "numpy"
                return

            d = emb.shape[1]
            # Inner-product search is equivalent to cosine after normalization.
            index = faiss.IndexHNSWFlat(d, self._faiss_hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 80
            index.add(emb)
            self._faiss_index[label] = index

    def topk(self, label: str, query_emb: np.ndarray, k: int) -> List[Tuple[float, str]]:
        """Return the top-k retrieved bundles for one class label.

        Results are scored by cosine similarity and include both the score and
        the original bundle text used later for prompting.
        """
        if k <= 0:
            return []
        if label not in self._emb or self._emb[label].shape[0] == 0:
            return []

        q = _l2_normalize_vec(query_emb.astype(np.float32, copy=False))

        if self._backend == "faiss" and _FAISS_OK and label in self._faiss_index:
            index = self._faiss_index[label]
            k_eff = min(k, self._emb[label].shape[0])
            # FAISS expects a batch dimension even for a single query.
            D, I = index.search(q.reshape(1, -1), k_eff)
            sims = D[0]
            idxs = I[0]
            out: List[Tuple[float, str]] = []
            for sim, i in zip(sims, idxs):
                if i < 0:
                    continue
                out.append((float(sim), self._txt[label][int(i)]))
            return out

        # NumPy provides an exact fallback when FAISS is disabled or unavailable.
        sims = _dot_sims(q, self._emb[label])
        k_eff = min(k, sims.shape[0])
        idx = np.argpartition(-sims, k_eff - 1)[:k_eff]
        idx = idx[np.argsort(-sims[idx])]
        return [(float(sims[i]), self._txt[label][int(i)]) for i in idx]


# ---- Config ----
@dataclass(frozen=True)
class RAGLLMConfig:
    """Configuration for bundling, retrieval, and optional LLM fallback."""
    # bundling hyperparams
    bundle_size: int = 50
    bundle_strategy: str = "fixed"   # fixed | sliding
    sliding_stride: int = 25         # only used if sliding
    drop_last_incomplete: bool = True

    # retrieval hyperparams
    per_class_k: int = 5
    max_chars_per_retrieved: int = 1400  # prompt budget control

    # retrieval backend
    retrieval_backend: str = "numpy"     # "numpy" or "faiss"
    faiss_hnsw_m: int = 32              # only used if retrieval_backend="faiss"

    # OPEN-SOURCE embedding (local, free)
    local_embedding_model: str = "BAAI/bge-base-en-v1.5"
    local_embedding_batch_size: int = 32
    local_embedding_device: str = "cuda"  # "cuda" or "cpu"
    local_normalize_embeddings: bool = True

    # Prediction embedding batching
    predict_embedding_batch_size: int = 64  # NEW: batch size for embedding VAL/TEST

    # OpenAI chat classification (fallback)
    chat_model: str = "gpt-4.1-mini"
    temperature: float = 0.0
    max_output_tokens: int = 30
    timeout_s: float = 60.0
    max_retries: int = 2
    retry_backoff_s: float = 1.5

    # Hybrid gating to reduce LLM calls
    use_llm_fallback: bool = True
    llm_uncertainty_margin: float = 0.08  # call LLM if |score| < margin
    score_agg: str = "mean"               # "mean" or "median"

    # misc
    seed: int = 42


# ---- OpenAI fallback helpers ----
def _truncate(s: str, max_chars: int) -> str:
    """Clip long text blocks to keep retrieval context within prompt budget."""
    if max_chars <= 0:
        return ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n…(truncated)…"


def _build_messages(
    query_bundle: str,
    *,
    label_a: str,
    label_b: str,
    retrieved_a: List[Tuple[float, str]],
    retrieved_b: List[Tuple[float, str]],
    max_chars_per_retrieved: int,
) -> Tuple[str, str]:
    """Build the system and user prompts for fallback classification.

    The prompt presents retrieved examples from both classes explicitly so the
    model compares balanced evidence rather than relying on prior frequency.
    """
    sys = (
        "You are a strict binary classifier for log bundles.\n"
        "Return ONLY valid JSON with exactly one key: \"label\".\n"
        f"Valid labels are: \"{label_a}\" and \"{label_b}\".\n"
        "No extra keys. No commentary. No markdown."
    )

    def fmt_block(name: str, items: List[Tuple[float, str]]) -> str:
        """Format one class-specific retrieval block for the prompt."""
        lines = [f"Class {name} retrieved examples:"]
        for i, (sim, txt) in enumerate(items, 1):
            lines.append(f"[{name} ex {i}] sim={sim:.4f}\n{_truncate(txt, max_chars_per_retrieved)}\n")
        return "\n".join(lines)

    user = (
        f"Task: classify the QUERY bundle as either {label_a} or {label_b}.\n"
        "Use the retrieved labeled examples as reference.\n\n"
        f"{fmt_block(label_a, retrieved_a)}\n\n"
        f"{fmt_block(label_b, retrieved_b)}\n\n"
        "QUERY bundle:\n"
        f"{_truncate(query_bundle, 9000)}\n\n"
        f"Output JSON only, like: {{\"label\":\"{label_a}\"}}"
    )
    return sys, user


def _parse_label(raw: str, *, valid_labels: Set[str]) -> str:
    """Parse and validate the fallback model response.

    Only a single JSON field named ``label`` is accepted to keep the output
    contract stable during retries and evaluation.
    """
    raw = (raw or "").strip()
    obj = json.loads(raw)
    if not isinstance(obj, dict) or set(obj.keys()) != {"label"}:
        raise ValueError(f"Expected JSON with exactly {{'label'}}, got: {raw[:200]}")
    lab = obj["label"]
    if lab not in valid_labels:
        raise ValueError(f"Invalid label '{lab}', expected one of {sorted(valid_labels)}")
    return str(lab)


def _chat_classify(
    client: OpenAI,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    max_retries: int,
    retry_backoff_s: float,
    system_msg: str,
    user_msg: str,
    valid_labels: Set[str],
) -> str:
    """Classify one bundle with the chat model using bounded retries.

    Retries are reserved for transient API failures; invalid outputs still
    surface as exceptions after parsing and validation.
    """
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                timeout=timeout_s,
            )
            content = resp.choices[0].message.content
            return _parse_label(content, valid_labels=valid_labels)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(retry_backoff_s * (attempt + 1))
                continue
            raise last_err


def _aggregate_sims(sims: np.ndarray, *, agg: str) -> float:
    """Aggregate retrieved similarities into a single class score."""
    if sims.size == 0:
        return float("nan")
    if agg == "mean":
        return float(np.mean(sims))
    if agg == "median":
        return float(np.median(sims))
    raise ValueError(f"Unknown score_agg='{agg}', expected 'mean' or 'median'.")


# ---- Local embedding helpers ----
def _load_local_embedder(cfg: RAGLLMConfig) -> SentenceTransformer:
    """Load the local sentence-transformer used for retrieval embeddings."""
    return SentenceTransformer(cfg.local_embedding_model, device=cfg.local_embedding_device)


def _embed_texts_local(
    embedder: SentenceTransformer,
    texts: List[str],
    *,
    batch_size: int,
    normalize: bool,
    desc: str = "embed",
    verbose: bool = True,
) -> np.ndarray:
    """Embed texts in batches and return a float32 matrix.

    Normalization is applied consistently so retrieval and fast-path scoring
    can use cosine similarity via dot products.
    """
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    bs = max(1, int(batch_size))
    out_chunks: List[np.ndarray] = []

    rng = range(0, len(texts), bs)
    it = rng if not verbose else tqdm(rng, desc=desc, leave=False)
    for i in it:
        chunk = texts[i:i + bs]
        emb = embedder.encode(
            chunk,
            batch_size=len(chunk),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        out_chunks.append(emb.astype(np.float32, copy=False))

    out = np.vstack(out_chunks) if out_chunks else np.zeros((0, 1), dtype=np.float32)

    # Keep normalization explicit because downstream retrieval assumes it.
    if normalize:
        out = _l2_normalize_rows(out)
    return out


# ---- Single-run evaluation ----
def run_one(
    examples: List[Example],
    split: Split,
    cfg: RAGLLMConfig,
    evaluate_test: bool = True,
    *,
    verbose: bool = True,
) -> Dict[str, EvalResult]:
    """Run one configuration on a fixed train/val/test split.

    Training bundles are indexed by class, validation and test bundles are
    predicted bundle-wise, and metrics are returned for each evaluated split.
    """
    if verbose:
        print("\n" + "=" * 80)
        print("[LLM] RUN_ONE START")
        print(f"[LLM] Config: {cfg}")
        print(f"[LLM] evaluate_test: {evaluate_test}")
        print("=" * 80)

    llm_client: Optional[OpenAI] = None
    if cfg.use_llm_fallback:
        llm_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    if verbose:
        print("\n[LLM] Loading local embedding model...")
    embedder = _load_local_embedder(cfg)
    if verbose:
        print(f"[LLM] Local embedder: {cfg.local_embedding_model} on {cfg.local_embedding_device}")
        print(f"[LLM] Retrieval backend: {cfg.retrieval_backend} (faiss_available={_FAISS_OK})")

    X_all = np.array([ex.text for ex in examples], dtype=object)
    y_all = np.array([ex.label for ex in examples], dtype=object)
    labels_sorted = sorted(set(map(str, y_all.tolist())))

    if verbose:
        print(f"[LLM] Total examples: {len(X_all)}")
        print(f"[LLM] Labels present: {labels_sorted}")

    if len(labels_sorted) != 2:
        raise ValueError(f"Expected exactly 2 labels, got {labels_sorted}")

    label_a, label_b = labels_sorted[0], labels_sorted[1]
    valid_labels = {label_a, label_b}

    X_train = X_all[split.train_idx].tolist()
    y_train = [str(x) for x in y_all[split.train_idx].tolist()]

    X_val = X_all[split.val_idx].tolist()
    y_val = [str(x) for x in y_all[split.val_idx].tolist()]

    X_test = X_all[split.test_idx].tolist()
    y_test = [str(x) for x in y_all[split.test_idx].tolist()]

    if verbose:
        print("\n[LLM] --- Split sizes ---")
        print(f"[LLM] TRAIN: {len(X_train)}")
        print(f"[LLM] VAL  : {len(X_val)}")
        print(f"[LLM] TEST : {len(X_test)}")

    # ---- Build class-specific training bundles ----
    train_lines_by_class: Dict[str, List[str]] = {label_a: [], label_b: []}
    for txt, lab in zip(X_train, y_train):
        train_lines_by_class[lab].append(str(txt))

    train_bundles_by_class: Dict[str, List[str]] = {}
    for lab in (label_a, label_b):
        bs = _bundle_texts(
            train_lines_by_class[lab],
            bundle_size=cfg.bundle_size,
            stride=cfg.sliding_stride,
            strategy=cfg.bundle_strategy,
            drop_last=cfg.drop_last_incomplete,
        )
        train_bundles_by_class[lab] = bs
        if verbose:
            print(f"[LLM] {lab}: {len(bs)} TRAIN bundles")

    # ---- Embed and index training bundles ----
    if verbose:
        print("\n[LLM] Embedding TRAIN bundles (local, free)...")

    index = _InMemoryPerClassIndex(backend=cfg.retrieval_backend, faiss_hnsw_m=cfg.faiss_hnsw_m)

    for lab in (label_a, label_b):
        if verbose:
            print(f"[LLM]   Embedding {lab}: {len(train_bundles_by_class[lab])} bundles")

        emb = _embed_texts_local(
            embedder,
            train_bundles_by_class[lab],
            batch_size=cfg.local_embedding_batch_size,
            normalize=cfg.local_normalize_embeddings,
            desc=f"[LLM] embed TRAIN {lab}",
            verbose=verbose,
        )
        index.add(lab, emb, train_bundles_by_class[lab])

    # ---- Predict bundles with retrieval fast-path and optional LLM fallback ----
    def predict_bundles(name: str, bundles: List[str]) -> List[str]:
        """Predict labels for bundled texts from one evaluation split.

        Retrieval scores provide the default decision rule; the LLM is queried
        only when similarity evidence is missing or falls inside the uncertainty
        margin defined in the configuration.
        """
        if verbose:
            print(f"\n[LLM] Predicting {name}: {len(bundles)} bundles")
        preds: List[str] = []
        
        llm_available = bool(cfg.use_llm_fallback)
        
        if len(bundles) == 0:
            if verbose:
                print(f"[LLM] ⚠ WARNING: No bundles for {name}!")
            return preds

        # Normalizing once keeps retrieval consistent while avoiding repeated
        # string cleanup inside the prediction loop.
        bundles_norm = [_normalize_bundle(b) for b in bundles]

        # Query embeddings are batched once to keep GPU use efficient and to
        # separate embedding cost from per-item retrieval and prompting.
        q_embs = _embed_texts_local(
            embedder,
            bundles_norm,
            batch_size=cfg.predict_embedding_batch_size,
            normalize=cfg.local_normalize_embeddings,
            desc=f"[LLM] embed QUERIES {name}",
            verbose=verbose,
        )

        llm_calls = 0
        fast_calls = 0

        it = range(len(bundles_norm))
        it = it if not verbose else tqdm(it, desc=f"[LLM] predict {name}", leave=False)

        for i in it:
            b_norm = bundles_norm[i]
            q_emb = q_embs[i]  # already normalized if cfg.local_normalize_embeddings

            r_a = index.topk(label_a, q_emb, cfg.per_class_k)
            r_b = index.topk(label_b, q_emb, cfg.per_class_k)

            sims_a = np.array([sim for sim, _ in r_a], dtype=np.float32)
            sims_b = np.array([sim for sim, _ in r_b], dtype=np.float32)

            # If one class has no retrieved support, the similarity comparison
            # is not meaningful and the LLM is forced when available.
            force_llm = (sims_a.size == 0 or sims_b.size == 0)

            if force_llm:
                score = 0.0
            else:
                a_agg = _aggregate_sims(sims_a, agg=cfg.score_agg)
                b_agg = _aggregate_sims(sims_b, agg=cfg.score_agg)
                score = float(a_agg - b_agg)

            # The fast path is used only when retrieval yields a sufficiently
            # decisive margin between the two class-specific scores.
            use_llm = llm_available and (force_llm or abs(score) < cfg.llm_uncertainty_margin)

            if use_llm:
                if llm_client is None:
                    raise RuntimeError("use_llm_fallback=True but OpenAI client is not available.")
                llm_calls += 1

                sys, user = _build_messages(
                    b_norm,
                    label_a=label_a,
                    label_b=label_b,
                    retrieved_a=r_a,
                    retrieved_b=r_b,
                    max_chars_per_retrieved=cfg.max_chars_per_retrieved,
                )

                try:
                    pred = _chat_classify(
                        llm_client,
                        model=cfg.chat_model,
                        temperature=cfg.temperature,
                        max_tokens=cfg.max_output_tokens,
                        timeout_s=cfg.timeout_s,
                        max_retries=cfg.max_retries,
                        retry_backoff_s=cfg.retry_backoff_s,
                        system_msg=sys,
                        user_msg=user,
                        valid_labels=valid_labels,
                    )
                    preds.append(pred)

                except Exception as e:
                    msg = str(e)

                    # Once the API becomes unavailable, the run degrades
                    # gracefully to embedding-only decisions rather than failing
                    # mid-evaluation and mixing partial outputs.
                    if ("insufficient_quota" in msg) or ("Error code: 429" in msg) or ("429" in msg):
                        llm_available = False

                        if verbose:
                            print("[LLM] ⚠ OpenAI unavailable → embedding-only for rest of this run")

                        # This item is resolved with the same retrieval score
                        # used by the normal fast path.
                        fast_calls += 1
                        pred = label_a if score >= 0 else label_b
                        preds.append(pred)

                    else:
                        raise

            else:
                fast_calls += 1
                pred = label_a if score >= 0 else label_b
                preds.append(pred)

            if verbose and hasattr(it, "set_postfix"):
                total = llm_calls + fast_calls
                it.set_postfix(
                    fast=fast_calls,
                    llm=llm_calls,
                    llm_rate=f"{(llm_calls / max(1, total)):.0%}",
                    backend=index.backend,
                )

        if verbose:
            total = llm_calls + fast_calls
            if total > 0:
                print(f"[LLM] [{name}] Summary: fast={fast_calls}, llm={llm_calls}, llm_rate={llm_calls/total:.1%}")

        return preds

    # ---- Bundle one evaluation split by true class ----
    def bundle_split(name: str, texts: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
        """Bundle one split while preserving ground-truth labels per bundle.

        Validation and test bundles are created separately within each true
        class so every bundled example remains label-pure during evaluation.
        """
        lines_by_class: Dict[str, List[str]] = {label_a: [], label_b: []}
        for t, lab in zip(texts, labels):
            if lab not in lines_by_class:
                raise ValueError(f"Unexpected label '{lab}' in {name}; expected {sorted(valid_labels)}")
            lines_by_class[lab].append(str(t))

        bundles: List[str] = []
        bundle_labels: List[str] = []

        for lab in (label_a, label_b):
            bs = _bundle_texts(
                lines_by_class[lab],
                bundle_size=cfg.bundle_size,
                stride=cfg.sliding_stride,
                strategy=cfg.bundle_strategy,
                drop_last=cfg.drop_last_incomplete,
            )
            if verbose:
                print(f"[LLM] {lab}: {len(bs)} {name} bundles")
            bundles.extend(bs)
            bundle_labels.extend([lab] * len(bs))

        return bundles, bundle_labels

    # ---- Validation ----
    X_val_bundles, y_val_bundles = bundle_split("VAL", X_val, y_val)

    if len(y_val_bundles) == 0:
        raise ValueError(
            "No VAL bundles were produced. Likely causes:\n"
            f"- bundle_size={cfg.bundle_size} is too large for VAL per-class line counts\n"
            f"- drop_last_incomplete={cfg.drop_last_incomplete} discards partial bundles\n"
            "Fix: lower bundle_size, or set drop_last_incomplete=False, or add more data."
        )

    y_val_pred_list = predict_bundles("VAL", X_val_bundles)

    y_val_true = np.array(y_val_bundles, dtype=object)
    y_val_pred = np.array(y_val_pred_list, dtype=object)

    out: Dict[str, EvalResult] = {
        "val": evaluate_classifier(y_val_true, y_val_pred, labels=labels_sorted)
    }

    # ---- Test ----
    if evaluate_test:
        X_test_bundles, y_test_bundles = bundle_split("TEST", X_test, y_test)

        if len(y_test_bundles) == 0:
            raise ValueError("No TEST bundles were produced. Fix: lower bundle_size or set drop_last_incomplete=False.")

        y_test_pred_list = predict_bundles("TEST", X_test_bundles)

        y_test_true = np.array(y_test_bundles, dtype=object)
        y_test_pred = np.array(y_test_pred_list, dtype=object)

        out["test"] = evaluate_classifier(y_test_true, y_test_pred, labels=labels_sorted)

    return out


# ---- Hyperparameter search ----
@dataclass(frozen=True)
class Candidate:
    """Wrapper for one candidate pipeline configuration."""
    cfg: RAGLLMConfig


def search(
    examples: List[Example],
    split: Split,
    candidates: Iterable[Candidate],
    *,
    metric: str = "f1_macro",
    evaluate_test_for_all: bool = False,
    verbose: bool = True,
) -> Tuple[Candidate, EvalResult, EvalResult, List[Tuple[Candidate, EvalResult]]]:
    """Select the best configuration on a fixed validation split.

    Candidates are ranked by the requested validation metric, and the returned
    tuple includes the best candidate, its validation result, its test result,
    and all validation outcomes for later inspection.
    """
    if metric not in {"f1_macro", "f1_weighted", "accuracy"}:
        raise ValueError("metric must be one of: f1_macro, f1_weighted, accuracy")

    def score(res: EvalResult) -> float:
        """Extract the configured selection metric from one evaluation result."""
        return getattr(res, metric)

    candidates = list(candidates)
    if verbose:
        print(f"\n[LLM] Starting search over {len(candidates)} candidates...\n")

    best: Optional[Candidate] = None
    best_val: Optional[EvalResult] = None
    best_test: Optional[EvalResult] = None
    all_val: List[Tuple[Candidate, EvalResult]] = []

    pbar = tqdm(candidates, desc="[LLM] candidates", disable=not verbose)

    for cand in pbar:
        pbar.set_postfix(
            bs=cand.cfg.bundle_size,
            k=cand.cfg.per_class_k,
            fb="on" if cand.cfg.use_llm_fallback else "off",
            margin=f"{cand.cfg.llm_uncertainty_margin:.2f}",
            agg=cand.cfg.score_agg,
            backend=cand.cfg.retrieval_backend,
        )

        out = run_one(
            examples,
            split,
            cand.cfg,
            evaluate_test=evaluate_test_for_all,
            verbose=verbose,
        )
        val_res = out["val"]
        all_val.append((cand, val_res))

        if verbose:
            print(f"[LLM] VAL {metric}: {score(val_res):.4f}")

        if best_val is None or score(val_res) > score(best_val):
            best = cand
            best_val = val_res
            best_test = out.get("test") if evaluate_test_for_all else None
            if verbose:
                print("[LLM] -> New BEST candidate")

    assert best is not None and best_val is not None

    if best_test is None:
        if verbose:
            print("\n[LLM] Evaluating BEST candidate on TEST...\n")
        out_best = run_one(examples, split, best.cfg, evaluate_test=True, verbose=verbose)
        best_test = out_best["test"]

    return best, best_val, best_test, all_val
