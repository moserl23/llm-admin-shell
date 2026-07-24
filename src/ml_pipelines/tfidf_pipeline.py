"""TF-IDF baseline pipeline for text classification experiments.

The module defines vectorizer/model factories and a simple candidate search
routine that selects on validation performance and evaluates the test set in a
research-safe way by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable
from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

from src.core.ml.data import Example
from src.core.ml.splits import Split
from src.core.ml.eval import EvalResult, evaluate_classifier


# ---- Vectorizer configuration ----
@dataclass(frozen=True)
class VectorizerConfig:
    analyzer: str = "word"
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    sublinear_tf: bool = True
    lowercase: bool = True
    max_features: Optional[int] = None
    binary: bool = False


def build_vectorizer(cfg: VectorizerConfig) -> TfidfVectorizer:
    """Construct a TF-IDF vectorizer from an immutable configuration.

    The config is kept hashable so identical settings can be cached during
    search. Returns an unfitted sklearn vectorizer instance.
    """
    return TfidfVectorizer(
        analyzer=cfg.analyzer,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        sublinear_tf=cfg.sublinear_tf,
        lowercase=cfg.lowercase,
        max_features=cfg.max_features,
        binary=cfg.binary
    )


# ---- Model factory ----
def build_model(
    model_name: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    random_state: int = 42,
) -> BaseEstimator:
    """Instantiate a supported linear baseline or dummy classifier.

    Defaults are chosen to be sensible for sparse text features, while
    `params` can override model-specific settings. Returns an unfitted estimator.
    """

    params = params or {}

    if model_name == "dummy_most_frequent":
        return DummyClassifier(strategy="most_frequent")

    if model_name == "dummy_stratified":
        base = {"strategy": "stratified", "random_state": random_state}
        return DummyClassifier(**{**base, **params})

    if model_name == "svm":
        return LinearSVC(**{"C": 1.0, **params})

    if model_name == "logreg":
        base = {"max_iter": 2000, "solver": "saga"}
        return LogisticRegression(**{**base, **params})

    if model_name == "sgd_hinge":
        base = {"loss": "hinge", "random_state": random_state}
        return SGDClassifier(**{**base, **params})

    if model_name == "sgd_log":
        base = {"loss": "log_loss", "random_state": random_state}
        return SGDClassifier(**{**base, **params})

    if model_name == "pa_like":
        base = {
            "loss": "hinge",
            "penalty": None,
            "learning_rate": "pa1",
            "eta0": 1.0,
            "random_state": random_state,
        }
        return SGDClassifier(**{**base, **params})

    if model_name == "ridge":
        return RidgeClassifier(**params)

    if model_name == "mnb":
        return MultinomialNB(**{"alpha": 0.1, **params})

    if model_name == "cnb":
        return ComplementNB(**{"alpha": 0.1, **params})

    if model_name == "bnb":
        return BernoulliNB(**{"alpha": 0.1, **params})

    raise ValueError(f"Unknown model_name='{model_name}'")


# ---- Candidate search ----
@dataclass(frozen=True)
class Candidate:
    vectorizer: VectorizerConfig
    model_name: str
    model_params: Dict[str, Any]


def search(
    examples: List[Example],
    split: Split,
    candidates: Iterable[Candidate],
    *,
    metric: str = "f1_macro",
    random_state: int = 42,
    evaluate_test_for_all: bool = False,
    verbose: bool = True,
) -> Tuple[Candidate, EvalResult, EvalResult, List[Tuple[Candidate, EvalResult]]]:
    """Search over vectorizer-model candidates using a fixed train/val/test split.

    Selection is based only on validation performance. Test evaluation is
    deferred to the best candidate by default to avoid optimistic reporting.
    Returns the best candidate, its validation result, its test result, and all
    validation results.
    """

    if metric not in {"f1_macro", "f1_weighted", "accuracy"}:
        raise ValueError("metric must be one of: f1_macro, f1_weighted, accuracy")

    def score(res: EvalResult) -> float:
        return getattr(res, metric)

    candidates = list(candidates)
    total = len(candidates)

    if verbose:
        print(f"\nStarting search over {total} candidates...\n")

    X = np.array([ex.text for ex in examples], dtype=object)
    y = np.array([ex.label for ex in examples], dtype=object)
    labels_sorted = sorted(set(y.tolist()))

    X_train, y_train = X[split.train_idx], y[split.train_idx]
    X_val, y_val = X[split.val_idx], y[split.val_idx]
    X_test, y_test = X[split.test_idx], y[split.test_idx]

    # Vectorization is the expensive part; reuse it across candidates that only
    # differ in classifier settings.
    vec_cache = {}
    grouped: Dict[VectorizerConfig, List[Candidate]] = defaultdict(list)
    for cand in candidates:
        grouped[cand.vectorizer].append(cand)

    best: Optional[Candidate] = None
    best_val: Optional[EvalResult] = None
    best_test: Optional[EvalResult] = None
    all_val: List[Tuple[Candidate, EvalResult]] = []

    pbar = tqdm(total=total, disable=not verbose)

    for vec_cfg, cand_group in grouped.items():

        if verbose:
            print(f"\nFitting vectorizer: {vec_cfg}")

        vec = build_vectorizer(vec_cfg)

        X_train_vec = vec.fit_transform(X_train)
        X_val_vec = vec.transform(X_val)

        X_test_vec = None
        if evaluate_test_for_all:
            # This mode is useful for diagnostics, but it should not be used for
            # headline results because it exposes test performance during search.
            X_test_vec = vec.transform(X_test)

        for cand in cand_group:

            model = build_model(
                cand.model_name,
                cand.model_params,
                random_state=random_state,
            )

            model.fit(X_train_vec, y_train)

            y_val_pred = model.predict(X_val_vec)
            val_res = evaluate_classifier(
                y_val, y_val_pred, labels=labels_sorted
            )
            all_val.append((cand, val_res))

            if best_val is None or score(val_res) > score(best_val):
                # Keep the test score synchronized with the current validation
                # leader only when test evaluation is explicitly enabled.
                best = cand
                best_val = val_res
                if evaluate_test_for_all:
                    y_test_pred = model.predict(X_test_vec)
                    best_test = evaluate_classifier(
                        y_test, y_test_pred, labels=labels_sorted
                    )
                else:
                    best_test = None

            pbar.update(1)

    pbar.close()

    assert best is not None and best_val is not None

    # Recommended evaluation path: touch the test set once after model selection.
    if best_test is None:
        if verbose:
            print("\nEvaluating best candidate on TEST set...\n")

        vec = build_vectorizer(best.vectorizer)
        X_train_vec = vec.fit_transform(X_train)
        X_test_vec = vec.transform(X_test)

        model = build_model(
            best.model_name,
            best.model_params,
            random_state=random_state,
        )
        model.fit(X_train_vec, y_train)

        y_test_pred = model.predict(X_test_vec)
        best_test = evaluate_classifier(
            y_test, y_test_pred, labels=labels_sorted
        )

    if verbose:
        print("\nSearch complete.\n")

    return best, best_val, best_test, all_val
