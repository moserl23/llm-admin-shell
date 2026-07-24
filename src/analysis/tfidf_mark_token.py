#!/usr/bin/env python3
"""
Rebuild a selected TF-IDF experiment row and render an attribution report.

The script reconstructs the original data split and model configuration from a
results CSV row, retrains the corresponding classifier, and highlights the most
influential n-grams present in each example.
"""

from __future__ import annotations

import argparse
import ast
import csv
import html
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB

from src.core.shared.loader import load_examples, LoadConfig
from src.core.ml.splits import make_splits

from pathlib import Path

# -----------------------------
# Vectorizer config (must match tfidf/pipeline.py)
# -----------------------------
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
    """Build the TF-IDF vectorizer from the serialized experiment settings."""
    return TfidfVectorizer(
        analyzer=cfg.analyzer,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        sublinear_tf=cfg.sublinear_tf,
        lowercase=cfg.lowercase,
        max_features=cfg.max_features,
        binary=cfg.binary,
    )


# -----------------------------
# Model factory (aligned with tfidf/pipeline.py)
# -----------------------------
def build_model(
    model_name: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    random_state: int = 42,
) -> BaseEstimator:
    """Instantiate the classifier used in the original experiment row.

    Defaults are chosen to mirror the training pipeline so that attribution is
    computed on a faithful reconstruction of the selected configuration.
    """
    params = params or {}

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


# -----------------------------
# LoadConfig reconstruction
# -----------------------------
@dataclass(frozen=True)
class NamedLoad:
    name: str
    cfg: LoadConfig


def infer_log_files_from_results_csv(results_csv: str) -> tuple[str, ...]:
    """Infer the log source from the results filename.

    This assumes the experiment CSV naming scheme encodes the dataset family,
    which keeps report generation aligned with the original benchmark run.
    """
    name = Path(results_csv).name.lower()

    if "syslog" in name:
        return ("syslog.log",)

    if "audit" in name:
        return ("audit.log",)

    if "nextcloud" in name:
        return ("nextcloud.log",)

    raise ValueError(
        f"Could not infer log type from CSV filename: {results_csv!r}. "
        "Expected filename to contain 'audit', 'syslog', or 'nextcloud'."
    )

def make_load_configs(log_files: tuple[str, ...]) -> List[NamedLoad]:
    """
    Recreate the load-configuration grid used to produce the results CSV.

    The names must stay in sync with `tfidf_360_nested.py`, otherwise a
    selected row cannot be mapped back to the data-loading setup it used.
    """
    base = dict(
        log_files=log_files,
        prefix_with_log_type=False,
        max_lines_per_file=None,
    )

    out: List[NamedLoad] = []

    # Keep the naming scheme identical to the experiment grid so row lookup is stable.
    for preprocess_mode in ["raw", "soft", "aggressive"]:
        out.append(
            NamedLoad(
                name=f"none_{preprocess_mode}",
                cfg=LoadConfig(
                    **base,
                    preprocess_mode=preprocess_mode,
                    window_mode="none",
                ),
            )
        )

    # Line windows expand local context while preserving the original stride choices.
    for preprocess_mode in ["soft", "aggressive"]:
        for ws in [4, 5, 10, 25, 50]:
            st = max(1, 2, ws // 2)
            out.append(
                NamedLoad(
                    name=f"lines_ws{ws}_st{st}_{preprocess_mode}",
                    cfg=LoadConfig(
                        **base,
                        preprocess_mode=preprocess_mode,
                        window_mode="lines",
                        window_size=ws,
                        window_stride=st,
                        window_drop_last=True,
                        join_token=" <EOL> ",
                    ),
                )
            )

    # CID windows mirror the subset explored in the nested TF-IDF search.
    for preprocess_mode in ["soft"]:
        for ws in [10, 25, 50]:
            st = max(1, ws // 2)
            out.append(
                NamedLoad(
                    name=f"cids_ws{ws}_st{st}_{preprocess_mode}",
                    cfg=LoadConfig(
                        **base,
                        preprocess_mode=preprocess_mode,
                        window_mode="cids",
                        window_size=ws,
                        window_stride=st,
                        window_drop_last=True,
                        cid_prefix="CID",
                    ),
                )
            )

    return out

def resolve_load_config(load_name: str, log_files: tuple[str, ...]) -> LoadConfig:
    """Resolve a serialized load-config name back to its `LoadConfig`.

    Raises when the local reconstruction no longer matches the grid used during
    experiment generation, which would invalidate the report.
    """
    grid = make_load_configs(log_files)
    for named in grid:
        if named.name == load_name:
            return named.cfg
    raise KeyError(
        f"Could not resolve selected_load_name='{load_name}'.\n"
        "Make sure make_load_configs() here matches the grid used in experiments/tfidf_360_nested.py."
    )




# -----------------------------
# Parsing helpers for CSV fields
# -----------------------------
_VECTOR_RE = re.compile(r"VectorizerConfig\s*\((.*)\)\s*$")


def _parse_vectorizer_config(s: str) -> VectorizerConfig:
    """
    Parse a serialized vectorizer configuration from the results CSV.

    The CSV may store either the dataclass-style repr or a plain dict literal.
    Both forms are supported to keep older result files readable.
    """
    s = (s or "").strip()

    if s.startswith("{") and s.endswith("}"):
        d = ast.literal_eval(s)
        return VectorizerConfig(**d)

    m = _VECTOR_RE.match(s)
    if not m:
        raise ValueError(f"Cannot parse vectorizer config from: {s!r}")

    inside = m.group(1).strip()

    # Split only on top-level commas so tuple-valued fields survive intact.
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in inside:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)
    last = "".join(buf).strip()
    if last:
        parts.append(last)

    kv: Dict[str, Any] = {}
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        kv[k.strip()] = ast.literal_eval(v.strip())

    return VectorizerConfig(**kv)


def _parse_model_params(s: str) -> Dict[str, Any]:
    """Parse the serialized model-parameter dict stored in the results CSV."""
    s = (s or "").strip()
    if not s:
        return {}
    obj = ast.literal_eval(s)
    if not isinstance(obj, dict):
        raise ValueError(f"selected_model_params must be a dict repr, got: {s!r}")
    return obj


# -----------------------------
# Attribution
# -----------------------------
@dataclass
class TopFeature:
    text: str
    contribution: float


@dataclass
class ModelAttribution:
    model: str
    pred: str
    decision: float
    top_features: List[TopFeature]
    highlighted_html: str


def _decision_value(clf: BaseEstimator, X_row) -> float:
    """Return a scalar decision score with a consistent sign convention.

    When a classifier exposes different confidence APIs, this normalizes them
    to a single value used only for display in the report.
    """
    if hasattr(clf, "decision_function"):
        return float(clf.decision_function(X_row)[0])
    if hasattr(clf, "predict_log_proba"):
        lp = clf.predict_log_proba(X_row)[0]
        return float(lp[1] - lp[0])
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X_row)[0]
        return float(p[1] - p[0])
    return 0.0


def _get_weights_for_attribution(clf: BaseEstimator) -> np.ndarray:
    """
    Return feature weights for additive attribution toward the positive class.

    Linear models expose `coef_` directly; Naive Bayes uses the classwise
    log-probability difference so contributions remain comparable in sign.
    """
    if hasattr(clf, "coef_"):
        return np.asarray(getattr(clf, "coef_")).ravel()
    if hasattr(clf, "feature_log_prob_"):
        logp = np.asarray(getattr(clf, "feature_log_prob_"))
        if logp.shape[0] != 2:
            raise ValueError("Binary classification expected for NB attribution.")
        return (logp[1] - logp[0]).ravel()
    raise TypeError(f"Unsupported model type for attribution: {type(clf).__name__}")


def _top_k_features_for_text(
    text: str,
    vectorizer: TfidfVectorizer,
    clf: BaseEstimator,
    *,
    labels_sorted: List[str],
    k: int = 3,
) -> Tuple[str, float, List[TopFeature]]:
    """
    Return the strongest present features supporting the predicted class.

    Contributions are computed on the transformed row only, so the report
    highlights evidence actually present in the example rather than global weights.
    """
    X = vectorizer.transform([text])
    pred_lab = str(clf.predict(X)[0])
    dec = _decision_value(clf, X)

    w = _get_weights_for_attribution(clf)
    feat_names = vectorizer.get_feature_names_out()

    nz = X.nonzero()[1]
    if nz.size == 0:
        return pred_lab, dec, [TopFeature("<NO_FEATURES>", 0.0)]

    x_vals = X.data
    contrib1 = w[nz] * x_vals

    # `w_j * x_j` is directional toward `labels_sorted[1]`; flip the sign when the
    # prediction is the other class so "top" always means evidence for the prediction.
    pos_label = labels_sorted[1]
    is_pos = (pred_lab == pos_label)
    scores = contrib1 if is_pos else -contrib1

    order = np.argsort(-scores)[:k]
    top_feats: List[TopFeature] = []
    for idx in order:
        fidx = int(nz[idx])
        ftxt = str(feat_names[fidx])
        fscore = float(scores[idx])
        top_feats.append(TopFeature(ftxt, fscore))

    return pred_lab, dec, top_feats


# -----------------------------
# Robust highlighting
# -----------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


@dataclass(frozen=True)
class _TokenSpan:
    tok: str
    start: int
    end: int


def _tokenize_with_spans(text: str) -> List[_TokenSpan]:
    """Tokenize text into word-like spans used for robust highlighting.

    The tokenizer is intentionally simple but close enough to sklearn's default
    behavior to recover word n-grams across punctuation boundaries.
    """
    out: List[_TokenSpan] = []
    for m in _WORD_RE.finditer(text):
        out.append(_TokenSpan(m.group(0), m.start(), m.end()))
    return out


def _ranges_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """Return whether two half-open character ranges overlap."""
    return not (a[1] <= b[0] or b[1] <= a[0])


def _find_word_ngram_ranges(text: str, ngram: str, *, max_hits: int = 6) -> List[Tuple[int, int]]:
    """
    Locate word n-gram matches by token sequence rather than raw substring.

    This makes highlighting resilient to punctuation and casing differences
    between the original text and the vectorizer's tokenization.
    """
    toks = _tokenize_with_spans(text)
    if not toks:
        return []

    pat_toks = [t for t in _WORD_RE.findall(ngram)]
    if not pat_toks:
        return []

    pat = [t.lower() for t in pat_toks]
    seq = [t.tok.lower() for t in toks]
    m = len(pat)

    hits: List[Tuple[int, int]] = []
    for i in range(0, len(seq) - m + 1):
        if seq[i : i + m] == pat:
            start = toks[i].start
            end = toks[i + m - 1].end
            hits.append((start, end))
            if len(hits) >= max_hits:
                break
    return hits


def _find_char_ngram_ranges(text: str, needle: str, *, max_hits: int = 6) -> List[Tuple[int, int]]:
    """
    Locate character n-gram matches with case-insensitive substring search.

    Overlaps are allowed here and resolved later so higher-ranked features can
    claim the most informative spans first.
    """
    if not needle:
        return []
    low_text = text.lower()
    low_need = needle.lower()
    hits: List[Tuple[int, int]] = []
    pos = 0
    while True:
        idx = low_text.find(low_need, pos)
        if idx == -1:
            break
        hits.append((idx, idx + len(needle)))
        if len(hits) >= max_hits:
            break
        pos = idx + 1
    return hits


def _apply_highlights_non_overlapping(
    text: str,
    spans: List[Tuple[int, int, int, str]],
) -> str:
    """
    Apply ranked highlight spans while preserving valid escaped HTML.

    Overlapping candidates are discarded so the rendered text stays readable
    and each character belongs to at most one highlighted feature.
    """
    styles = [
        "background:#ffe5e5;color:#b00000;font-weight:700;padding:0 2px;border-radius:3px;",
        "background:#fff1db;color:#b05a00;font-weight:700;padding:0 2px;border-radius:3px;",
        "background:#e6ffe6;color:#006b00;font-weight:700;padding:0 2px;border-radius:3px;",
    ]

    # Prefer earlier spans, then higher-ranked features, then longer matches.
    spans_sorted = sorted(spans, key=lambda x: (x[0], x[2], -(x[1] - x[0])))

    accepted: List[Tuple[int, int, int, str]] = []
    for s, e, r, tip in spans_sorted:
        if s < 0 or e <= s or e > len(text):
            continue
        if any(_ranges_overlap((s, e), (as_, ae)) for as_, ae, _, _ in accepted):
            continue
        accepted.append((s, e, r, tip))

    accepted.sort(key=lambda x: x[0])

    out_parts: List[str] = []
    cur = 0
    for s, e, r, tip in accepted:
        out_parts.append(html.escape(text[cur:s]))
        tooltip_esc = html.escape(tip)
        out_parts.append(
            f"<span class='hl' style='{styles[r]}' title='{tooltip_esc}'>"
            f"{html.escape(text[s:e])}</span>"
        )
        cur = e
    out_parts.append(html.escape(text[cur:]))
    return "".join(out_parts)


def _highlight_text_html(
    text: str,
    ranked_features: List[TopFeature],
    *,
    analyzer: str,
    max_hits_per_feature: int = 4,
) -> str:
    """
    Render HTML highlights for the top attributed features in one example.

    Word analyzers match by token sequence; character analyzers match by
    substring. In both cases matches are capped and forced to be non-overlapping.
    """
    spans: List[Tuple[int, int, int, str]] = []

    for rank, feat in enumerate(ranked_features[:3]):
        token = (feat.text or "").strip()
        if not token:
            continue

        tooltip = f"{token} | contribution={feat.contribution:+.6f}"

        if analyzer == "word":
            ranges = _find_word_ngram_ranges(text, token, max_hits=max_hits_per_feature)
        else:
            ranges = _find_char_ngram_ranges(text, token, max_hits=max_hits_per_feature)

        for (s, e) in ranges:
            spans.append((s, e, rank, tooltip))

    if not spans:
        return html.escape(text)

    return _apply_highlights_non_overlapping(text, spans)


# -----------------------------
# HTML report
# -----------------------------
def _build_html_report(
    rows: List[Tuple[str, str, List[ModelAttribution]]],
    *,
    title: str,
    subtitle: str,
) -> str:
    """Build the standalone HTML attribution report for the selected examples."""
    css = """
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Noto Sans", "Liberation Sans", sans-serif;
           margin: 20px; color: #111; }
    h1 { margin: 0 0 8px 0; }
    .sub { color:#444; margin: 0 0 18px 0; max-width: 1100px; }
    .logblock { border: 1px solid #ddd; border-radius: 10px; padding: 12px 14px; margin: 14px 0; }
    .logline { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
               background: #fafafa; padding: 10px; border-radius: 8px; white-space: pre-wrap; word-break: break-word; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    th, td { text-align: left; padding: 8px; border-top: 1px solid #eee; vertical-align: top; }
    th { background: #fcfcfc; color: #333; font-weight: 600; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    .pill { display:inline-block; padding: 2px 8px; border-radius: 999px; background:#f1f5f9; color:#0f172a; font-size: 12px; }
    .note { color:#555; font-size: 13px; margin-top: 12px; max-width: 1100px; }
    .meta { color:#555; font-size: 13px; margin: 10px 0 0 0; }
    """

    head = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(title)}</title>
  <style>{css}</style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="sub">{html.escape(subtitle)}</p>
  <p class="meta">
    Hover highlighted n-grams to see contribution. Top1=red, Top2=orange, Top3=green.
  </p>
"""

    blocks: List[str] = []
    for i, (text, true_label, attribs) in enumerate(rows, start=1):
        blocks.append("<div class='logblock'>")
        blocks.append(f"<div class='pill'>Example #{i}</div>")
        blocks.append(f"<div class='mono' style='margin-top:6px;'>True label: <b>{html.escape(true_label)}</b></div>")
        blocks.append(f"<div class='logline' style='margin-top:8px;'>{html.escape(text)}</div>")

        blocks.append("<table>")
        blocks.append("<tr><th>Model</th><th>Pred</th><th>Decision</th><th>Highlighted (top-3)</th><th>Top features</th></tr>")

        for a in attribs:
            top_list = "<br/>".join(
                f"<span class='mono'>{html.escape(f.text)}</span> "
                f"<span class='mono' style='color:#555'>({f.contribution:+.6f})</span>"
                for f in a.top_features[:3]
            )
            blocks.append(
                "<tr>"
                f"<td class='mono'>{html.escape(a.model)}</td>"
                f"<td class='mono'>{html.escape(a.pred)}</td>"
                f"<td class='mono'>{a.decision:+.6f}</td>"
                f"<td class='logline'>{a.highlighted_html}</td>"
                f"<td>{top_list}</td>"
                "</tr>"
            )

        blocks.append("</table>")
        blocks.append("</div>")

    tail = """
  <p class="note">
    Note: word highlighting uses token-sequence matching (case-insensitive, punctuation-robust).
    Char highlighting uses case-insensitive substring matching with overlap avoidance.
  </p>
</body>
</html>
"""
    return head + "\n".join(blocks) + tail


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for row selection and report generation."""
    p = argparse.ArgumentParser()
    p.add_argument("--results_csv", type=str, required=True, help="CSV from experiments/tfidf_360_nested.py")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--row_index", type=int, help="0-based row index in CSV")
    g.add_argument("--outer_i", type=int, help="Select row by outer_i column (1..)")
    g.add_argument(
        "--best_by_metric",
        action="store_true",
        help="Automatically select the row with the best value for --selection_metric",
    )

    p.add_argument(
        "--selection_metric",
        type=str,
        default="test_f1_macro",
        help=(
            "Metric column used when --best_by_metric is enabled, "
            "e.g. test_f1_macro, test_balanced_accuracy, test_mcc"
        ),
    )
    p.add_argument(
        "--selection_mode",
        type=str,
        default="max",
        choices=["max", "min"],
        help="Whether the best row is the maximum or minimum value of --selection_metric (default: max)",
    )

    p.add_argument("--which", type=str, default="test", choices=["val", "test"], help="Generate report for VAL or TEST split")
    p.add_argument("--max_lines", type=int, default=200, help="Max examples included in report")
    p.add_argument("--out_html", type=str, default="results/tfidf_attribution_report.html")
    p.add_argument("--top_k", type=int, default=3, help="How many top features to extract (highlight uses top-3)")
    p.add_argument("--max_hits_per_feature", type=int, default=4, help="Highlight up to this many occurrences per top feature")
    return p.parse_args()


def read_selected_row(
    path: str,
    *,
    row_index: Optional[int],
    outer_i: Optional[int],
    best_by_metric: bool,
    selection_metric: str,
    selection_mode: str,
) -> Dict[str, str]:
    """Select one experiment row from the results CSV.

    Rows can be chosen directly, by outer split index, or by the best value of
    a requested metric. The returned mapping is the raw CSV row.
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"No rows found in CSV: {path}")

    if row_index is not None:
        if row_index < 0 or row_index >= len(rows):
            raise IndexError(f"--row_index out of range: {row_index}, CSV has {len(rows)} rows")
        return rows[row_index]

    if outer_i is not None:
        matches = [r for r in rows if str(r.get("outer_i", "")).strip() == str(outer_i)]
        if not matches:
            raise KeyError(f"No row found with outer_i={outer_i}")
        if len(matches) > 1:
            raise RuntimeError(f"Multiple rows found with outer_i={outer_i}; CSV might contain duplicates.")
        return matches[0]

    if best_by_metric:
        if selection_metric not in rows[0]:
            raise KeyError(
                f"Metric column '{selection_metric}' not found in CSV. "
                f"Available columns include: {list(rows[0].keys())}"
            )

        # Filter out missing or non-numeric cells so selection reflects only
        # rows where the requested metric was actually computed.
        scored_rows: List[Tuple[float, Dict[str, str]]] = []
        for r in rows:
            raw = str(r.get(selection_metric, "")).strip()
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            scored_rows.append((value, r))

        if not scored_rows:
            raise RuntimeError(
                f"No numeric values found in selection metric column '{selection_metric}'."
            )

        if selection_mode == "max":
            best_value, best_row = max(scored_rows, key=lambda x: x[0])
        else:
            best_value, best_row = min(scored_rows, key=lambda x: x[0])

        print(
            f"Selected best row by metric: {selection_metric} "
            f"({selection_mode}) = {best_value:.6f}, "
            f"outer_i={best_row.get('outer_i')}"
        )
        return best_row

    raise RuntimeError("No row selection mode was provided.")

def main() -> None:
    """Rebuild the selected TF-IDF model and write an attribution report."""
    args = parse_args()

    # ---- Recover the original experiment configuration ----
    log_files = infer_log_files_from_results_csv(args.results_csv)
    print(f"Inferred log_files from CSV: {log_files}")

    row = read_selected_row(
        args.results_csv,
        row_index=args.row_index,
        outer_i=args.outer_i,
        best_by_metric=args.best_by_metric,
        selection_metric=args.selection_metric,
        selection_mode=args.selection_mode,
    )

    # The CSV stores the human/AI group identifiers for the outer split. Reusing
    # them here ensures the report is generated on the same validation/test split.
    val_groups = [row["val_human"], row["val_ai"]]
    test_groups = [row["test_human"], row["test_ai"]]

    load_name = row["selected_load_name"]
    model_name = row["selected_model"]
    model_params = _parse_model_params(row["selected_model_params"])
    vec_cfg = _parse_vectorizer_config(row["selected_vectorizer"])

    if model_name not in {"svm", "logreg", "sgd_hinge", "sgd_log", "pa_like", "ridge", "mnb", "cnb", "bnb"}:
        raise ValueError(f"Unsupported selected_model from CSV: {model_name!r}")

    load_cfg = resolve_load_config(load_name, log_files)
    examples = load_examples(load_cfg)
    if not examples:
        raise RuntimeError("load_examples produced no examples for this LoadConfig.")

    # ---- Recreate the train/validation/test partition ----
    y_all = np.array([e.label for e in examples], dtype=object)
    groups = np.array([e.group for e in examples], dtype=object)

    split = make_splits(
        y_all,
        groups=groups,
        val_groups=val_groups,
        test_groups=test_groups,
    )

    X_all = np.array([e.text for e in examples], dtype=object)
    y_all = np.array([e.label for e in examples], dtype=object)
    labels_sorted = sorted(set(map(str, y_all.tolist())))
    if len(labels_sorted) != 2:
        raise ValueError(f"Expected binary labels, got {labels_sorted}")

    # ---- Train the reconstructed model ----
    X_train, y_train = X_all[split.train_idx], y_all[split.train_idx]
    X_val, y_val = X_all[split.val_idx], y_all[split.val_idx]
    X_test, y_test = X_all[split.test_idx], y_all[split.test_idx]

    vec = build_vectorizer(vec_cfg)
    X_train_vec = vec.fit_transform(X_train)

    clf = build_model(model_name, model_params, random_state=42)
    clf.fit(X_train_vec, y_train)

    if args.which == "val":
        X_target = X_val
        y_target = y_val
        split_name = "VAL"
    else:
        X_target = X_test
        y_target = y_test
        split_name = "TEST"

    # Shuffle with a fixed seed so the report samples are deterministic without
    # depending on the original dataset order.
    n = min(int(args.max_lines), int(len(X_target)))

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(X_target))

    X_target = X_target[perm]
    y_target = y_target[perm]

    X_target = X_target[:n]
    y_target = y_target[:n]

    rows_html: List[Tuple[str, str, List[ModelAttribution]]] = []

    # sklearn may expose a callable analyzer internally; the report only needs
    # to distinguish between word- and character-style highlighting.
    analyzer = str(getattr(vec, "analyzer", "word"))
    if callable(analyzer):
        analyzer = "word"

    # ---- Attribute and render selected examples ----
    for text, y_true in zip(X_target.tolist(), y_target.tolist()):
        text_s = str(text)
        pred, dec, top_feats = _top_k_features_for_text(
            text_s,
            vec,
            clf,
            labels_sorted=labels_sorted,
            k=int(args.top_k),
        )

        highlighted = _highlight_text_html(
            text_s,
            top_feats,
            analyzer=vec_cfg.analyzer,
            max_hits_per_feature=int(args.max_hits_per_feature),
        )

        per_model = [
            ModelAttribution(
                model=f"{model_name} {model_params}",
                pred=str(pred),
                decision=float(dec),
                top_features=top_feats,
                highlighted_html=highlighted,
            )
        ]

        rows_html.append((text_s, str(y_true), per_model))

    title = f"TF-IDF Attribution Report ({split_name})"
    subtitle = (
        f"CSV row selection: "
        f"{'row_index='+str(args.row_index) if args.row_index is not None else 'outer_i='+str(args.outer_i)} | "
        f"LoadConfig={load_name} | "
        f"Vectorizer={vec_cfg} | "
        f"Model={model_name} params={model_params} | "
        f"val_groups={val_groups} test_groups={test_groups} | "
        f"shown_examples={len(rows_html)}"
    )

    html_doc = _build_html_report(rows_html, title=title, subtitle=subtitle)

    os.makedirs(os.path.dirname(args.out_html) or ".", exist_ok=True)
    with open(args.out_html, "w", encoding="utf-8") as f:
        f.write(html_doc)

    print(f"Wrote HTML report to: {args.out_html}")
    print("Open it in your browser. Hover highlights for contribution scores.")


if __name__ == "__main__":
    main()
