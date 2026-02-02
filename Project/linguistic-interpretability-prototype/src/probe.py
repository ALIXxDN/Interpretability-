from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .extract import extract_hidden_states_word_level

try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score
except Exception as e:  # pragma: no cover
    SGDClassifier = None
    LabelEncoder = None
    accuracy_score = None


@dataclass
class ProbeResult:
    layer: int
    accuracy: float
    majority_baseline: float
    n_train_tokens: int
    n_eval_tokens: int


def _flatten_majority_baseline(y: np.ndarray) -> float:
    """Compute majority-class accuracy."""
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


def build_token_level_matrices(
    tokens_sents: Sequence[Sequence[str]],
    upos_sents: Sequence[Sequence[str]],
    tokenizer,
    model,
    layers: Sequence[int],
    device: str = "cpu",
    max_length: int = 256,
    drop_upos: Optional[Sequence[str]] = ("PUNCT",),
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Build token-level matrices for probing.

    Returns
    -------
    X_by_layer:
        dict[layer] -> (N_tokens, hidden_dim)
    y:
        (N_tokens,) integer labels (not encoded yet)
    """
    # Accumulate per-layer vectors in lists to avoid repeated concatenations
    xs: Dict[int, List[np.ndarray]] = {int(l): [] for l in layers}
    ys: List[str] = []

    for toks, tags in tqdm(list(zip(tokens_sents, upos_sents)), desc="Extract", leave=False):
        ext = extract_hidden_states_word_level(
            toks,
            tokenizer=tokenizer,
            model=model,
            layers=layers,
            device=device,
            max_length=max_length,
            return_attentions=False,
        )
        # Align tags to survived words
        # ext.word_tokens are the kept tokens; we need their original indices
        # We approximate by re-matching tokens in order:
        # Since we used is_split_into_words, kept word ids preserve the order.
        # So we keep the first len(ext.word_tokens) tags.
        # (In practice, truncation is rare with max_length=256 on UD sentences.)
        kept_n = len(ext.word_tokens)
        kept_tags = list(tags)[:kept_n]

        for i, (tok, tag) in enumerate(zip(ext.word_tokens, kept_tags)):
            if drop_upos and tag in set(drop_upos):
                continue
            ys.append(tag)

        # Because we potentially dropped some tokens (e.g., PUNCT), we must apply the same filter to embeddings
        keep_mask = [not (drop_upos and t in set(drop_upos)) for t in kept_tags]
        for layer in layers:
            mat = ext.hidden_by_layer[int(layer)]
            if mat.shape[0] != len(keep_mask):
                # Defensive: if mismatch, truncate to min length
                m = min(mat.shape[0], len(keep_mask))
                mat = mat[:m]
                km = keep_mask[:m]
            else:
                km = keep_mask
            xs[int(layer)].append(mat[np.array(km, dtype=bool)])

    # Concatenate
    X_by_layer: Dict[int, np.ndarray] = {}
    for layer, parts in xs.items():
        if len(parts) == 0:
            X_by_layer[layer] = np.zeros((0, 0), dtype=np.float32)
        else:
            X_by_layer[layer] = np.concatenate(parts, axis=0).astype(np.float32)

    y_arr = np.array(ys, dtype=object)
    return X_by_layer, y_arr


def train_and_eval_layerwise_probes(
    X_train_by_layer: Dict[int, np.ndarray],
    y_train_str: np.ndarray,
    X_eval_by_layer: Dict[int, np.ndarray],
    y_eval_str: np.ndarray,
    seed: int = 42,
    max_iter: int = 2000,
    tol: float = 1e-3,
) -> Tuple[List[ProbeResult], Dict[int, object], object]:
    """Train one linear probe per layer and evaluate on eval split."""
    if SGDClassifier is None or LabelEncoder is None:
        raise ImportError("scikit-learn is required for probing")

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_str)
    y_eval = le.transform(y_eval_str)

    majority = _flatten_majority_baseline(y_eval)

    probes: Dict[int, object] = {}
    results: List[ProbeResult] = []

    for layer in sorted(X_train_by_layer.keys()):
        Xtr = X_train_by_layer[layer]
        Xev = X_eval_by_layer[layer]

        # Defensive: if shapes mismatch, skip
        if Xtr.size == 0 or Xev.size == 0:
            continue

        clf = SGDClassifier(
            loss="log_loss",
            random_state=seed,
            max_iter=max_iter,
            tol=tol,
        )
        clf.fit(Xtr, y_train)

        pred = clf.predict(Xev)
        acc = float(accuracy_score(y_eval, pred))

        probes[layer] = clf
        results.append(
            ProbeResult(
                layer=int(layer),
                accuracy=acc,
                majority_baseline=float(majority),
                n_train_tokens=int(Xtr.shape[0]),
                n_eval_tokens=int(Xev.shape[0]),
            )
        )

    return results, probes, le
