from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

try:
    from datasets import load_dataset
except Exception as e:  # pragma: no cover
    load_dataset = None


@dataclass
class UDSentence:
    tokens: List[str]
    upos: List[str]


def load_ud(
    ud_config: str,
    split: str = "train",
    max_sentences: Optional[int] = None,
    seed: int = 42,
) -> List[UDSentence]:
    """Load a Universal Dependencies (UD) treebank split via 🤗 datasets.

    Parameters
    ----------
    ud_config:
        The UD configuration name, e.g. 'en_ewt', 'fa_seraji'.
    split:
        One of: 'train', 'validation', 'test'.
    max_sentences:
        If provided, subsample to at most this many sentences.
    seed:
        Random seed used by datasets shuffle.

    Returns
    -------
    List[UDSentence]
        Each sentence includes tokens and UPOS tags.

    Notes
    -----
    This function expects that the dataset provides fields compatible with UD:
    typically 'tokens' and 'upos'. If field names differ, this function attempts
    to infer them.
    """
    if load_dataset is None:
        raise ImportError(
            "datasets is not available. Install dependencies: pip install -r requirements.txt"
        )

    ds = load_dataset("universal_dependencies", ud_config, split=split)
    # Try to infer fields robustly
    sample = ds[0]
    token_field = "tokens" if "tokens" in sample else None
    upos_field = "upos" if "upos" in sample else None

    if token_field is None:
        # Some variants might use 'token' or similar; fall back to first list-of-strings
        for k, v in sample.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str):
                token_field = k
                break

    if upos_field is None:
        # Look for UPOS-like field
        for k in ("upos", "pos", "xpos"):
            if k in sample:
                upos_field = k
                break

    if token_field is None or upos_field is None:
        raise ValueError(
            f"Could not infer required fields from UD dataset. Keys: {list(sample.keys())}"
        )

    if max_sentences is not None:
        ds = ds.shuffle(seed=seed).select(range(min(max_sentences, len(ds))))

    sents: List[UDSentence] = []
    for ex in ds:
        tokens = ex[token_field]
        upos = ex[upos_field]
        if tokens is None or upos is None:
            continue
        if len(tokens) != len(upos):
            # Skip malformed samples
            continue
        sents.append(UDSentence(tokens=list(tokens), upos=list(upos)))

    return sents


def flatten_tokens_and_labels(sents: List[UDSentence]) -> Tuple[List[List[str]], List[List[str]]]:
    """Return tokens and labels as lists-of-lists for downstream alignment."""
    return [s.tokens for s in sents], [s.upos for s in sents]
