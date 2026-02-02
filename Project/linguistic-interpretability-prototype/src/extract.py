from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except Exception as e:  # pragma: no cover
    torch = None
    AutoModel = None
    AutoTokenizer = None


@dataclass
class Extracted:
    word_tokens: List[str]
    # layer -> (n_words, hidden_dim)
    hidden_by_layer: Dict[int, np.ndarray]
    # layer -> (n_heads, seq_len, seq_len) attention for batch 1 (includes special tokens)
    attn_by_layer: Optional[Dict[int, np.ndarray]] = None
    # tokenizer tokens (including special tokens)
    model_tokens: Optional[List[str]] = None


def load_model_and_tokenizer(
    model_name: str = "bert-base-multilingual-cased",
    device: str = "cpu",
):
    """Load a HuggingFace Transformer model + tokenizer."""
    if torch is None or AutoTokenizer is None or AutoModel is None:
        raise ImportError(
            "transformers/torch not available. Install dependencies: pip install -r requirements.txt"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return tokenizer, model


def _mean_pool_subwords(
    token_embeddings: np.ndarray, word_ids: List[Optional[int]]
) -> Tuple[List[int], np.ndarray]:
    """Pool subword token embeddings into word-level embeddings via mean pooling.

    Parameters
    ----------
    token_embeddings:
        Array of shape (seq_len, hidden_dim)
    word_ids:
        List length seq_len with word index (0..n_words-1) or None for special tokens.

    Returns
    -------
    kept_word_ids:
        Word indices that survived tokenization/truncation, in ascending order.
    word_embeddings:
        Array shape (n_kept_words, hidden_dim)
    """
    # Collect positions for each word id
    positions: Dict[int, List[int]] = {}
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        positions.setdefault(wid, []).append(i)

    kept = sorted(positions.keys())
    pooled = []
    for wid in kept:
        idxs = positions[wid]
        pooled.append(token_embeddings[idxs].mean(axis=0))
    if len(pooled) == 0:
        return [], np.zeros((0, token_embeddings.shape[-1]), dtype=np.float32)

    return kept, np.stack(pooled, axis=0).astype(np.float32)


def extract_hidden_states_word_level(
    tokens: Sequence[str],
    tokenizer,
    model,
    layers: Optional[Sequence[int]] = None,
    device: str = "cpu",
    max_length: int = 256,
    return_attentions: bool = False,
) -> Extracted:
    """Extract word-level hidden states (layer-wise) for a tokenized sentence.

    Parameters
    ----------
    tokens:
        Pre-split tokens (UD tokens recommended).
    layers:
        Which layers to return. If None, return all layers available.
        Note: layer 0 corresponds to the embeddings output.
    return_attentions:
        If True, also return attention matrices (layer-wise).
    """
    if torch is None:
        raise ImportError("torch is required")

    # Use is_split_into_words to preserve alignment to original tokens
    enc = tokenizer(
        list(tokens),
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )

    # word_ids() exists on fast tokenizers
    word_ids = enc.word_ids(batch_index=0)

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(
            **enc,
            output_hidden_states=True,
            output_attentions=return_attentions,
            return_dict=True,
        )

    hidden_states = out.hidden_states  # tuple: (n_layers+1, batch, seq_len, dim)
    n_layers = len(hidden_states) - 1

    if layers is None:
        layers = list(range(n_layers + 1))

    hidden_by_layer: Dict[int, np.ndarray] = {}
    kept_word_ids: Optional[List[int]] = None
    word_tokens: List[str] = []

    for layer in layers:
        hs = hidden_states[layer][0].detach().cpu().numpy()  # (seq_len, dim)
        kept, pooled = _mean_pool_subwords(hs, word_ids)
        hidden_by_layer[int(layer)] = pooled
        if kept_word_ids is None:
            kept_word_ids = kept

    if kept_word_ids is None:
        kept_word_ids = []

    # Keep only tokens that survived truncation
    word_tokens = [tokens[i] for i in kept_word_ids if i < len(tokens)]

    attn_by_layer: Optional[Dict[int, np.ndarray]] = None
    model_tokens = None
    if return_attentions and getattr(out, "attentions", None) is not None:
        attn_by_layer = {}
        attentions = out.attentions  # tuple: (n_layers, batch, n_heads, seq_len, seq_len)
        # Note: attentions do not include the embedding layer (layer 0)
        for layer_idx, attn in enumerate(attentions, start=1):
            attn_by_layer[layer_idx] = attn[0].detach().cpu().numpy()
        model_tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].detach().cpu().tolist())

    return Extracted(
        word_tokens=word_tokens,
        hidden_by_layer=hidden_by_layer,
        attn_by_layer=attn_by_layer,
        model_tokens=model_tokens,
    )
