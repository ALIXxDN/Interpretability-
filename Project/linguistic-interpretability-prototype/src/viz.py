from __future__ import annotations

from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_attention_heatmap(
    attn: np.ndarray,
    tokens: List[str],
    title: Optional[str] = None,
    max_tokens: int = 50,
):
    """Plot a single attention head heatmap.

    Parameters
    ----------
    attn:
        (seq_len, seq_len) attention matrix.
    tokens:
        token strings (length seq_len).
    max_tokens:
        if too long, truncate for readability.
    """
    seq_len = min(len(tokens), attn.shape[0], max_tokens)
    tokens = tokens[:seq_len]
    attn = attn[:seq_len, :seq_len]

    plt.figure(figsize=(10, 8))
    plt.imshow(attn, aspect="auto")
    plt.xticks(range(seq_len), tokens, rotation=90, fontsize=8)
    plt.yticks(range(seq_len), tokens, fontsize=8)
    plt.colorbar(fraction=0.046, pad=0.04)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()
