# Experiments (Prototype)

This file documents the small experiments run in this repository.

## 1) Layer-wise POS probing (UPOS)

**Goal:** estimate which Transformer layers encode information useful for POS prediction using a *lightweight probe*.

- Dataset: Universal Dependencies (UD) via 🤗 `datasets`
  - Example configs:
    - `en_ewt` (English)
    - `fa_seraji` (Persian)
- Model: `bert-base-multilingual-cased` (default)
- Representation:
  - Extract hidden states from all layers (embeddings + encoder layers).
  - Pool subword pieces to word-level vectors by **mean pooling**.
- Probe:
  - `SGDClassifier` (linear) as a lightweight probe
  - Metric: accuracy (token-level)
- Runtime mode:
  - Subsample sentences (`--max_sentences`) for CPU-friendly execution.

### Recommended settings (fast CPU)

- max_sentences: 200
- layers: 0..12
- seed: 42
- exclude UPOS = PUNCT: optional (enabled by default in code)

### Outputs

- CSV with accuracy per layer (and a majority-class baseline)
- Optional saved probe + label encoder (joblib)

## 2) Attention visualization (qualitative)

**Goal:** show how attention weights can be inspected for specific sentences.

- Extract `attentions` from the model (`output_attentions=True`).
- Plot a heatmap for a chosen layer/head and a given sentence.

## Reproducibility

- Fix seeds for:
  - Python `random`
  - NumPy
- Note: exact results may still vary slightly across hardware / library versions.

