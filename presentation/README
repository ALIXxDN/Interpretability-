# Linguistic Interpretability Prototype

A survey-based course project prototype for analyzing linguistic interpretability in Transformer-based PLMs using probing and layer-wise analysis.

---

## 1) What is this repo? (Overview)

This repository contains an *educational prototype aligned with a survey (review) paper on linguistic interpretability of Transformer-based pre-trained language models (PLMs).  
Because the reference paper is a survey and does not provide runnable code or a single fixed dataset for replication, this project does not aim to reproduce paper results.  
Instead, it demonstrates how core interpretability methods highlighted in the survey can be implemented in practice.

---

## 2) Key Features / What’s inside

- Layer-wise probing for syntactic information (POS tagging)
- Extraction of hidden representations across Transformer layers
- Attention heatmap* visualization (qualitative example)
- Reproducible, lightweight setup suitable for a course project

---

## 3) Requirements

- Python 3.10+
- Main libraries:
  - `torch`
  - `transformers`
  - `datasets==2.14.6`
  - `scikit-learn`
  - `conllu`
  - `numpy`, `pandas`, `matplotlib`, `joblib`

> Note: `datasets==2.14.6` is used due to compatibility issues with newer versions when loading UD-style datasets.

---

## 4) How to Run (Quickstart)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Open and run the notebook:

demo/linguistic_interpretability_prototype.ipynb

Run all cells (Jupyter/Colab)





---

5) Outputs

After running the notebook, outputs are created in demo/output/:

results.csv — layer-wise probing results (accuracy per layer)

probe.joblib — saved probing classifier

Plots (layer-wise accuracy curve) and attention heatmaps (shown in the notebook)



---

6) Limitations / Disclaimer

This is a survey-based prototype, not a replication study

Probing measures extractability, not causal usage by the model

Results are illustrative/educational and based on a small-scale setup



---

7) Reference

Linguistic Interpretability of Transformer-based Pre-trained Language Models
A Systematic Survey
