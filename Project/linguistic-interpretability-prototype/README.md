# Linguistic Interpretability Prototype
**Survey-based Prototype for Linguistic Interpretability of Transformer-based PLMs**

## Overview
This project is a **small, reproducible prototype** inspired by a recent **survey paper on linguistic interpretability of Transformer-based Pre-trained Language Models (PLMs)**.
Since the reference paper is a **survey (review) study** and does not provide executable code or datasets, this project implements a **demonstration prototype** aligned with the **dominant interpretability methods identified in the survey**, namely:

- Probing classifiers
- Layer-wise analysis
- Attention visualization

The goal is **not to reproduce numerical results**, but to **illustrate how linguistic knowledge (syntax/morphology) can be analyzed in PLM representations**.

## Project Structure
```
.
├── README.md
├── src/
│   ├── ud.py
│   ├── extract.py
│   ├── probes.py
│   └── viz.py
├── demo/
│   ├── linguistic_interpretability_prototype.ipynb
│   └── output/
│       ├── results.csv
│       └── probe.joblib
```

## Environment & Dependencies
### Python
- Python 3.10+ (tested on Google Colab)

### Required Libraries
```
torch
transformers
datasets==2.14.6
scikit-learn
conllu
numpy
pandas
matplotlib
joblib
```

> Important:
> Due to breaking changes in the HuggingFace `datasets` library,
> version **2.14.6** is required for loading Universal Dependencies datasets.

## Installation
```
pip install torch transformers datasets==2.14.6 scikit-learn conllu numpy pandas matplotlib joblib
```

## Data
- Uses **Universal Dependencies (UD)** datasets
- Loaded automatically via HuggingFace `datasets`
- No dataset files are uploaded to the repository

## Running the Demo
1. Open:
   `demo/linguistic_interpretability_prototype.ipynb`
2. Run:
   `Run → Run All`

## Short Experiment Description
The conducted experiments follow a **simple and controlled probing setup**. 
A pre-trained multilingual BERT model is used to extract **contextual word representations** from different Transformer layers. 
A lightweight probing classifier is then trained to predict **part-of-speech (POS) tags** based on these representations.

**Experimental details:**
- **Model:** `bert-base-multilingual-cased`
- **Layers analyzed:** 1–12 (layer-wise comparison)
- **Task:** POS tagging (syntactic probing)
- **Classifier:** Logistic Regression / shallow MLP
- **Evaluation metric:** Accuracy
- **Repetitions:** Single run with a fixed random seed (for reproducibility)

The results illustrate how **syntactic information is distributed across Transformer layers**, without making causal claims about model decision-making.

## Outputs
Generated in `demo/output/`:
- `results.csv`: layer-wise probing results
- `probe.joblib`: trained probing classifier
- Plots and attention heatmaps

## Notes
- Probing measures extractability, not causal usage
- Results are illustrative and educational

## Author
Neo  
MSc Artificial Intelligence
