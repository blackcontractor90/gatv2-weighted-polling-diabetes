# GATv2 Weighted-Polling Classification — Pima Diabetes (Leakage-Aware Evaluation)

Code for the manuscript *"Does Graph Attention Help Weighted-Polling
Classification on Small Clinical Data? A Leakage-Aware Evaluation on the
Pima Indians Diabetes Dataset"* (submitted to the *Bulletin of the National
Research Centre*).

This repository implements a two-stage classifier — a GATv2 graph encoder
followed by a weighted-polling k-NN/naive Bayes classifier — and evaluates
it on the Pima Indians Diabetes Dataset under two protocols: the original
protocol used during development, and a leakage-free protocol used for the
results reported in the paper.

## Contents

| File | Description |
|---|---|
| `gatv2_weighted_polling.py` | The core pipeline: data loading, graph construction, the GATv2 encoder, and the weighted-polling classifier (Methods 1–4). Also implements the *original* evaluation protocol (`run_pipeline`), in which the encoder is trained once on a fixed 80/20 split and polling is then evaluated over all instances. |
| `gatv2_cv_validation.py` | The *leakage-free* evaluation protocol used for the paper's reported results. In every fold, preprocessing statistics, the GATv2 encoder, and the polling vote pool are all computed from that fold's training rows only. Includes a raw-feature baseline, paired exact McNemar tests, and a label-shuffle control. Imports from `gatv2_weighted_polling.py` without modifying it. |

## Requirements

- Python 3.9+
- `torch` and `torch-geometric` (GATv2Conv)
- `numpy`, `pandas`, `scikit-learn`, `scipy`

```bash
pip install torch torch-geometric numpy pandas scikit-learn scipy
```

GPU is not required; all reported runs were performed on CPU.

## Data

Place `pima_diabetes.csv` (columns: `Pregnancies, Glucose, BloodPressure,
SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`) in
the working directory. The dataset is publicly available at
<https://github.com/jbrownlee/Datasets> and originates from Smith et al.
(1988), *Using the ADAP learning algorithm to forecast the onset of
diabetes mellitus*.

## Usage

### Original protocol (as used during development)

```bash
python gatv2_weighted_polling.py diabetes
```

### Leakage-free protocol (as used for the paper's reported results)

```bash
python gatv2_cv_validation.py <mode> [dataset]
```

- `mode`:
  - `leaky` — reproduces the original protocol and reports accuracy
    separately for instances inside vs. outside the encoder's training
    mask, to diagnose the leak.
  - `sweep` — leakage-free hyperparameter search: GATv2 retrained inside
    every fold, all polling configurations scored on the same embeddings.
  - `sweep_raw` — the same polling search for the raw-feature baseline
    (no GATv2 stage), so the baseline receives a comparable tuning budget.
  - `validate` — repeated stratified k-fold (5 seeds) with per-fold
    retraining; GATv2 vs. the raw-feature baseline on identical folds,
    compared with paired exact McNemar tests. Saves predictions to
    `cv_results_<dataset>.npz`.
  - `control` — label-shuffle sanity check for both protocols.
- `dataset`: `pima` (default; leakage-free search winner), `pima_orig`
  (the configuration selected under the original protocol), or `bean`
  (Dry Bean dataset settings, for comparison).

Example:

```bash
python gatv2_cv_validation.py validate pima
python gatv2_cv_validation.py leaky pima_orig
```

## Reproducing the paper's headline results

```bash
python gatv2_cv_validation.py leaky pima_orig    # 79.17% acc, 77.38% macro-F1 (original protocol)
python gatv2_cv_validation.py sweep pima         # leakage-free search
python gatv2_cv_validation.py sweep_raw pima     # raw-feature baseline search
python gatv2_cv_validation.py validate pima      # leakage-free validation vs. baseline (main result)
```

## Origin of the pipeline

The GATv2 + weighted-polling architecture, including the four polling
methods and the tie-break procedure, was originally developed for
multiclass variety identification on the Dry Bean dataset (Koklu &
Ozkan, 2020). `gatv2_weighted_polling.py` is a reconstruction of that
pipeline from the method description in the original study, adapted here
for binary clinical classification.

## License

[TODO: MIT, or your preferred license]

## Citation

If you use this code, please cite:

[TODO: full citation once the manuscript has a DOI/volume/issue]

A citable, versioned archive of this repository is available via Zenodo:
[TODO: Zenodo DOI badge/link, added at submission]
