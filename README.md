# GATv2 Weighted-Polling Classification — Pima Diabetes (Nested Cross-Validation Comparison)

Code for the manuscript *"Does Graph Attention Improve Weighted-Polling
Classification on the Pima Indians Diabetes Dataset? A Nested
Cross-Validation Comparison with Standard Models"* (under revision at the
*Bulletin of the National Research Centre*).

This repository implements a two-stage classifier — a GATv2 graph encoder
followed by a weighted-polling k-NN/naive Bayes classifier — and tests
whether the graph stage helps on the Pima Indians Diabetes Dataset. The main
analysis is a repeated nested cross-validation in which every hyperparameter,
including those of the encoder, is chosen in an inner loop. The repository
also keeps the original development protocol and a fixed-configuration
leakage-free protocol, which the manuscript uses to quantify how much the
original protocol overstated performance.

## Contents

| File | Description |
|---|---|
| `nested_cv_comparison.py` | **Main analysis.** Repeated nested stratified cross-validation (10 outer folds, 5 inner folds, 3 repeats). GATv2 weighted polling (tuned encoder and original encoder), raw-feature weighted polling, logistic regression, RBF-kernel SVM, random forest, gradient boosting and k-nearest neighbours are scored on identical outer folds, with all hyperparameters selected in the inner loop. Reports accuracy, macro-F1, sensitivity, specificity and AUROC, 95% bootstrap intervals for paired differences, exact McNemar tests and the configurations chosen in each outer fold. Imports from the two files below without modifying them. |
| `gatv2_weighted_polling.py` | The core pipeline: data loading, graph construction, the GATv2 encoder, and the weighted-polling classifier (Methods 1–4). Also implements the *original* evaluation protocol (`run_pipeline`), in which the encoder is trained once on a fixed 80/20 split and polling is then evaluated over all instances. |
| `gatv2_cv_validation.py` | The fixed-configuration *leakage-free* protocol. In every fold, preprocessing statistics, the GATv2 encoder, and the polling vote pool are all computed from that fold's training rows only. Includes a raw-feature baseline, paired exact McNemar tests, and a label-shuffle control. Imports from `gatv2_weighted_polling.py` without modifying it. |
| `shuffle_control_repeated.py` | Repeated label-shuffle control (default 20 permutations) for the original and the leakage-free protocols. |
| `nested_cv_out/` | Raw outputs of the three nested-CV repeats reported in the manuscript (predictions, class-1 scores, chosen configurations, LaTeX table rows). |

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
diabetes mellitus*. The header row was added to the file; no values were
changed.

## Usage

Run all commands from the folder that holds the scripts and `pima_diabetes.csv`.

### Nested cross-validation (main analysis)

```bash
python nested_cv_comparison.py check      # self-test against the original code (about 10 s)
python nested_cv_comparison.py bench      # runtime estimate
python nested_cv_comparison.py run --repeats 3
python nested_cv_comparison.py summary    # metrics, paired CIs, McNemar, LaTeX rows
```

- `run` writes one result file per repeat to `nested_cv_out/` and skips repeats
  that already exist, so an interrupted run can be restarted with the same
  command. To run repeats in parallel, give each its own range, for example
  `--start 0 --repeats 1`, `--start 1 --repeats 2`, `--start 2 --repeats 3`.
- Useful options: `--trials` (GATv2 candidates, default 24), `--outer`
  (default 10), `--inner` (default 5), `--select acc|f1` (inner selection
  criterion), `--out` (output folder), `--boot` (bootstrap resamples).
- A single repeat took about 1.8 hours on the author's CPU (estimate from
  `bench`); timing depends on the machine.

### Original protocol (as used during development)

```bash
python gatv2_weighted_polling.py diabetes
```

### Leakage-free fixed-configuration protocol

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
    (no GATv2 stage).
  - `validate` — repeated stratified k-fold (5 seeds) with per-fold
    retraining; GATv2 vs. the raw-feature baseline on identical folds,
    compared with paired exact McNemar tests. Saves predictions to
    `cv_results_<dataset>.npz`.
  - `control` — label-shuffle sanity check for both protocols.
- `dataset`: `pima` (default; leakage-free search winner), `pima_orig`
  (the configuration selected under the original protocol), or `bean`
  (Dry Bean dataset settings, for comparison).

### Repeated label-shuffle control

```bash
python shuffle_control_repeated.py [dataset] [n_perms] [start]
```

Per-permutation results are appended to `shuffle_rows_<dataset>.csv`; delete
that file before a fresh run.

## Reproducing the paper's results

```bash
python nested_cv_comparison.py run --repeats 3     # main result
python nested_cv_comparison.py summary
python gatv2_cv_validation.py leaky pima_orig      # original protocol (leakage check)
python gatv2_cv_validation.py validate pima        # fixed-configuration leakage-free validation
python shuffle_control_repeated.py pima 20         # label-shuffle control
```

## Origin of the pipeline

The GATv2 + weighted-polling architecture, including the four polling
methods and the tie-break procedure, was originally developed for
multiclass variety identification on the Dry Bean dataset (Koklu &
Ozkan, 2020). `gatv2_weighted_polling.py` is a reconstruction of that
pipeline from the method description in the original study, adapted here
for binary clinical classification.

## License

MIT License. See `LICENSE` for details.

## Citation

If you use this code, please cite the manuscript:

Morsidi, F. Does Graph Attention Improve Weighted-Polling Classification on
the Pima Indians Diabetes Dataset? A Nested Cross-Validation Comparison with
Standard Models. *Bulletin of the National Research Centre* (under revision).

A citable, versioned archive of this repository is available via Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22911107.svg)](https://doi.org/10.5281/zenodo.22911107)

https://doi.org/10.5281/zenodo.23466912
