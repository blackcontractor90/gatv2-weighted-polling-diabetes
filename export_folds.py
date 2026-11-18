# ============================================================
# export_folds.py
#
# Regenerates the exact fold assignments used by nested_cv_comparison.py and
# nested_cv_inductive.py (Reviewer 3, point 14), with no training. The splits are fully
# determined by the seeds below, so the CSV files are a convenience copy for the code release.
#
#   outer folds : StratifiedKFold(n_splits=10, shuffle=True, random_state=r), r = repeat seed 0..2
#   inner folds : StratifiedKFold(n_splits=5,  shuffle=True, random_state=1000*(r+1)+f),
#                 f = outer fold number 1..10, applied to the outer-training rows
#
# Writes into --out (default fold_assignments):
#   outer_folds.csv   row_index, outcome, outer_fold_repeat0, outer_fold_repeat1, ...
#   inner_folds.csv   repeat, outer_fold, row_index, inner_validation_fold
#                     (one line per outer-training row: the inner fold in which it is validation)
#   seeds.txt         all seeds and split settings, in plain text
# row_index is the 0-based position of the row in the order the dataset is loaded (768 rows).
#
# Usage:  python export_folds.py [--repeats 3] [--outer 10] [--inner 5] [--out fold_assignments]
# ============================================================

import os
import argparse

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold

from gatv2_weighted_polling import load_dataset
from gatv2_cv_validation import DATASETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pima")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--outer", type=int, default=10)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--out", default="fold_assignments")
    args = ap.parse_args()

    D = DATASETS[args.dataset]
    X_raw, y, classes, _ = load_dataset(D["path"], D["features"], D["label"], sep=D["sep"],
                                        zero_as_missing_cols=D["zero_as_missing"])
    n = len(y)
    os.makedirs(args.out, exist_ok=True)

    outer_tab = {"row_index": np.arange(n), "outcome": y}
    inner_rows = []
    for r in range(args.repeats):
        outer = StratifiedKFold(n_splits=args.outer, shuffle=True, random_state=r)
        fold_of = np.zeros(n, dtype=int)
        for f, (tr, te) in enumerate(outer.split(X_raw, y), 1):
            fold_of[te] = f
            inner = StratifiedKFold(n_splits=args.inner, shuffle=True, random_state=1000 * (r + 1) + f)
            for g, (_, b) in enumerate(inner.split(np.zeros(len(tr)), y[tr]), 1):
                inner_rows.extend((r, f, int(i), g) for i in tr[b])
        outer_tab[f"outer_fold_repeat{r}"] = fold_of
        sizes = np.bincount(fold_of)[1:]
        print(f"repeat {r}: outer test-fold sizes {sizes.min()}-{sizes.max()}, all rows assigned: {bool((fold_of > 0).all())}")

    pd.DataFrame(outer_tab).to_csv(os.path.join(args.out, "outer_folds.csv"), index=False)
    pd.DataFrame(inner_rows, columns=["repeat", "outer_fold", "row_index", "inner_validation_fold"]) \
        .to_csv(os.path.join(args.out, "inner_folds.csv"), index=False)

    with open(os.path.join(args.out, "seeds.txt"), "w") as fh:
        fh.write(f"dataset: {args.dataset} (n = {n})\n")
        fh.write(f"scikit-learn version used to generate this file: {sklearn.__version__}\n")
        fh.write(f"repeats (seeds r): 0..{args.repeats - 1}\n")
        fh.write(f"outer split: StratifiedKFold(n_splits={args.outer}, shuffle=True, random_state=r)\n")
        fh.write(f"inner split: StratifiedKFold(n_splits={args.inner}, shuffle=True, "
                 f"random_state=1000*(r+1)+f), f = outer fold 1..{args.outer}\n")
        fh.write("GATv2 initialisation and training seed: r\n")
        fh.write("standard classifiers with a random state: random_state = r\n")
        fh.write("random encoder candidates: make_candidates(n_trials=24, seed=12345), identical in every repeat and fold\n")
    print(f"written to {args.out}: outer_folds.csv, inner_folds.csv, seeds.txt")


if __name__ == "__main__":
    main()