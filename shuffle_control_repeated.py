# shuffle_control_repeated.py
# Repeated label-shuffle control for both evaluation protocols.
# Sits next to gatv2_cv_validation.py and gatv2_weighted_polling.py (both untouched).
#
# Usage:  python shuffle_control_repeated.py [dataset] [n_perms] [start]
#   dataset: pima (default) | pima_orig
#   n_perms: end of the permutation range (default 20; runs seeds start..n_perms-1)
#   start:   first permutation seed to run (default 0; use it to resume an interrupted run)
# Per-permutation results are APPENDED to shuffle_rows_<dataset>.csv; delete that file
# before a fresh run, otherwise new rows are added to the old ones in the summary.

import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from gatv2_weighted_polling import load_dataset
from gatv2_cv_validation import DATASETS, leaky_protocol, clean_validate

name = sys.argv[1] if len(sys.argv) > 1 else "pima"
n_perms = int(sys.argv[2]) if len(sys.argv) > 2 else 20
start = int(sys.argv[3]) if len(sys.argv) > 3 else 0   # first permutation seed to run (resume support)
D = DATASETS[name]
CFG = D["cfg"]

X_raw, y, classes, _ = load_dataset(D["path"], D["features"], D["label"], sep=D["sep"],
                                    zero_as_missing_cols=D["zero_as_missing"])
nc = len(classes)
maj = 100 * np.max(np.bincount(y)) / len(y)
print(f"dataset={name} n={len(y)} config={CFG} majority={maj:.2f}% permutations={n_perms}", flush=True)

rows = []
import os
for s in range(start, n_perms):
    y_s = np.random.RandomState(s).permutation(y)
    pred, tr, te = leaky_protocol(X_raw, y_s, nc, CFG, eval_mode=D["leaky_eval"])
    old_acc = 100 * accuracy_score(y_s, pred)
    old_f1 = 100 * f1_score(y_s, pred, average="macro")
    old_tr = 100 * accuracy_score(y_s[tr], pred[tr])
    old_te = 100 * accuracy_score(y_s[te], pred[te])
    res = clean_validate(X_raw, y_s, nc, CFG, seeds=(0,), n_splits=D["n_splits"])
    cl_acc = 100 * accuracy_score(y_s, res[0]["p_gat"])
    cl_f1 = 100 * f1_score(y_s, res[0]["p_gat"], average="macro")
    rows.append((old_acc, old_f1, old_tr, old_te, cl_acc, cl_f1))
    with open(f"shuffle_rows_{name}.csv", "a") as fh:
        fh.write(f"{s},{old_acc},{old_f1},{old_tr},{old_te},{cl_acc},{cl_f1}\n")
    print(f"perm {s:2d}: old acc {old_acc:.2f} f1 {old_f1:.2f} (trained {old_tr:.2f} / never {old_te:.2f}) | "
          f"clean acc {cl_acc:.2f} f1 {cl_f1:.2f}", flush=True)

R = np.loadtxt(f"shuffle_rows_{name}.csv", delimiter=",", ndmin=2)[:, 1:]
n_perms = len(R)
names = ["old acc", "old macro-F1", "old acc (GAT-trained nodes)", "old acc (never-trained nodes)",
         "clean acc", "clean macro-F1"]
print(f"\nSummary over {n_perms} permutations (mean +/- sd, min-max); majority-class rate {maj:.2f}%")
for i, n in enumerate(names):
    print(f"  {n:32s}: {R[:, i].mean():.2f} +/- {R[:, i].std(ddof=1):.2f}   ({R[:, i].min():.2f}-{R[:, i].max():.2f})")
print(f"  permutations with old acc > majority : {(R[:, 0] > maj).sum()}/{n_perms}")
print(f"  permutations with clean acc > majority: {(R[:, 4] > maj).sum()}/{n_perms}")
np.save(f"shuffle_control_{name}.npy", R)
