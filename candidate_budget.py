# ============================================================
# candidate_budget.py
#
# Reviewer 3, point 8: is 24 encoder candidates enough, or could the negative result be due to
# under-searching? Uses the per-candidate inner-loop scores saved by the edited run_repeat
# (`inner_by_candidate` in repeat_*_chosen.json: for every outer fold and every candidate, the best
# inner out-of-fold accuracy and best macro-F1 over the 20 polling configurations).
#
# For a budget of j candidates (the default encoder first, then j-1 random ones), the curve is the
# best inner score found within that budget, averaged over the outer folds, and over random
# orderings of the 23 random candidates (so it does not depend on one lucky ordering).
#
# Read it as: does the inner score still rise when the budget doubles? The inner score is
# optimistic (it is the maximum of many noisy estimates, and is also the quantity that was
# maximised), so the curve shows diminishing returns of the search, not outer-test performance.
#
# Usage:  python candidate_budget.py --out nested_cv_out_f1 [--perms 2000] [--seed 0]
# ============================================================

import os
import glob
import json
import argparse

import numpy as np


def load(out):
    rows, winners = [], []
    for p in sorted(glob.glob(os.path.join(out, "repeat_*_chosen.json"))):
        for fold in json.load(open(p)):
            if "inner_by_candidate" not in fold:
                raise SystemExit(f"{p} has no inner_by_candidate: it was produced before the run_repeat edit")
            rows.append(np.array(fold["inner_by_candidate"], dtype=float))   # (C, 2)
            winners.append(int(fold["tuned_candidate"]))
    if not rows:
        raise SystemExit(f"no repeat_*_chosen.json found in {out}")
    return np.stack(rows), np.array(winners)                                 # (F, C, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="nested_cv_out_f1")
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A, winners = load(args.out)
    F, C, _ = A.shape
    print(f"{F} outer folds, {C} candidates (candidate 0 = original encoder) in {args.out}\n")

    rng = np.random.RandomState(args.seed)
    curves = np.zeros((args.perms, C, 2))
    for i in range(args.perms):
        order = np.concatenate([[0], 1 + rng.permutation(C - 1)])
        curves[i] = np.maximum.accumulate(A[:, order, :], axis=1).mean(axis=0)
    mean = curves.mean(axis=0) * 100
    lo, hi = np.percentile(curves, [5, 95], axis=0) * 100

    print("Best inner score found within a budget of j candidates (%, mean over folds and random orderings)")
    print(f"{'j':>3s} {'inner acc':>10s} {'[5%,95% orderings]':>22s} {'inner macro-F1':>15s}")
    budgets = sorted(set([1, 2, 4, 8, 12, 16, 20, C]))
    latex = []
    for j in budgets:
        i = j - 1
        print(f"{j:3d} {mean[i, 0]:10.2f} [{lo[i, 0]:.2f}, {hi[i, 0]:.2f}] {mean[i, 1]:15.2f}")
        latex.append(f"{j} & {mean[i, 0]:.2f} & {mean[i, 1]:.2f} \\\\")

    h = C // 2
    print(f"\nGain from doubling the budget {h} -> {C}: inner accuracy {mean[C - 1, 0] - mean[h - 1, 0]:+.2f} points, "
          f"inner macro-F1 {mean[C - 1, 1] - mean[h - 1, 1]:+.2f} points")
    print(f"Gain from the last 4 candidates ({C - 4} -> {C}): inner accuracy {mean[C - 1, 0] - mean[C - 5, 0]:+.2f}, "
          f"inner macro-F1 {mean[C - 1, 1] - mean[C - 5, 1]:+.2f}")
    thirds = [int(np.sum((winners >= a) & (winners < b)))
              for a, b in ((0, 1), (1, 1 + (C - 1) // 3), (1 + (C - 1) // 3, 1 + 2 * (C - 1) // 3),
                           (1 + 2 * (C - 1) // 3, C))]
    print(f"Selected candidate index per outer fold: default (0): {thirds[0]}; "
          f"first third of the random ones: {thirds[1]}; middle third: {thirds[2]}; last third: {thirds[3]}")

    print("\nLaTeX rows: budget j & best inner accuracy (%) & best inner macro-F1 (%)")
    print("\n".join(latex))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(1, C + 1)
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        for k, name in ((0, "Inner accuracy"), (1, "Inner macro-F1")):
            ax.plot(x, mean[:, k], marker="o", ms=3, label=name)
            ax.fill_between(x, lo[:, k], hi[:, k], alpha=0.2)
        ax.set_xlabel("Number of encoder candidates searched")
        ax.set_ylabel("Best inner score (%)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(args.out, "candidate_budget.png")
        fig.savefig(path, dpi=300)
        print(f"\n(plot saved to {path})")
    except Exception as e:
        print(f"\n(plot skipped: {e})")


if __name__ == "__main__":
    main()