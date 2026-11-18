# ============================================================
# posthoc_metrics.py
#
# Extra metrics from the saved nested-CV outputs (no retraining):
#   PR-AUC (average precision) with paired bootstrap CIs, Brier score,
#   reliability (calibration) table + plot, and the ROC point with
#   sensitivity >= 80% / 90% (specificity read off the pooled outer-test scores).
#
# Usage:   python posthoc_metrics.py --out nested_cv_out [--boot 1000] [--seed 0]
#
# Notes
#  * Scores are the pooled outer-test scores stored in repeat_*.npz. Scores from different
#    outer folds come from different fitted models, so the operating points are DESCRIPTIVE ROC
#    readings, not thresholds chosen inside the training folds.
#  * LogReg / RandForest / GradBoost / kNN scores are predict_proba outputs. For the polling
#    models (GATv2-*, Raw-polling) the score is the weighted share of class-1 votes: it is not a
#    calibrated probability and takes few distinct values. SVM-RBF gives a decision function,
#    so it has no Brier score or calibration curve.
# ============================================================

import os
import sys
import glob
import argparse

import numpy as np
from sklearn.metrics import average_precision_score, roc_curve, brier_score_loss

PROB_MODELS = ["LogReg", "RandForest", "GradBoost", "kNN"]
VOTE_MODELS = ["GATv2-tuned", "GATv2-default", "Raw-polling"]
TARGETS = (0.8, 0.9)


def load_runs(out):
    files = sorted(glob.glob(os.path.join(out, "repeat_*.npz")),
                   key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    if not files:
        sys.exit(f"no repeat_*.npz found in {out}")
    return [np.load(p) for p in files]


def spec_at_sens(y, s, target):
    """First ROC point whose sensitivity reaches the target: (achieved sensitivity, specificity)."""
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    i = int(np.argmax(tpr >= target))
    return float(tpr[i]), float(1 - fpr[i])


def reliability(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    b = np.digitize(p, edges[1:-1])                 # 0 .. bins-1
    rows = []
    for i in range(bins):
        mk = b == i
        if mk.sum() > 0:
            rows.append((edges[i], edges[i + 1], int(mk.sum()), float(p[mk].mean()), float(y[mk].mean())))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="nested_cv_out")
    parser.add_argument("--boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    runs = load_runs(args.out)
    y = runs[0]["y"]
    n = len(y)
    R = len(runs)
    models = [str(m) for m in runs[0]["models"]]
    ref = "GATv2-tuned"
    sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
    print(f"{R} repeat(s) in {args.out}; n = {n}; class-1 prevalence = {y.mean():.3f}; "
          f"Brier of a constant-prevalence predictor = {y.mean() * (1 - y.mean()):.4f}\n")

    # ---------- per-repeat metrics ----------
    res = {}
    for m in models:
        pr = [average_precision_score(y, run[f"score__{m}"]) for run in runs]
        br = [brier_score_loss(y, run[f"score__{m}"]) for run in runs] if m in PROB_MODELS + VOTE_MODELS else None
        op = {t: [spec_at_sens(y, run[f"score__{m}"], t) for run in runs] for t in TARGETS}
        res[m] = dict(pr=pr, br=br, op=op)

    print("PR-AUC (average precision), Brier score, and ROC point with sensitivity >= target "
          "(mean over repeats; sd in brackets)")
    print("  Brier for the polling models (marked *) is a vote share, not a calibrated probability; "
          "SVM-RBF has none.")
    header = f"{'model':14s} {'PR-AUC':>14s} {'Brier':>14s}"
    for t in TARGETS:
        header += f" {'spec@sens>=' + str(int(t * 100)):>20s}"
    print(header)
    latex = []
    for m in models:
        r = res[m]
        pr_s = f"{np.mean(r['pr']):.3f} ({sd(r['pr']):.3f})"
        if r["br"] is None:
            br_s, br_l = "--", "--"
        else:
            star = "*" if m in VOTE_MODELS else ""
            br_s = f"{np.mean(r['br']):.3f} ({sd(r['br']):.3f}){star}"
            br_l = f"{np.mean(r['br']):.3f}" + ("$^{*}$" if star else "")
        cells, lcells = [], []
        for t in TARGETS:
            sens_a = 100 * np.mean([a for a, _ in r["op"][t]])
            spec_a = 100 * np.mean([b for _, b in r["op"][t]])
            cells.append(f"{spec_a:.1f} (sens {sens_a:.1f})")
            lcells.append(f"{spec_a:.1f}\\% ({sens_a:.1f}\\%)")
        print(f"{m:14s} {pr_s:>14s} {br_s:>14s} " + " ".join(f"{c:>20s}" for c in cells))
        latex.append(f"{m} & {pr_s.split(' ')[0]} & {br_l} & " + " & ".join(lcells) + " \\\\")

    # ---------- paired bootstrap CI for PR-AUC: GATv2-tuned minus each other model ----------
    rng = np.random.RandomState(args.seed)
    idx = rng.randint(0, n, size=(args.boot, n))
    ap_b = {m: np.zeros(args.boot) for m in models}
    for b in range(args.boot):
        ib = idx[b]
        yb = y[ib]
        for m in models:
            ap_b[m][b] = np.mean([average_precision_score(yb, run[f"score__{m}"][ib]) for run in runs])
    print(f"\nPaired PR-AUC difference, {ref} minus model, in PR-AUC x 100 "
          f"(95% bootstrap CI over instances, {args.boot} resamples; averaged over repeats)")
    latex_diff = []
    for m in models:
        if m == ref:
            continue
        d = 100 * (np.mean(res[ref]["pr"]) - np.mean(res[m]["pr"]))
        lo, hi = np.percentile(100 * (ap_b[ref] - ap_b[m]), [2.5, 97.5])
        print(f"  vs {m:14s} {d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        latex_diff.append(f"{m} & {d:+.2f} [{lo:+.2f}, {hi:+.2f}] \\\\")

    # ---------- reliability tables (pooled over repeats) ----------
    y_pool = np.concatenate([y] * R)
    print("\nReliability tables, pooled over repeats (bin, n, mean predicted, observed class-1 fraction)")
    curves = {}
    for m in PROB_MODELS + VOTE_MODELS:
        p_pool = np.concatenate([run[f"score__{m}"] for run in runs])
        rows = reliability(y_pool, p_pool)
        curves[m] = rows
        print(f"\n{m}" + ("  (vote share, not a calibrated probability)" if m in VOTE_MODELS else ""))
        for lo, hi, cnt, mp, of in rows:
            print(f"  [{lo:.1f}, {hi:.1f})  n={cnt:5d}  predicted={mp:.3f}  observed={of:.3f}")

    # ---------- optional plot ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        for m, rows in curves.items():
            if m == "GATv2-default":
                continue
            ax.plot([r[3] for r in rows], [r[4] for r in rows], marker="o", ms=4, lw=1.2, label=m)
        ax.set_xlabel("Mean predicted score")
        ax.set_ylabel("Observed fraction diabetic")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = os.path.join(args.out, "calibration.png")
        fig.savefig(path, dpi=300)
        print(f"\n(calibration plot saved to {path})")
    except Exception as e:
        print(f"\n(plot skipped: {e})")

    # ---------- LaTeX rows ----------
    with open(os.path.join(args.out, "posthoc_table_rows.tex"), "w") as fh:
        fh.write("% model & PR-AUC & Brier & spec (sens) at sens>=80% & spec (sens) at sens>=90%\n")
        fh.write("\n".join(latex) + "\n\n")
        fh.write(f"% PR-AUC x100, {ref} minus model, 95% CI\n")
        fh.write("\n".join(latex_diff) + "\n")
    print(f"(LaTeX rows written to {os.path.join(args.out, 'posthoc_table_rows.tex')})")


if __name__ == "__main__":
    main()