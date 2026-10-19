# ============================================================
# nested_cv_comparison.py
#
# Nested cross-validation comparison for the BNRC revision of the Pima paper.
# Sits next to gatv2_weighted_polling.py and gatv2_cv_validation.py (both untouched)
# and imports from them.
#
# What it does, per repeat (seed r = 0, 1, 2, ...):
#   OUTER loop : stratified k-fold (default 10), fold assignment fixed by r.
#                EVERY model below is scored on exactly these outer folds.
#   INNER loop : stratified k-fold (default 5) INSIDE each outer-training set.
#                All hyperparameters are chosen here, never on outer-test rows.
#   Preprocessing (median imputation + z-score) always uses the training rows of the
#   split it belongs to; the kNN graph is built over all nodes (transductive: held-out
#   FEATURES visible, held-out LABELS never), exactly as in the leakage-free protocol.
#
# Models (all tuned in the same inner loop, on the same inner folds):
#   GATv2-tuned   : GATv2 encoder + weighted polling. Joint random search over graph k,
#                   edge weighting, encoder width/heads/dropout/lr/weight decay/epochs
#                   AND the 20 polling configs. Candidate #0 is always the original
#                   untuned encoder, so the search space contains the paper's setting.
#   GATv2-default : the original untuned encoder (candidate #0); only the polling config
#                   is chosen in the inner loop.
#   Raw-polling   : weighted polling on z-scored features; polling config chosen in the
#                   inner loop (same 20 configs).
#   LogReg, SVM-RBF, RandForest, GradBoost, kNN : standard models with small grids.
#
# Outputs (per repeat, in --out): predictions + class-1 scores for every model, and the
# configuration chosen in every outer fold. The `summary` command then reports accuracy,
# macro-F1, sensitivity, specificity, AUROC (mean +/- sd over repeats), paired-difference
# 95% bootstrap CIs (GATv2-tuned minus each other model), McNemar p-values, and
# ready-to-paste LaTeX table rows.
#
# AUROC note: for the polling models the class-1 score is the (weighted) share of class-1
# votes among the k neighbours. It takes few distinct values, so its AUROC is coarse.
#
# Usage (from the folder holding the .py files and pima_diabetes.csv):
#   python nested_cv_comparison.py check                       (10-second self-test, run this first)
#   python nested_cv_comparison.py bench   [--trials 24]       (runtime estimate)
#   python nested_cv_comparison.py run     [--repeats 3] [--outer 10] [--inner 5]
#                                          [--trials 24] [--select acc|f1] [--out nested_cv_out]
#                                          [--start 0] [--force]
#   python nested_cv_comparison.py summary [--out nested_cv_out] [--boot 10000]
#
# `run` skips repeats whose result file already exists, so an interrupted run can simply
# be started again with the same command (use --force to redo repeats).
# ============================================================

import os
import sys
import glob
import json
import time
import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from gatv2_weighted_polling import (WeightedPollingClassifier, load_dataset,
                                    build_knn_graph, device)
from gatv2_cv_validation import DATASETS, preprocess, mcnemar_exact

warnings.filterwarnings("ignore")

POLL_KS = [3, 5, 7, 10, 15]
POLL_METHODS = [1, 2, 3, 4]
POLL_CFGS = [(k, m) for k in POLL_KS for m in POLL_METHODS]          # 20 polling configs

CORE_MODELS = ["GATv2-tuned", "GATv2-default", "Raw-polling"]
STD_MODELS = ["LogReg", "SVM-RBF", "RandForest", "GradBoost", "kNN"]
ALL_MODELS = CORE_MODELS + STD_MODELS

# Encoder / graph search space (random search; candidate #0 is always DEFAULT_CAND)
ENC_SPACE = dict(
    k_graph=[3, 5, 7, 10, 15],
    weighted=[False, True],
    hidden=[16, 32, 64],
    heads=[2, 4],
    dropout=[0.1, 0.3, 0.5],
    lr=[0.01, 0.005],
    weight_decay=[5e-4, 5e-3],
    epochs=[50, 75, 150],
)
DEFAULT_CAND = dict(k_graph=3, weighted=False, hidden=32, heads=4, dropout=0.3,
                    lr=0.01, weight_decay=5e-4, epochs=75)     # the paper's untuned encoder


def make_candidates(n_trials, seed=12345):
    """Candidate list (same for every fold/repeat): the default encoder first, then random draws."""
    rng = np.random.RandomState(seed)
    cands = [dict(DEFAULT_CAND)]
    seen = {tuple(sorted(DEFAULT_CAND.items()))}
    while len(cands) < n_trials:
        c = {name: vals[rng.randint(len(vals))] for name, vals in ENC_SPACE.items()}
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            cands.append(c)
    return cands


# ------------------------------------------------------------
# GATv2 encoder with tunable width / heads / dropout
# (defaults reproduce gatv2_weighted_polling.GATv2Embedder exactly)
# ------------------------------------------------------------

class TunableGATv2(nn.Module):
    def __init__(self, in_channels, num_classes, hidden=32, heads=4, dropout=0.3, emb_dim=16):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden, heads=heads, concat=True,
                               dropout=dropout, edge_dim=1)
        self.conv2 = GATv2Conv(hidden * heads, hidden, heads=heads, concat=True,
                               dropout=dropout, edge_dim=1)
        self.conv3 = GATv2Conv(hidden * heads, emb_dim, heads=1, concat=False,
                               dropout=dropout, edge_dim=1)
        self.dropout = dropout
        self.classifier_head = nn.Linear(emb_dim, num_classes)   # training-only head

    def embed(self, x, edge_index, edge_attr):
        h = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.conv2(h, edge_index, edge_attr=edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv3(h, edge_index, edge_attr=edge_attr)

    def forward(self, x, edge_index, edge_attr):
        return self.classifier_head(self.embed(x, edge_index, edge_attr))


def train_embed(X_std, y, train_idx, cand, seed, graph_cache, num_classes):
    """Train GATv2 from scratch (loss on train_idx nodes only); return (n, 16) embeddings.
    graph_cache: dict reused across candidates that share (k_graph, weighted) on the same X_std."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    key = (cand["k_graph"], cand["weighted"])
    if key not in graph_cache:
        ei, ew = build_knn_graph(X_std, k=cand["k_graph"], weighted=cand["weighted"])
        graph_cache[key] = (ei.to(device), ew.view(-1, 1).to(device))
    ei, ew = graph_cache[key]

    x = torch.tensor(X_std, dtype=torch.float, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    train_mask = torch.zeros(len(y), dtype=torch.bool, device=device)
    train_mask[torch.as_tensor(train_idx, device=device)] = True

    model = TunableGATv2(X_std.shape[1], num_classes, hidden=cand["hidden"],
                         heads=cand["heads"], dropout=cand["dropout"]).to(device)
    counts = torch.bincount(yt[train_mask], minlength=num_classes).float()
    class_weights = 1.0 / counts.clamp(min=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=cand["lr"],
                                 weight_decay=cand["weight_decay"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for _ in range(cand["epochs"]):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x, ei, ew)[train_mask], yt[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        emb = model.embed(x, ei, ew)
    return emb.cpu().numpy()


# ------------------------------------------------------------
# Weighted polling: predictions for many configs from one neighbour ordering,
# plus the class-1 vote share (for AUROC)
# ------------------------------------------------------------

def _classify_from_order(wpc, order, k, method):
    """Identical logic to WeightedPollingClassifier._classify_one, given a precomputed ordering."""
    yr = wpc.y
    nb = yr[order[:k]]
    if k == 1:
        return int(nb[0])
    tally = np.zeros(wpc.num_classes)
    for rank, cls in enumerate(nb):
        wpc._tally_update(tally, cls, method, k, rank)
    scored = tally / np.maximum(wpc.class_counts, 1) if method == 4 else tally
    winner = wpc._winner(scored)
    extend = k
    while winner == -1 and extend < len(order):
        wpc._tally_update(tally, yr[order[extend]], method, k, extend)
        scored = tally / np.maximum(wpc.class_counts, 1) if method == 4 else tally
        winner = wpc._winner(scored)
        extend += 1
    if winner == -1:
        winner = int(nb[0])
    return winner


def poll_all(emb, y, ref_idx, query_idx, num_classes, cfgs):
    """Predictions of every (k, method) in cfgs for query_idx, voting only among ref_idx nodes."""
    wpc = WeightedPollingClassifier(emb[ref_idx], y[ref_idx], num_classes)
    out = {c: np.empty(len(query_idx), dtype=np.int64) for c in cfgs}
    for qi, i in enumerate(query_idx):
        d = np.sum((wpc.X - emb[i]) ** 2, axis=1)       # same distance as _classify_one
        order = np.argsort(d)
        for (k, m) in cfgs:
            out[(k, m)][qi] = _classify_from_order(wpc, order, k, m)
    return out


def vote_share(emb, y, ref_idx, query_idx, k, method):
    """Weighted share of class-1 votes among the k nearest reference nodes (binary task)."""
    ref = emb[ref_idx].astype(np.float64)
    yr = y[ref_idx]
    counts = np.bincount(yr, minlength=2)
    ranks = np.arange(k)
    if method in (1, 4):
        w = np.ones(k)
    elif method == 2:
        w = (k - ranks).astype(float)
    else:
        w = ((k - ranks) ** 2).astype(float)
    share = np.empty(len(query_idx))
    for qi, i in enumerate(query_idx):
        order = np.argsort(np.sum((ref - emb[i]) ** 2, axis=1))[:k]
        nb = yr[order]
        s = np.array([w[nb == c].sum() for c in (0, 1)])
        if method == 4:
            s = s / np.maximum(counts, 1)
        tot = s.sum()
        share[qi] = s[1] / tot if tot > 0 else 0.5
    return share


# ------------------------------------------------------------
# Standard models
# ------------------------------------------------------------

def std_grids():
    return {
        "LogReg": [dict(C=C) for C in (0.01, 0.1, 1, 10, 100)],
        "SVM-RBF": [dict(C=C, gamma=g) for C in (0.1, 1, 10) for g in ("scale", 0.01, 0.1)],
        "RandForest": [dict(max_depth=d, min_samples_leaf=l) for d in (3, 5, None) for l in (1, 3, 5)],
        "GradBoost": [dict(n_estimators=n, learning_rate=lr, max_depth=d)
                      for n in (50, 100, 200) for lr in (0.03, 0.1) for d in (2, 3)],
        "kNN": [dict(n_neighbors=k, weights=w) for k in (3, 5, 7, 10, 15, 21, 31)
                for w in ("uniform", "distance")],
    }


def build_std(name, params, seed):
    if name == "LogReg":
        return LogisticRegression(max_iter=2000, **params)
    if name == "SVM-RBF":
        return SVC(kernel="rbf", **params)
    if name == "RandForest":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1, **params)
    if name == "GradBoost":
        return GradientBoostingClassifier(random_state=seed, **params)
    if name == "kNN":
        return KNeighborsClassifier(**params)
    raise ValueError(name)


def std_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


# ------------------------------------------------------------
# One repeat of nested CV
# ------------------------------------------------------------

def crit(y_true, y_pred, how):
    if how == "f1":
        return f1_score(y_true, y_pred, average="macro")
    return accuracy_score(y_true, y_pred)


def best_of(scores):
    """Key of the max score; the first maximum in insertion order wins (deterministic)."""
    best_key, best_val = None, -1.0
    for key, val in scores.items():
        if val > best_val:
            best_key, best_val = key, val
    return best_key, best_val


def run_repeat(r, X_raw, y, nc, cands, args, out_path):
    n = len(y)
    grids = std_grids()
    pred = {m: np.full(n, -1, dtype=np.int64) for m in ALL_MODELS}
    score = {m: np.full(n, np.nan) for m in ALL_MODELS}
    chosen = []
    outer = StratifiedKFold(n_splits=args.outer, shuffle=True, random_state=r)

    for f, (tr, te) in enumerate(outer.split(X_raw, y), 1):
        t0 = time.time()
        inner = StratifiedKFold(n_splits=args.inner, shuffle=True,
                                random_state=1000 * (r + 1) + f)
        inner_abs = [(tr[a], tr[b]) for a, b in inner.split(np.zeros(len(tr)), y[tr])]

        # ---------- inner loop: collect out-of-inner-fold predictions for every candidate ----------
        gat_inner = [{c: np.full(n, -1, dtype=np.int64) for c in POLL_CFGS} for _ in cands]
        raw_inner = {c: np.full(n, -1, dtype=np.int64) for c in POLL_CFGS}
        std_inner = {name: [np.full(n, -1, dtype=np.int64) for _ in g] for name, g in grids.items()}

        for (tri, vai) in inner_abs:
            Xs = preprocess(X_raw, tri)                 # stats from inner-training rows only
            gcache = {}
            for ci, c in enumerate(cands):
                emb = train_embed(Xs, y, tri, c, r, gcache, nc)
                P = poll_all(emb, y, tri, vai, nc, POLL_CFGS)
                for cfg in POLL_CFGS:
                    gat_inner[ci][cfg][vai] = P[cfg]
            P = poll_all(Xs, y, tri, vai, nc, POLL_CFGS)
            for cfg in POLL_CFGS:
                raw_inner[cfg][vai] = P[cfg]
            for name, grid in grids.items():
                for gi, params in enumerate(grid):
                    mdl = build_std(name, params, r).fit(Xs[tri], y[tri])
                    std_inner[name][gi][vai] = mdl.predict(Xs[vai])

        # ---------- selection on inner out-of-fold predictions (outer-training rows only) ----------
        gat_scores = {(ci, cfg): crit(y[tr], gat_inner[ci][cfg][tr], args.select)
                      for ci in range(len(cands)) for cfg in POLL_CFGS}
        (ci_best, cfg_best), inner_best = best_of(gat_scores)
        cfg_def, _ = best_of({cfg: gat_scores[(0, cfg)] for cfg in POLL_CFGS})
        cfg_raw, _ = best_of({cfg: crit(y[tr], raw_inner[cfg][tr], args.select) for cfg in POLL_CFGS})
        std_best = {}
        for name, grid in grids.items():
            gi, _ = best_of({gi: crit(y[tr], std_inner[name][gi][tr], args.select)
                             for gi in range(len(grid))})
            std_best[name] = gi

        # ---------- refit on the whole outer-training set, score the outer-test fold ----------
        Xs = preprocess(X_raw, tr)
        gcache, emb_cache = {}, {}

        def get_emb(ci):
            if ci not in emb_cache:
                emb_cache[ci] = train_embed(Xs, y, tr, cands[ci], r, gcache, nc)
            return emb_cache[ci]

        for name, ci, (k, m) in (("GATv2-tuned", ci_best, cfg_best), ("GATv2-default", 0, cfg_def)):
            emb = get_emb(ci)
            pred[name][te] = poll_all(emb, y, tr, te, nc, [(k, m)])[(k, m)]
            score[name][te] = vote_share(emb, y, tr, te, k, m)
        k, m = cfg_raw
        pred["Raw-polling"][te] = poll_all(Xs, y, tr, te, nc, [(k, m)])[(k, m)]
        score["Raw-polling"][te] = vote_share(Xs, y, tr, te, k, m)
        for name, grid in grids.items():
            mdl = build_std(name, grid[std_best[name]], r).fit(Xs[tr], y[tr])
            pred[name][te] = mdl.predict(Xs[te])
            score[name][te] = std_score(mdl, Xs[te])

        chosen.append(dict(
            outer_fold=f, tuned_candidate=int(ci_best), tuned_candidate_params=cands[ci_best],
            tuned_polling=list(cfg_best), tuned_inner_score=float(inner_best),
            default_polling=list(cfg_def), raw_polling=list(cfg_raw),
            std={name: grids[name][gi] for name, gi in std_best.items()},
            seconds=round(time.time() - t0, 1)))
        msg = "  ".join(f"{mm}={100 * accuracy_score(y[te], pred[mm][te]):.1f}"
                        for mm in ("GATv2-tuned", "GATv2-default", "Raw-polling", "LogReg", "SVM-RBF"))
        print(f"  repeat {r} outer fold {f}/{args.outer} ({time.time() - t0:.0f}s): test acc {msg}", flush=True)

    np.savez(out_path, y=y, models=np.array(ALL_MODELS),
             **{f"pred__{m}": pred[m] for m in ALL_MODELS},
             **{f"score__{m}": score[m] for m in ALL_MODELS})
    with open(out_path.replace(".npz", "_chosen.json"), "w") as fh:
        json.dump(chosen, fh, indent=1, default=str)
    accs = {m: 100 * accuracy_score(y, pred[m]) for m in ALL_MODELS}
    print(f"repeat {r} done: " + ", ".join(f"{m} {a:.2f}" for m, a in accs.items()), flush=True)


# ------------------------------------------------------------
# Summary: metrics, paired-difference CIs, McNemar, LaTeX rows
# ------------------------------------------------------------

def conf_counts(y, p):
    tp = np.sum((y == 1) & (p == 1)); fn = np.sum((y == 1) & (p == 0))
    tn = np.sum((y == 0) & (p == 0)); fp = np.sum((y == 0) & (p == 1))
    return tp, fn, tn, fp


def metrics_from_counts(tp, fn, tn, fp):
    """Vectorised over arrays of counts. Returns dict of acc, f1 (macro), sens, spec."""
    tp, fn, tn, fp = (np.asarray(a, dtype=float) for a in (tp, fn, tn, fp))
    tot = tp + fn + tn + fp
    f1_1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    f1_0 = 2 * tn / np.maximum(2 * tn + fn + fp, 1)
    return dict(acc=(tp + tn) / tot, f1=(f1_0 + f1_1) / 2,
                sens=tp / np.maximum(tp + fn, 1), spec=tn / np.maximum(tn + fp, 1))


def boot_counts(y_b, p_b):
    """y_b, p_b: (B, n) bootstrap-resampled labels/predictions -> counts arrays of length B."""
    tp = ((y_b == 1) & (p_b == 1)).sum(1); fn = ((y_b == 1) & (p_b == 0)).sum(1)
    tn = ((y_b == 0) & (p_b == 0)).sum(1); fp = ((y_b == 0) & (p_b == 1)).sum(1)
    return tp, fn, tn, fp


def summarize(args):
    files = sorted(glob.glob(os.path.join(args.out, "repeat_*.npz")),
                   key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    files = [p for p in files if not p.endswith("_chosen.json")]
    if not files:
        sys.exit(f"no repeat_*.npz found in {args.out}")
    runs = [np.load(p) for p in files]
    y = runs[0]["y"]
    models = [str(m) for m in runs[0]["models"]]
    R, n = len(runs), len(y)
    print(f"{R} repeat(s) found in {args.out}; n = {n}\n")

    # ---- per-repeat metrics ----
    per = {m: {k: [] for k in ("acc", "f1", "sens", "spec", "auc")} for m in models}
    for run in runs:
        for m in models:
            p, s = run[f"pred__{m}"], run[f"score__{m}"]
            mt = metrics_from_counts(*conf_counts(y, p))
            for k in ("acc", "f1", "sens", "spec"):
                per[m][k].append(float(mt[k]))
            per[m]["auc"].append(roc_auc_score(y, s))
    sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")

    print("Nested-CV results, mean (sd) over repeats, %  [sd = fold assignment + initialisation only]")
    print(f"{'model':14s} {'acc':>14s} {'macro-F1':>14s} {'sens':>14s} {'spec':>14s} {'AUROC':>14s}")
    latex = []
    for m in models:
        cells = []
        for k in ("acc", "f1", "sens", "spec", "auc"):
            v = 100 * np.array(per[m][k])
            cells.append(f"{v.mean():.2f} ({sd(v):.2f})")
        print(f"{m:14s} " + " ".join(f"{c:>14s}" for c in cells))
        latex.append(f"{m} & " + " & ".join(
            f"{100 * np.mean(per[m][k]):.2f}\\% ($\\pm${100 * sd(per[m][k]):.2f}\\%)" if k != "auc"
            else f"{np.mean(per[m][k]):.3f} ($\\pm${sd(per[m][k]):.3f})"
            for k in ("acc", "f1", "sens", "spec", "auc")) + " \\\\")

    # ---- paired bootstrap CIs: GATv2-tuned minus each other model ----
    rng = np.random.RandomState(args.seed)
    idx = rng.randint(0, n, size=(args.boot, n))
    y_b = y[idx]
    boot = {}
    for m in models:
        accum = {k: np.zeros(args.boot) for k in ("acc", "f1", "sens", "spec")}
        for run in runs:
            mt = metrics_from_counts(*boot_counts(y_b, run[f"pred__{m}"][idx]))
            for k in accum:
                accum[k] += mt[k] / R
        boot[m] = accum
    point = {m: {k: float(np.mean(per[m][k])) for k in ("acc", "f1", "sens", "spec")} for m in models}

    # AUROC bootstrap (fewer replicates: roc_auc_score is slower)
    B_auc = min(args.boot, 1000)
    auc_boot = {m: np.zeros(B_auc) for m in models}
    for b in range(B_auc):
        ib = idx[b]
        yb = y[ib]
        for m in models:
            auc_boot[m][b] = np.mean([roc_auc_score(yb, run[f"score__{m}"][ib]) for run in runs])

    ref = "GATv2-tuned"
    print(f"\nPaired difference, {ref} minus other model, percentage points (95% bootstrap CI over instances)")
    print("[averaged over repeats; resamples the 768 instances, so it does NOT capture variance from "
          "re-drawing the training data]")
    print(f"{'vs':14s} {'acc':>24s} {'macro-F1':>24s} {'sens':>24s} {'spec':>24s} {'AUROC':>24s}")
    latex_diff = []
    for m in models:
        if m == ref:
            continue
        cells, lcells = [], []
        for k in ("acc", "f1", "sens", "spec"):
            d = 100 * (point[ref][k] - point[m][k])
            lo, hi = np.percentile(100 * (boot[ref][k] - boot[m][k]), [2.5, 97.5])
            cells.append(f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            lcells.append(f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        d = 100 * (np.mean(per[ref]["auc"]) - np.mean(per[m]["auc"]))
        lo, hi = np.percentile(100 * (auc_boot[ref] - auc_boot[m]), [2.5, 97.5])
        cells.append(f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        lcells.append(f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        print(f"{m:14s} " + " ".join(f"{c:>24s}" for c in cells))
        latex_diff.append(f"{m} & " + " & ".join(lcells) + " \\\\")

    # ---- McNemar per repeat ----
    for other in ("Raw-polling", "GATv2-default"):
        print(f"\nExact McNemar, {ref} vs {other} (per repeat: tuned-only-correct vs other-only-correct, p)")
        for i, run in enumerate(runs):
            a, b, p = mcnemar_exact(y, run[f"pred__{ref}"], run[f"pred__{other}"])
            print(f"  repeat {i}: {a} vs {b}, p = {p:.4f}")

    # ---- what the inner loop chose ----
    chosen_all = []
    for p in files:
        cp = p.replace(".npz", "_chosen.json")
        if os.path.exists(cp):
            chosen_all += json.load(open(cp))
    if chosen_all:
        nf = len(chosen_all)
        n_def = sum(c["tuned_candidate"] == 0 for c in chosen_all)
        print(f"\nInner-loop choices over {nf} outer folds: default encoder chosen in {n_def}; "
              f"tuned (non-default) encoder chosen in {nf - n_def}")
        for key in ("k_graph", "weighted", "hidden", "heads", "dropout", "lr", "weight_decay", "epochs"):
            vals = {}
            for c in chosen_all:
                v = str(c["tuned_candidate_params"][key]); vals[v] = vals.get(v, 0) + 1
            print(f"  {key:13s}: " + ", ".join(f"{v}:{cnt}" for v, cnt in sorted(vals.items())))
        pm = {}
        for c in chosen_all:
            v = f"k={c['tuned_polling'][0]},m={c['tuned_polling'][1]}"; pm[v] = pm.get(v, 0) + 1
        print("  polling       : " + ", ".join(f"{v}:{cnt}" for v, cnt in sorted(pm.items())))

    # ---- write text + LaTeX rows ----
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "nested_cv_table_rows.tex"), "w") as fh:
        fh.write("% Main table rows: model & accuracy & macro-F1 & sensitivity & specificity & AUROC\n")
        fh.write("\n".join(latex) + "\n\n")
        fh.write(f"% Paired difference rows ({ref} minus model), pp, 95% CI: acc & macro-F1 & sens & spec & AUROC\n")
        fh.write("\n".join(latex_diff) + "\n")
    print(f"\n(LaTeX rows written to {os.path.join(args.out, 'nested_cv_table_rows.tex')})")


# ------------------------------------------------------------
# Timing helper
# ------------------------------------------------------------

def bench(args, X_raw, y, nc, cands):
    outer = StratifiedKFold(n_splits=args.outer, shuffle=True, random_state=0)
    tr, te = next(iter(outer.split(X_raw, y)))
    Xs = preprocess(X_raw, tr)
    gcache, times = {}, []
    for ci, c in enumerate(cands):
        t0 = time.time()
        train_embed(Xs, y, tr, c, 0, gcache, nc)
        times.append(time.time() - t0)
    sweep = sum(times)
    t0 = time.time()
    P = poll_all(Xs, y, tr, te, nc, POLL_CFGS)
    t_poll = time.time() - t0
    t0 = time.time()
    for name, grid in std_grids().items():
        for params in grid:
            build_std(name, params, 0).fit(Xs[tr], y[tr])
    t_std = time.time() - t0
    scale = (len(tr) / args.inner) / len(te)          # inner validation fold vs outer test fold size
    t_poll_inner = t_poll * scale * len(cands)
    per_outer = args.inner * (sweep + t_poll_inner + t_std) + 2 * (sum(times) / len(times))
    print(f"device={device}  candidates={len(cands)}")
    print(f"one GATv2 training: {np.mean(times):.1f}s mean (min {min(times):.1f}s, max {max(times):.1f}s)")
    print(f"one inner split, all candidates: {sweep:.0f}s training + ~{t_poll_inner:.0f}s polling; "
          f"standard-model grids: {t_std:.0f}s")
    print(f"estimated per outer fold: {per_outer / 60:.1f} min; "
          f"per repeat ({args.outer} outer folds): {per_outer * args.outer / 3600:.2f} h")


# ------------------------------------------------------------
# Self-check against the original project code
# ------------------------------------------------------------

def selfcheck(X_raw, y, nc):
    """(1) default encoder here vs gatv2_cv_validation.gat_embed_fn; (2) poll_all vs poll()."""
    from gatv2_cv_validation import gat_embed_fn, poll
    tr, te = next(iter(StratifiedKFold(n_splits=10, shuffle=True, random_state=0).split(X_raw, y)))
    Xs = preprocess(X_raw, tr)
    a = train_embed(Xs, y, tr, DEFAULT_CAND, 0, {}, nc)
    b = gat_embed_fn(Xs, y, tr, DEFAULT_CAND["k_graph"], DEFAULT_CAND["weighted"], 0)
    print(f"max |embedding difference|, default encoder here vs gat_embed_fn: {np.abs(a - b).max():.2e}")
    print("  (expect 0 or ~1e-6 on CPU; small float differences are possible on GPU)")
    P = poll_all(b, y, tr, te, nc, POLL_CFGS)
    bad = sum(int(np.sum(P[(k, m)] != poll(b, y, tr, te, k, m, nc))) for (k, m) in POLL_CFGS)
    print(f"polling predictions differing from original poll(): {bad} of {len(POLL_CFGS) * len(te)} (expect 0)")


# ------------------------------------------------------------
# Command line
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "bench", "run", "summary"])
    ap.add_argument("--dataset", default="pima")
    ap.add_argument("--repeats", type=int, default=3, help="number of repeats (seeds 0..repeats-1)")
    ap.add_argument("--start", type=int, default=0, help="first repeat seed to run")
    ap.add_argument("--outer", type=int, default=10)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--trials", type=int, default=24, help="GATv2 candidates incl. the default encoder")
    ap.add_argument("--select", choices=["acc", "f1"], default="acc", help="inner selection criterion")
    ap.add_argument("--out", default="nested_cv_out")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    args = ap.parse_args()

    if args.mode == "summary":
        summarize(args)
        return

    D = DATASETS[args.dataset]
    X_raw, y, classes, _ = load_dataset(D["path"], D["features"], D["label"], sep=D["sep"],
                                        zero_as_missing_cols=D["zero_as_missing"])
    nc = len(classes)
    if nc != 2:
        sys.exit("this script is written for the binary Pima task")
    cands = make_candidates(args.trials)
    print(f"dataset={args.dataset} n={len(y)} outer={args.outer} inner={args.inner} "
          f"candidates={len(cands)} select={args.select} device={device}", flush=True)

    if args.mode == "check":
        selfcheck(X_raw, y, nc)
        return
    if args.mode == "bench":
        bench(args, X_raw, y, nc, cands)
        return

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "candidates.json"), "w") as fh:
        json.dump(cands, fh, indent=1)
    for r in range(args.start, args.repeats):
        path = os.path.join(args.out, f"repeat_{r}.npz")
        if os.path.exists(path) and not args.force:
            print(f"repeat {r}: {path} exists, skipping (use --force to redo)")
            continue
        print(f"=== repeat {r} ===", flush=True)
        run_repeat(r, X_raw, y, nc, cands, args, path)


if __name__ == "__main__":
    main()
