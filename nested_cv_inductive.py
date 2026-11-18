# ============================================================
# nested_cv_inductive.py
#
# Inductive control for the BNRC revision (Reviewer 3, point 3; Reviewer 4).
# Sits next to nested_cv_comparison.py and imports from it.
#
# Difference from the primary (transductive) protocol, and nothing else:
#   * the kNN graph and the GATv2 encoder are built/fitted on TRAINING rows only;
#   * each held-out row is attached at inference time: it receives edges FROM its k nearest
#     training rows (training -> held-out only), so held-out rows never influence training rows
#     and never influence each other;
#   * the held-out features are therefore NOT visible during graph construction or training.
# Folds (outer seed r, inner seed 1000*(r+1)+f), the 24 candidates, the 20 polling configs,
# the selection criterion and the refit step are the same as in nested_cv_comparison.py.
#
# Raw-polling and the five standard classifiers do not use the graph, so their predictions are
# copied from the primary run (same folds). Only GATv2-tuned and GATv2-default are recomputed.
# The output files have the same layout as the primary run, so `summary` and
# `posthoc_metrics.py` work on the output folder unchanged.
#
# Usage (from the folder holding the .py files and pima_diabetes.csv):
#   python nested_cv_inductive.py check
#   python nested_cv_inductive.py run [--primary nested_cv_out] [--out nested_cv_out_ind]
#                                     [--repeats 3] [--start 0] [--select acc|f1] [--force]
#   python nested_cv_comparison.py summary --out nested_cv_out_ind
# ============================================================

import os
import sys
import json
import time
import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors

from gatv2_weighted_polling import build_knn_graph, load_dataset, device
from gatv2_cv_validation import DATASETS, preprocess
from nested_cv_comparison import (TunableGATv2, make_candidates, poll_all, vote_share,
                                  crit, best_of, POLL_CFGS, ALL_MODELS, DEFAULT_CAND)

warnings.filterwarnings("ignore")


def train_embed_inductive(X_std, y, train_idx, query_idx, cand, seed, cache, num_classes):
    """Train GATv2 on a kNN graph over train_idx rows ONLY (loss on all of them), then attach
    query_idx rows (edges from their k nearest training rows into them) and embed everything.
    Returns an (n, 16) array filled for train_idx and query_idx rows (zeros elsewhere).
    cache: dict reused across candidates that share (k_graph, weighted) for the same split."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    k, wtd = cand["k_graph"], cand["weighted"]
    Xtr = X_std[train_idx]
    m = len(train_idx)
    nq = len(query_idx)

    if (k, wtd) not in cache:
        ei, ew = build_knn_graph(Xtr, k=k, weighted=wtd)                 # training graph (local indices)
        nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(Xtr)
        dist, nbr = nbrs.kneighbors(X_std[query_idx])                    # (nq, k)
        src = nbr.reshape(-1)                                            # training node (source)
        dst = m + np.repeat(np.arange(nq), k)                            # held-out node (target)
        aei = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
        aw = torch.tensor((1.0 / (1.0 + dist.reshape(-1))) if wtd else np.ones(len(src)),
                          dtype=torch.float)
        cache[(k, wtd)] = (ei.to(device), ew.view(-1, 1).to(device),
                           aei.to(device), aw.view(-1, 1).to(device))
    ei, ew, aei, aw = cache[(k, wtd)]

    x_tr = torch.tensor(Xtr, dtype=torch.float, device=device)
    yt = torch.tensor(y[train_idx], dtype=torch.long, device=device)
    model = TunableGATv2(X_std.shape[1], num_classes, hidden=cand["hidden"],
                         heads=cand["heads"], dropout=cand["dropout"]).to(device)
    counts = torch.bincount(yt, minlength=num_classes).float()
    criterion = nn.CrossEntropyLoss(weight=1.0 / counts.clamp(min=1))
    optimizer = torch.optim.Adam(model.parameters(), lr=cand["lr"], weight_decay=cand["weight_decay"])
    for _ in range(cand["epochs"]):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x_tr, ei, ew), yt)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        x_all = torch.cat([x_tr, torch.tensor(X_std[query_idx], dtype=torch.float, device=device)], 0)
        e_all = torch.cat([ei, aei], 1)
        w_all = torch.cat([ew, aw], 0)
        emb = model.embed(x_all, e_all, w_all).cpu().numpy()
    out = np.zeros((len(y), emb.shape[1]))
    out[train_idx] = emb[:m]
    out[query_idx] = emb[m:]
    return out


def selfcheck(X_raw, y, nc):
    """Held-out rows must not influence training rows or each other: embeddings of the training
    rows and of query set A must be identical whether A is attached alone or together with B."""
    tr, te = next(iter(StratifiedKFold(n_splits=10, shuffle=True, random_state=0).split(X_raw, y)))
    Xs = preprocess(X_raw, tr)
    A, B = te[:30], te[30:]
    e1 = train_embed_inductive(Xs, y, tr, A, DEFAULT_CAND, 0, {}, nc)
    e2 = train_embed_inductive(Xs, y, tr, np.concatenate([A, B]), DEFAULT_CAND, 0, {}, nc)
    print(f"max |difference| in training-row embeddings  : {np.abs(e1[tr] - e2[tr]).max():.2e}")
    print(f"max |difference| in query-set-A embeddings   : {np.abs(e1[A] - e2[A]).max():.2e}")
    print("  (expect 0 or ~1e-6: held-out rows do not affect training rows or each other)")


def run_repeat(r, X_raw, y, nc, cands, args, primary, out_path):
    n = len(y)
    pred = {m: primary[f"pred__{m}"].copy() for m in ALL_MODELS}
    score = {m: primary[f"score__{m}"].copy() for m in ALL_MODELS}
    for m in ("GATv2-tuned", "GATv2-default"):
        pred[m][:] = -1
        score[m][:] = np.nan
    outer_fold = np.full(n, -1, dtype=np.int64)
    chosen = []
    outer = StratifiedKFold(n_splits=args.outer, shuffle=True, random_state=r)

    for f, (tr, te) in enumerate(outer.split(X_raw, y), 1):
        outer_fold[te] = f
        t0 = time.time()
        inner = StratifiedKFold(n_splits=args.inner, shuffle=True, random_state=1000 * (r + 1) + f)
        inner_abs = [(tr[a], tr[b]) for a, b in inner.split(np.zeros(len(tr)), y[tr])]

        # ---------- inner loop (inductive: only inner-training rows are in the graph) ----------
        gat_inner = [{c: np.full(n, -1, dtype=np.int64) for c in POLL_CFGS} for _ in cands]
        for (tri, vai) in inner_abs:
            Xs = preprocess(X_raw, tri)
            cache = {}
            for ci, c in enumerate(cands):
                emb = train_embed_inductive(Xs, y, tri, vai, c, r, cache, nc)
                P = poll_all(emb, y, tri, vai, nc, POLL_CFGS)
                for cfg in POLL_CFGS:
                    gat_inner[ci][cfg][vai] = P[cfg]

        gat_scores = {(ci, cfg): crit(y[tr], gat_inner[ci][cfg][tr], args.select)
                      for ci in range(len(cands)) for cfg in POLL_CFGS}
        (ci_best, cfg_best), inner_best = best_of(gat_scores)
        cfg_def, _ = best_of({cfg: gat_scores[(0, cfg)] for cfg in POLL_CFGS})
        inner_by_candidate = []
        for ci in range(len(cands)):
            accs = [accuracy_score(y[tr], gat_inner[ci][cfg][tr]) for cfg in POLL_CFGS]
            f1s = [f1_score(y[tr], gat_inner[ci][cfg][tr], average="macro") for cfg in POLL_CFGS]
            inner_by_candidate.append([float(max(accs)), float(max(f1s))])

        # ---------- refit on the outer-training rows only, attach the outer-test rows ----------
        Xs = preprocess(X_raw, tr)
        emb_cache = {}

        def get_emb(ci):
            if ci not in emb_cache:
                emb_cache[ci] = train_embed_inductive(Xs, y, tr, te, cands[ci], r, {}, nc)
            return emb_cache[ci]

        for name, ci, (k, m) in (("GATv2-tuned", ci_best, cfg_best), ("GATv2-default", 0, cfg_def)):
            emb = get_emb(ci)
            pred[name][te] = poll_all(emb, y, tr, te, nc, [(k, m)])[(k, m)]
            score[name][te] = vote_share(emb, y, tr, te, k, m)

        chosen.append(dict(
            outer_fold=f, tuned_candidate=int(ci_best), tuned_candidate_params=cands[ci_best],
            tuned_polling=list(cfg_best), tuned_inner_score=float(inner_best),
            default_polling=list(cfg_def), inner_by_candidate=inner_by_candidate,
            seconds=round(time.time() - t0, 1)))
        msg = "  ".join(f"{mm}={100 * accuracy_score(y[te], pred[mm][te]):.1f}"
                        for mm in ("GATv2-tuned", "GATv2-default", "Raw-polling", "LogReg"))
        print(f"  repeat {r} outer fold {f}/{args.outer} ({time.time() - t0:.0f}s): test acc {msg}", flush=True)

    np.savez(out_path, y=y, models=np.array(ALL_MODELS), outer_fold=outer_fold,
             **{f"pred__{m}": pred[m] for m in ALL_MODELS},
             **{f"score__{m}": score[m] for m in ALL_MODELS})
    with open(out_path.replace(".npz", "_chosen.json"), "w") as fh:
        json.dump(chosen, fh, indent=1, default=str)
    accs = {m: 100 * accuracy_score(y, pred[m]) for m in ("GATv2-tuned", "GATv2-default")}
    print(f"repeat {r} done (inductive): " + ", ".join(f"{m} {a:.2f}" for m, a in accs.items()), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "run"])
    ap.add_argument("--dataset", default="pima")
    ap.add_argument("--primary", default="nested_cv_out", help="folder with the primary run's repeat_*.npz")
    ap.add_argument("--out", default="nested_cv_out_ind")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--outer", type=int, default=10)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--select", choices=["acc", "f1"], default="acc")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

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

    os.makedirs(args.out, exist_ok=True)
    for r in range(args.start, args.repeats):
        path = os.path.join(args.out, f"repeat_{r}.npz")
        if os.path.exists(path) and not args.force:
            print(f"repeat {r}: {path} exists, skipping (use --force to redo)")
            continue
        ppath = os.path.join(args.primary, f"repeat_{r}.npz")
        if not os.path.exists(ppath):
            sys.exit(f"primary results not found: {ppath}")
        primary = np.load(ppath)
        if not np.array_equal(primary["y"], y):
            sys.exit("labels in the primary run differ from the dataset loaded here")
        print(f"=== repeat {r} (inductive) ===", flush=True)
        run_repeat(r, X_raw, y, nc, cands, args, primary, path)


if __name__ == "__main__":
    main()