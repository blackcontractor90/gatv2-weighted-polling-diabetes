# ============================================================
# gatv2_cv_validation.py   (v2: Pima + Dry Bean)
#
# Leakage-free validation for the GATv2 + weighted-polling pipeline.
# Sits next to gatv2_weighted_polling.py and imports from it (that file is untouched).
#
# Rules enforced in EVERY fold of the clean protocol:
#   1. Imputation medians and z-score parameters come from the fold's training rows only.
#   2. The kNN graph is built over all nodes (transductive: held-out FEATURES are visible
#      to message passing, held-out LABELS never are).
#   3. GATv2 is trained from scratch in the fold, with the loss computed on the fold's
#      training nodes only.
#   4. Polling votes come from the fold's training nodes only (class counts too).
#
# Usage (from the folder holding both .py files and the data files):
#   python gatv2_cv_validation.py <mode> [dataset]
#
#   mode:     sweep     clean 5-fold hyperparameter search (Pima grid; slow on bean)
#             sweep_raw same clean 5-fold search for the RAW-FEATURE baseline (no GATv2; fast)
#             validate  clean repeated k-fold x 5 seeds: GATv2 vs raw-feature baseline + McNemar
#             leaky     the OLD protocol (encoder trained once on 80%, all nodes evaluated)
#             control   label-shuffle control for both protocols
#   dataset:  pima (default; the clean-search winner) | pima_orig (the original winner) | bean
#
# Examples:
#   python gatv2_cv_validation.py leaky bean
#   python gatv2_cv_validation.py validate bean
# ============================================================

import sys
import itertools
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from gatv2_weighted_polling import WeightedPollingClassifier, load_dataset


# ------------------------------------------------------------
# Dataset registry (edit paths/settings here)
# ------------------------------------------------------------

BEAN_FEATURES = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
                 "ConvexArea", "EquivDiameter", "ShapeFactor1", "ShapeFactor3"]
PIMA_FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

DATASETS = {
    "pima": dict(
        path="pima_diabetes.csv", features=PIMA_FEATURES, label="Outcome", sep=",",
        zero_as_missing=["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
        cfg=dict(k_graph=3, weighted=False, k_poll=10, method=1),   # clean-search winner (GATv2)
        raw_cfg=dict(k_poll=15, method=1),                          # raw baseline's own best polling (sweep_raw)
        n_splits=10, leaky_eval="loo",
    ),
    # the ORIGINAL winner (k_graph=7, polling k=15, method 1): reproduces the manuscript's original-protocol numbers
    "pima_orig": dict(
        path="pima_diabetes.csv", features=PIMA_FEATURES, label="Outcome", sep=",",
        zero_as_missing=["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
        cfg=dict(k_graph=7, weighted=False, k_poll=15, method=1),
        raw_cfg=dict(k_poll=15, method=1),
        n_splits=10, leaky_eval="loo",
    ),
    "bean": dict(
        path="TestBeanDataset1.txt", features=BEAN_FEATURES, label="Class", sep=r"\s+",
        zero_as_missing=None,
        cfg=dict(k_graph=3, weighted=False, k_poll=10, method=4),   # Condition #7
        n_splits=5, leaky_eval="kfold",                             # matches Sec 4.5 protocol
    ),
}


# ------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------

def preprocess(X_raw, train_idx):
    """Median-impute (if any NaN) and z-score using statistics from train_idx rows ONLY."""
    X = X_raw.copy()
    for c in range(X.shape[1]):
        col = X[train_idx, c]
        col = col[~np.isnan(col)]
        if len(col) > 0:
            X[np.isnan(X[:, c]), c] = np.median(col)
    scaler = StandardScaler().fit(X[train_idx])
    return scaler.transform(X)


def gat_embed_fn(X_std, y, train_idx, k_graph, weighted_edges, seed):
    """Train GATv2 from scratch with the loss on train_idx nodes only; return (n, 16) embeddings."""
    import io
    import contextlib
    import torch
    from torch_geometric.data import Data
    from gatv2_weighted_polling import (build_knn_graph, GATv2Embedder,
                                        train_gatv2, extract_embeddings)

    torch.manual_seed(seed)
    np.random.seed(seed)
    num_classes = int(y.max()) + 1
    edge_index, edge_weight = build_knn_graph(X_std, k=k_graph, weighted=weighted_edges)
    data = Data(x=torch.tensor(X_std, dtype=torch.float),
                y=torch.tensor(y, dtype=torch.long),
                edge_index=edge_index, edge_attr=edge_weight)
    train_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[torch.as_tensor(train_idx)] = True

    model = GATv2Embedder(in_channels=X_std.shape[1], num_classes=num_classes)
    with contextlib.redirect_stdout(io.StringIO()):      # silence per-epoch prints
        model = train_gatv2(model, data, train_mask, num_classes)
    return extract_embeddings(model, data)


def poll(emb, y, ref_idx, query_idx, k, method, num_classes):
    """Classify query_idx nodes using votes from ref_idx nodes only (class counts from ref_idx only)."""
    wpc = WeightedPollingClassifier(emb[ref_idx], y[ref_idx], num_classes)
    return np.array([wpc._classify_one(emb[i], wpc.X, wpc.y, k, method) for i in query_idx],
                    dtype=np.int64)


def mcnemar_exact(y, pred_a, pred_b):
    """Exact McNemar test on paired correctness. Returns (a_only_correct, b_only_correct, p_value)."""
    a_ok, b_ok = pred_a == y, pred_b == y
    n_a = int(np.sum(a_ok & ~b_ok))
    n_b = int(np.sum(~a_ok & b_ok))
    n = n_a + n_b
    if n == 0:
        return n_a, n_b, 1.0
    try:
        from scipy.stats import binomtest
        p = binomtest(min(n_a, n_b), n, 0.5).pvalue
    except ImportError:                                   # older SciPy
        from scipy.stats import binom_test
        p = binom_test(min(n_a, n_b), n, 0.5)
    return n_a, n_b, float(p)


# ------------------------------------------------------------
# Clean protocols
# ------------------------------------------------------------

def clean_sweep(X_raw, y, num_classes, graph_grid, poll_k_grid, methods,
                n_splits=5, seed=0, embed_fn=gat_embed_fn):
    """
    Leakage-free version of the configuration search. GATv2 is trained once per
    (graph config, fold) on that fold's training nodes, then ALL polling configs are scored
    on the held-out nodes from the same embeddings.
    graph_grid: list of (k_graph, weighted_edges). Returns a list sorted by accuracy.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(X_raw, y))
    rows = []
    for (kg, w) in graph_grid:
        preds = {(k, m): np.full(len(y), -1, dtype=np.int64) for k in poll_k_grid for m in methods}
        for tr, te in folds:
            X_std = preprocess(X_raw, tr)
            emb = embed_fn(X_std, y, tr, kg, w, seed)
            for k in poll_k_grid:
                for m in methods:
                    preds[(k, m)][te] = poll(emb, y, tr, te, k, m, num_classes)
        for (k, m), p in preds.items():
            rows.append(dict(k_graph=kg, weighted=w, k_poll=k, method=m,
                             acc=accuracy_score(y, p), f1=f1_score(y, p, average="macro")))
        print(f"  finished graph config k_graph={kg}, weighted={w}", flush=True)
    return sorted(rows, key=lambda r: -r["acc"])


def clean_validate(X_raw, y, num_classes, cfg, seeds=(0, 1, 2, 3, 4), n_splits=10,
                   embed_fn=gat_embed_fn, verbose=False, raw_cfg=None):
    """
    Repeated stratified k-fold with per-fold retraining, plus a raw-feature baseline evaluated
    on the identical folds. cfg = dict(k_graph, weighted, k_poll, method).
    raw_cfg = dict(k_poll, method) for the baseline (defaults to cfg's polling settings),
    so the baseline can use its own best polling configuration.
    Returns per-seed dicts holding both models' predictions.
    """
    rc = raw_cfg or cfg
    out = []
    for s in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s)
        p_gat = np.full(len(y), -1, dtype=np.int64)
        p_raw = np.full(len(y), -1, dtype=np.int64)
        for f, (tr, te) in enumerate(skf.split(X_raw, y), 1):
            X_std = preprocess(X_raw, tr)
            emb = embed_fn(X_std, y, tr, cfg["k_graph"], cfg["weighted"], s)
            p_gat[te] = poll(emb, y, tr, te, cfg["k_poll"], cfg["method"], num_classes)
            p_raw[te] = poll(X_std, y, tr, te, rc["k_poll"], rc["method"], num_classes)
            if verbose:
                print(f"  seed {s}  fold {f}/{n_splits} done", flush=True)
        out.append(dict(seed=s, p_gat=p_gat, p_raw=p_raw))
    return out


def report_validation(y, results, classes):
    gat_acc = [accuracy_score(y, r["p_gat"]) for r in results]
    gat_f1 = [f1_score(y, r["p_gat"], average="macro") for r in results]
    raw_acc = [accuracy_score(y, r["p_raw"]) for r in results]
    raw_f1 = [f1_score(y, r["p_raw"], average="macro") for r in results]
    sd = lambda v: np.std(v, ddof=1) if len(v) > 1 else float("nan")
    print(f"GATv2 : acc {100*np.mean(gat_acc):.2f}% (sd {100*sd(gat_acc):.2f})  "
          f"macro-F1 {100*np.mean(gat_f1):.2f}% (sd {100*sd(gat_f1):.2f})")
    print(f"Raw   : acc {100*np.mean(raw_acc):.2f}% (sd {100*sd(raw_acc):.2f})  "
          f"macro-F1 {100*np.mean(raw_f1):.2f}% (sd {100*sd(raw_f1):.2f})")
    print(f"Gain  : acc {100*(np.mean(gat_acc)-np.mean(raw_acc)):+.2f} pts   "
          f"macro-F1 {100*(np.mean(gat_f1)-np.mean(raw_f1)):+.2f} pts")
    print("\nPaired exact McNemar per repetition (GATv2-only-correct vs raw-only-correct):")
    for r in results:
        a, b, p = mcnemar_exact(y, r["p_gat"], r["p_raw"])
        print(f"  seed {r['seed']}: {a} vs {b}, p = {p:.4f}")
    r0 = results[0]
    names = [str(c) for c in classes]
    print(f"\nPer-class report, seed {r0['seed']} (GATv2):")
    print(classification_report(y, r0["p_gat"], target_names=names, digits=4))
    print(confusion_matrix(y, r0["p_gat"]))
    print(f"\nPer-class report, seed {r0['seed']} (raw-feature baseline):")
    print(classification_report(y, r0["p_raw"], target_names=names, digits=4))
    print(confusion_matrix(y, r0["p_raw"]))


# ------------------------------------------------------------
# The OLD protocol, reproduced for comparison
# ------------------------------------------------------------

def leaky_protocol(X_raw, y, num_classes, cfg, seed=0, embed_fn=gat_embed_fn,
                   eval_mode="loo"):
    """
    What gatv2_weighted_polling.run_pipeline does: GATv2 trained ONCE on a fixed 80% train mask,
    then polling evaluation over ALL nodes (eval_mode 'loo' or 'kfold', as in run_pipeline).
    ~80% of the evaluated nodes had their labels used to train the encoder.
    Returns (predictions for all nodes, train_idx, test_idx).
    """
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    X_std = preprocess(X_raw, train_idx)
    emb = embed_fn(X_std, y, train_idx, cfg["k_graph"], cfg["weighted"], seed)
    wpc = WeightedPollingClassifier(emb, y, num_classes)
    if eval_mode == "loo":
        _, _, _, pred = wpc.leave_one_out_accuracy(cfg["k_poll"], cfg["method"])
    else:
        _, _, _, pred = wpc.kfold_accuracy(cfg["k_poll"], cfg["method"])
    return pred, train_idx, test_idx


def report_leaky(y, pred, train_idx, test_idx):
    print(f"  all nodes            : acc {100*accuracy_score(y, pred):.2f}%  "
          f"macro-F1 {100*f1_score(y, pred, average='macro'):.2f}%")
    print(f"  nodes in GAT train   : acc {100*accuracy_score(y[train_idx], pred[train_idx]):.2f}%  (n={len(train_idx)})")
    print(f"  nodes never trained  : acc {100*accuracy_score(y[test_idx], pred[test_idx]):.2f}%  (n={len(test_idx)})")


# ------------------------------------------------------------
# Command line
# ------------------------------------------------------------

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    name = sys.argv[2] if len(sys.argv) > 2 else "pima"
    if name not in DATASETS:
        sys.exit(f"unknown dataset '{name}' (choose from {list(DATASETS)})")
    D = DATASETS[name]
    CFG = D["cfg"]

    X_raw, y, classes, _ = load_dataset(D["path"], D["features"], D["label"], sep=D["sep"],
                                        zero_as_missing_cols=D["zero_as_missing"])
    nc = len(classes)
    RAW = D.get("raw_cfg")
    print(f"dataset={name}  n={len(y)}  classes={nc}  GATv2 config={CFG}  raw-baseline polling={RAW or 'same as GATv2'}", flush=True)

    if mode == "sweep":
        graph_grid = list(itertools.product([3, 5, 7, 10, 15], [False, True]))
        rows = clean_sweep(X_raw, y, nc, graph_grid, poll_k_grid=[3, 5, 7, 10, 15],
                           methods=[1, 2, 3, 4])
        print("\nTop 10 (clean 5-fold search):")
        for i, r in enumerate(rows[:10], 1):
            print(f"{i:2d}  k_graph={r['k_graph']:2d} weighted={str(r['weighted']):5s} "
                  f"k_poll={r['k_poll']:2d} method={r['method']}  "
                  f"acc={100*r['acc']:.2f}%  f1={100*r['f1']:.2f}%")
    elif mode == "sweep_raw":
        # raw-feature baseline gets the same polling search as GATv2 (identity "encoder")
        rows = clean_sweep(X_raw, y, nc, [(0, False)], poll_k_grid=[3, 5, 7, 10, 15],
                           methods=[1, 2, 3, 4], embed_fn=lambda Xs, yy, tr, kg, w, sd: Xs)
        print("\nTop 10 raw-feature configurations (clean 5-fold search):")
        for i, r in enumerate(rows[:10], 1):
            print(f"{i:2d}  k_poll={r['k_poll']:2d} method={r['method']}  "
                  f"acc={100*r['acc']:.2f}%  f1={100*r['f1']:.2f}%")
    elif mode == "validate":
        res = clean_validate(X_raw, y, nc, CFG, n_splits=D["n_splits"], verbose=True, raw_cfg=RAW)
        np.savez(f"cv_results_{name}.npz", y=y, seeds=[r["seed"] for r in res],
                 p_gat=np.array([r["p_gat"] for r in res]),
                 p_raw=np.array([r["p_raw"] for r in res]))
        print(f"(saved predictions to cv_results_{name}.npz)\n")
        report_validation(y, res, classes)
    elif mode == "leaky":
        pred, tr, te = leaky_protocol(X_raw, y, nc, CFG, eval_mode=D["leaky_eval"])
        report_leaky(y, pred, tr, te)
    elif mode == "control":
        y_shuf = np.random.RandomState(0).permutation(y)   # labels carry no signal
        print(f"Majority-class rate: {100*np.max(np.bincount(y))/len(y):.2f}%")
        print("Old protocol on SHUFFLED labels:")
        pred, tr, te = leaky_protocol(X_raw, y_shuf, nc, CFG, eval_mode=D["leaky_eval"])
        report_leaky(y_shuf, pred, tr, te)
        print(f"Clean protocol on SHUFFLED labels ({D['n_splits']}-fold, seed 0):")
        res = clean_validate(X_raw, y_shuf, nc, CFG, seeds=(0,), n_splits=D["n_splits"])
        print(f"  acc {100*accuracy_score(y_shuf, res[0]['p_gat']):.2f}%")
    else:
        print(__doc__)
