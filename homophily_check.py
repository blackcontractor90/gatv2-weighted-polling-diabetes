# ============================================================
# homophily_check.py
#
# Reviewer 3, point 10: label homophily of the kNN graph for each k_graph.
# Descriptive diagnostic of the graph the GATv2 encoder would see (symmetrized kNN graph over
# all 768 instances, built on z-scored features, as in build_knn_graph). Labels are used only
# to compute the statistic, never to fit anything. Imputation/z-score statistics here come from
# all rows (a descriptive check, not a validation estimate).
#
# Reported per k_graph:
#   edge homophily  = share of undirected edges joining two same-label nodes
#   null            = the same statistic with labels randomly permuted (mean and 95% range)
#   adjusted        = (h - sum_c p_c^2) / (1 - sum_c p_c^2), p_c = degree-weighted class share
#                     (0 = no better than random labelling of this graph, 1 = perfectly homophilous)
#   class-wise      = share of class-c nodes' kNN neighbours that are class c
#
# Usage:  python homophily_check.py [--perms 1000] [--seed 0]
# ============================================================

import argparse

import numpy as np
from sklearn.neighbors import NearestNeighbors

from gatv2_weighted_polling import load_dataset
from gatv2_cv_validation import DATASETS, preprocess

K_GRID = [3, 5, 7, 10, 15]


def knn_pairs(X, k):
    """Directed kNN lists (self skipped, as in build_knn_graph) and the unique undirected edges."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    _, idx = nn.kneighbors(X)
    src = np.repeat(np.arange(len(X)), k)
    dst = idx[:, 1:].reshape(-1)
    und = np.unique(np.sort(np.stack([src, dst], axis=1), axis=1), axis=0)
    return src, dst, und


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pima")
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    D = DATASETS[args.dataset]
    X_raw, y, classes, _ = load_dataset(D["path"], D["features"], D["label"], sep=D["sep"],
                                        zero_as_missing_cols=D["zero_as_missing"])
    X = preprocess(X_raw, np.arange(len(y)))
    n = len(y)
    prev = np.bincount(y) / n
    rng = np.random.RandomState(args.seed)
    perms = [rng.permutation(y) for _ in range(args.perms)]
    print(f"\nn = {n}; class prevalence = {np.round(prev, 3).tolist()}; "
          f"same-label chance for two random nodes = {np.sum(prev ** 2):.3f}\n")
    print(f"{'k_graph':>7s} {'edges':>6s} {'edge hom.':>10s} {'null mean [95% range]':>24s} "
          f"{'adjusted':>9s} {'class-0 nbrs':>13s} {'class-1 nbrs':>13s}")

    latex = []
    for k in K_GRID:
        src, dst, und = knn_pairs(X, k)
        a, b = und[:, 0], und[:, 1]
        h = float(np.mean(y[a] == y[b]))
        null = np.array([np.mean(yp[a] == yp[b]) for yp in perms])
        lo, hi = np.percentile(null, [2.5, 97.5])
        deg = np.bincount(np.concatenate([a, b]), minlength=n)
        share = np.array([deg[y == c].sum() for c in range(len(prev))]) / deg.sum()
        adj = (h - np.sum(share ** 2)) / (1 - np.sum(share ** 2))
        cw = [float(np.mean(y[dst][y[src] == c] == c)) for c in range(len(prev))]
        print(f"{k:7d} {len(und):6d} {h:10.3f} {null.mean():8.3f} [{lo:.3f}, {hi:.3f}] "
              f"{adj:9.3f} {cw[0]:13.3f} {cw[1]:13.3f}")
        latex.append(f"{k} & {len(und)} & {h:.3f} & {null.mean():.3f} [{lo:.3f}, {hi:.3f}] & "
                     f"{adj:.3f} & {cw[0]:.3f} & {cw[1]:.3f} \\\\")

    print("\nLaTeX rows: k_graph & undirected edges & edge homophily & null mean [95% range] "
          "& adjusted & class-0 neighbours & class-1 neighbours")
    print("\n".join(latex))


if __name__ == "__main__":
    main()