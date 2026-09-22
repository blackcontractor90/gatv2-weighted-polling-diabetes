# ============================================================
# gatv2_weighted_polling.py
#
# Reconstruction of the pipeline described in:
#   "Graph Attention Network-Augmented Weighted Polling
#    Classification for Multiclass Dry Bean Variety Identification"
#    (Morsidi, Sections 3.2-3.4)
#
# Two stages:
#   1. GATv2 graph representation learning (Sec 3.2-3.3)
#      -> produces 16-dim relational embeddings
#   2. Weighted-polling KNN/Naive Bayes classifier (Sec 3.4)
#      -> ported from Classifier.java, generalized to N classes
#
# Reconstructed from the paper's spec since the original script
# could not be located. Values (lr, epochs, dims, dropout) are
# taken verbatim from the Methodology section; anything the paper
# does not pin down (e.g. exact PyG conv kwargs) is flagged with
# a comment rather than silently guessed.
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# STAGE 0: DATA LOADING
# ============================================================

def load_dataset(csv_path, feature_cols, label_col, sep=",", zero_as_missing_cols=None):
    """
    Generic tabular loader. For the bean paper:
        feature_cols = ["Area","Perimeter","MajorAxisLength","MinorAxisLength",
                         "ConvexArea","EquivDiameter","ShapeFactor1","ShapeFactor3"]
        label_col = "Class"
    For Pima diabetes:
        feature_cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                         "Insulin","BMI","DiabetesPedigreeFunction","Age"]
        label_col = "Outcome"
    sep: use whitespace regex for space-delimited files (e.g. TestBeanDataset1.txt).
    zero_as_missing_cols: subset of feature_cols where 0 is a missing-value marker,
        not a real physiological reading (Glucose/BloodPressure/SkinThickness/
        Insulin/BMI in Pima -- Pregnancies=0 and Outcome=0 are legitimate values,
        so they're deliberately excluded even though they can be zero).
        Zeros in these columns are converted to NaN here; imputation happens
        later in run_pipeline using train-set-only statistics (see impute_missing).
    """
    df = pd.read_csv(csv_path, sep=sep, engine="python" if sep != "," else "c")
    X = df[feature_cols].values.astype(np.float64)

    if zero_as_missing_cols:
        col_idx = {c: i for i, c in enumerate(feature_cols)}
        for c in zero_as_missing_cols:
            i = col_idx[c]
            n_zero = np.sum(X[:, i] == 0)
            X[np.where(X[:, i] == 0)[0], i] = np.nan
            print(f"  {c}: marked {n_zero} zero-values as missing")

    labels_raw = df[label_col].values
    classes = sorted(pd.unique(labels_raw))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[c] for c in labels_raw], dtype=np.int64)
    return X, y, classes, class_to_idx


def impute_missing(X, train_idx):
    """
    Fill NaNs with the per-column median computed from TRAIN rows only
    (mirrors the paper's Sec 3.1 rule of computing normalization stats
    from the training set only, applied here to imputation stats instead).
    Returns the imputed array; does not mutate X in place.
    """
    X = X.copy()
    n_missing_total = np.isnan(X).sum()
    if n_missing_total == 0:
        return X

    medians = {}
    for col in range(X.shape[1]):
        col_train_vals = X[train_idx, col]
        col_train_vals = col_train_vals[~np.isnan(col_train_vals)]
        if len(col_train_vals) == 0:
            continue  # column entirely missing in train split -- nothing sane to impute with
        median = np.median(col_train_vals)
        medians[col] = median
        col_nan_mask = np.isnan(X[:, col])
        X[col_nan_mask, col] = median

    print(f"  Imputed {int(n_missing_total)} missing values using train-set medians")
    return X, medians


def save_model_artifacts(out_prefix, model, X_std, y, scaler, feature_cols, classes,
                          k_graph, weighted_edges, zero_as_missing_cols, impute_medians):
    """
    Saves everything needed for infer_embedding.py to classify a brand-new,
    single raw-feature sample later, without re-running the whole training
    pipeline:
      {prefix}_model.pt        -- trained GATv2Embedder weights
      {prefix}_graph_x.npy     -- the TRAINING set's normalized feature matrix
                                    (the fixed graph a new node gets attached to)
      {prefix}_graph_y.npy     -- training labels, same order as graph_x
      {prefix}_config.json     -- scaler mean/scale, feature order, k_graph,
                                    weighted_edges, zero_as_missing_cols + medians,
                                    class list -- everything infer_embedding.py
                                    needs to preprocess a new raw sample identically
    """
    import json
    torch.save(model.state_dict(), f"{out_prefix}_model.pt")
    np.save(f"{out_prefix}_graph_x.npy", X_std)
    np.save(f"{out_prefix}_graph_y.npy", y)

    config = {
        "feature_cols": feature_cols,
        "classes": [str(c) for c in classes],
        "k_graph": k_graph,
        "weighted_edges": weighted_edges,
        "zero_as_missing_cols": zero_as_missing_cols or [],
        "impute_medians": {feature_cols[col]: med for col, med in (impute_medians or {}).items()},
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "in_channels": X_std.shape[1],
        "num_classes": len(classes),
    }
    with open(f"{out_prefix}_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Saved model artifacts: {out_prefix}_model.pt, _graph_x.npy, _graph_y.npy, _config.json")


# ============================================================
# STAGE 1: GRAPH CONSTRUCTION (Sec 3.2)
#
# "Edge weights were computed by inverse distance weighting
#  w = 1/(1+d(x,y)) and the graph was symmetrized by including
#  reverse edges."
# ============================================================

def build_knn_graph(features_std, k, weighted=True):
    """
    features_std: z-score normalized feature matrix (train-set stats only)
    k: neighborhood size (paper sweeps k in {3,5,7,10,15,20}; best was k=3)
    weighted: True = inverse-distance edge weights; False = unweighted
              control condition (paper Condition #4)
    """
    n = features_std.shape[0]
    knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    knn.fit(features_std)
    distances, indices = knn.kneighbors(features_std)

    edges = []
    weights = []
    for i in range(n):
        for j in range(1, k + 1):  # skip self (index 0)
            neighbor = indices[i, j]
            d = distances[i, j]
            edges.append((i, neighbor))
            edges.append((neighbor, i))  # symmetrize via reverse edge
            w = (1.0 / (1.0 + d)) if weighted else 1.0
            weights.append(w)
            weights.append(w)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float)
    return edge_index, edge_weight


# ============================================================
# STAGE 1: GATv2 ARCHITECTURE (Sec 3.3, verbatim from the paper)
#
# Layer 1: GATv2Conv(in=8,  out=32, heads=4, concat=True)  -> 128-dim, ELU, Dropout(0.3)
# Layer 2: GATv2Conv(in=128,out=32, heads=4, concat=True)  -> 128-dim, ELU, Dropout(0.3)
# Layer 3: GATv2Conv(in=128,out=16, heads=1, concat=False) -> 16-dim embedding
#
# A linear head (16 -> num_classes) is attached for supervised training and
# removed afterward, per "Once training was done, the linear classifier head
# was removed from the model... taken from Layer 3."
# ============================================================

class GATv2Embedder(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.3):
        super().__init__()
        # edge_dim=1 required for GATv2Conv to accept scalar edge weights at all;
        # without it PyG rejects edge_attr with an assertion error.
        self.conv1 = GATv2Conv(in_channels, 32, heads=4, concat=True, dropout=dropout, edge_dim=1)
        self.conv2 = GATv2Conv(128, 32, heads=4, concat=True, dropout=dropout, edge_dim=1)
        self.conv3 = GATv2Conv(128, 16, heads=1, concat=False, dropout=dropout, edge_dim=1)
        self.dropout = dropout
        # training-only head, discarded after fit (see extract_embeddings)
        self.classifier_head = nn.Linear(16, num_classes)

    def embed(self, x, edge_index, edge_weight=None):
        edge_attr = edge_weight.view(-1, 1) if edge_weight is not None else None

        h = self.conv1(x, edge_index, edge_attr=edge_attr)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv3(h, edge_index, edge_attr=edge_attr)  # 16-dim, no activation
        return h

    def forward(self, x, edge_index, edge_weight=None):
        h = self.embed(x, edge_index, edge_weight)
        logits = self.classifier_head(h)
        return logits


# ============================================================
# STAGE 1: TRAINING (Sec 3.3)
#
# "transductive training procedure on a complete graph with a
#  node-level train mask... class weighted cross-entropy loss...
#  Adam optimizer (lr=0.01, weight decay 5e-4) for 75 epochs."
# ============================================================

def train_gatv2(model, data, train_mask, num_classes, epochs=75, lr=0.01, weight_decay=5e-4):
    model = model.to(device)
    data = data.to(device)

    # inverse class-frequency weighting, as stated in the paper
    class_counts = torch.bincount(data.y[train_mask], minlength=num_classes).float()
    class_weights = (1.0 / class_counts.clamp(min=1)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 15 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}  loss={loss.item():.4f}")

    return model


def extract_embeddings(model, data):
    """Drop the classifier_head; return the 16-dim Layer-3 embeddings for every node."""
    model.eval()
    with torch.no_grad():
        emb = model.embed(data.x.to(device), data.edge_index.to(device),
                           data.edge_attr.to(device) if data.edge_attr is not None else None)
    return emb.cpu().numpy()


# ============================================================
# STAGE 2: WEIGHTED-POLLING KNN / NAIVE BAYES CLASSIFIER (Sec 3.4)
#
# Direct port of Classifier.java, generalized from the bean paper's
# hardcoded NO_CLASSES=7 to an arbitrary number of classes (needed
# for a binary target like diabetes diagnosis).
#
# Method 1 (KNN Normal):              +1 per neighbour
# Method 2 (KNN Weighted):             +(k_rank) for closer neighbours
# Method 3 (KNN Weighted-Squared):     +(k_rank)^2
# Method 4 (Naive Bayes freq-weighted): raw votes / class frequency
#
# Tie-break: same cascade as the Java version -- if arrayWinner()
# finds a tie, pull in the next-farthest neighbour and re-tally;
# if ties persist through all neighbours, fall back to the single
# nearest neighbour's class.
# ============================================================

class WeightedPollingClassifier:
    def __init__(self, embeddings, labels, num_classes):
        """
        embeddings: (n_samples, dim) array -- the GATv2 16-dim output
        labels: (n_samples,) int array, values in [0, num_classes)
        """
        self.X = np.asarray(embeddings, dtype=np.float64)
        self.y = np.asarray(labels, dtype=np.int64)
        self.num_classes = num_classes
        self.class_counts = np.bincount(self.y, minlength=num_classes)

    def _tally_update(self, tally, class_idx, method, max_score, rank):
        # rank is 0-indexed distance from nearest (0 = nearest)
        if method == 1:
            tally[class_idx] += 1
        elif method == 2:
            tally[class_idx] += (max_score - rank)
        elif method == 3:
            tally[class_idx] += (max_score - rank) ** 2
        elif method == 4:
            tally[class_idx] += 1  # normalised by frequency separately
        else:
            raise ValueError("method must be 1-4")

    def _winner(self, tally):
        """Return argmax index, or -1 if there's a tie for first place."""
        order = np.argsort(-tally)
        if len(tally) > 1 and tally[order[0]] == tally[order[1]]:
            return -1
        return int(order[0])

    def _classify_one(self, query_idx_or_vec, train_X, train_y, k, method, exclude_idx=None):
        if isinstance(query_idx_or_vec, (int, np.integer)):
            query = train_X[query_idx_or_vec] if exclude_idx is None else self.X[query_idx_or_vec]
        else:
            query = query_idx_or_vec

        dists = np.sum((train_X - query) ** 2, axis=1)  # squared Euclidean, matches Classifier.java
        order = np.argsort(dists)
        neighbour_idx = order[:k]
        neighbour_classes = train_y[neighbour_idx]

        if k == 1:
            return int(neighbour_classes[0])

        max_score = k
        tally = np.zeros(self.num_classes)
        for rank, cls in enumerate(neighbour_classes):
            self._tally_update(tally, cls, method, max_score, rank)

        if method == 4:
            scored = tally / np.maximum(self.class_counts, 1)
        else:
            scored = tally

        winner = self._winner(scored)

        # tie-break cascade: pull in progressively farther neighbours
        extend = k
        while winner == -1 and extend < len(order):
            cls = train_y[order[extend]]
            self._tally_update(tally, cls, method, max_score, extend)
            scored = tally / np.maximum(self.class_counts, 1) if method == 4 else tally
            winner = self._winner(scored)
            extend += 1

        if winner == -1:
            winner = int(neighbour_classes[0])  # fall back to nearest neighbour

        return winner

    def leave_one_out_accuracy(self, k, method):
        """Mirrors Classifier.java's oneLeftOutTest: remove each sample, classify
        against the rest, restore. Returns (accuracy, macro_f1, y_true, y_pred)."""
        n = len(self.X)
        preds = np.zeros(n, dtype=np.int64)

        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            train_X, train_y = self.X[mask], self.y[mask]
            preds[i] = self._classify_one(self.X[i], train_X, train_y, k, method)

        acc = accuracy_score(self.y, preds)
        f1 = f1_score(self.y, preds, average="macro")
        return acc, f1, self.y, preds

    def kfold_accuracy(self, k, method, n_splits=5, random_state=42):
        """Cheaper alternative to full leave-one-out for larger datasets
        (leave-one-out on the full Dry Bean set is O(n^2) and expensive;
        for a smaller dataset like Pima diabetes it's more tractable)."""
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        preds = np.zeros(len(self.y), dtype=np.int64)

        for train_idx, test_idx in skf.split(self.X, self.y):
            train_X, train_y = self.X[train_idx], self.y[train_idx]
            for ti in test_idx:
                preds[ti] = self._classify_one(self.X[ti], train_X, train_y, k, method)

        acc = accuracy_score(self.y, preds)
        f1 = f1_score(self.y, preds, average="macro")
        return acc, f1, self.y, preds


# ============================================================
# MAIN PIPELINE
# ============================================================

def export_embeddings(embeddings, y, classes, out_prefix):
    """
    Writes two files for the Java side to consume:
      {out_prefix}_embeddings.csv  -- one row per sample: emb_0..emb_15, label_index, class_name
      {out_prefix}_classes.csv     -- label_index, class_name mapping (redundant but explicit)
    Label is exported both as the integer index (what the classifier trains on) and the
    original class name (for readable output), since Java has no direct view of the
    Python-side class_to_idx dict.
    """
    n, dim = embeddings.shape
    cols = {f"emb_{i}": embeddings[:, i] for i in range(dim)}
    cols["label_index"] = y
    cols["class_name"] = [str(classes[label]) for label in y]
    out_df = pd.DataFrame(cols)
    emb_path = f"{out_prefix}_embeddings.csv"
    out_df.to_csv(emb_path, index=False)

    class_df = pd.DataFrame({"label_index": range(len(classes)), "class_name": [str(c) for c in classes]})
    class_path = f"{out_prefix}_classes.csv"
    class_df.to_csv(class_path, index=False)

    print(f"  Exported {n} embeddings ({dim}-dim) to {emb_path}")
    print(f"  Exported class mapping to {class_path}")
    return emb_path, class_path


def run_pipeline(csv_path, feature_cols, label_col, sep=",", zero_as_missing_cols=None,
                  k_graph=3, weighted_edges=False,
                  polling_k=10, polling_method=4, eval_mode="kfold",
                  export_prefix=None):
    """
    Defaults reflect the paper's best configuration (Condition #7):
    k_graph=3, unweighted edges, Naive Bayes frequency-weighted polling.
    """
    print("Loading data...")
    X, y, classes, class_to_idx = load_dataset(csv_path, feature_cols, label_col, sep=sep,
                                                zero_as_missing_cols=zero_as_missing_cols)
    num_classes = len(classes)
    print(f"{len(X)} samples, {len(feature_cols)} features, {num_classes} classes: {classes}")

    # 80/20 stratified split; normalize using TRAIN stats only, per Sec 3.1
    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=0.2, stratify=y, random_state=42
    )

    if zero_as_missing_cols:
        print("Imputing missing values (train-set medians only, no leakage)...")
        X, impute_medians = impute_missing(X, train_idx)
    else:
        impute_medians = None

    scaler = StandardScaler().fit(X[train_idx])
    X_std = scaler.transform(X)

    print(f"Building k={k_graph} graph (weighted={weighted_edges})...")
    edge_index, edge_weight = build_knn_graph(X_std, k=k_graph, weighted=weighted_edges)

    data = Data(
        x=torch.tensor(X_std, dtype=torch.float),
        y=torch.tensor(y, dtype=torch.long),
        edge_index=edge_index,
        edge_attr=edge_weight,
    )
    train_mask = torch.zeros(len(X), dtype=torch.bool)
    train_mask[train_idx] = True

    print("Training GATv2...")
    model = GATv2Embedder(in_channels=X_std.shape[1], num_classes=num_classes)
    model = train_gatv2(model, data, train_mask, num_classes)

    print("Extracting 16-dim embeddings (classifier head discarded)...")
    embeddings = extract_embeddings(model, data)

    print(f"Running weighted-polling classifier (method={polling_method}, k={polling_k})...")
    wpc = WeightedPollingClassifier(embeddings, y, num_classes)

    if eval_mode == "loo":
        acc, f1, y_true, y_pred = wpc.leave_one_out_accuracy(polling_k, polling_method)
    else:
        acc, f1, y_true, y_pred = wpc.kfold_accuracy(polling_k, polling_method)

    print(f"\nAccuracy: {acc*100:.2f}%   Macro-F1: {f1*100:.2f}%")
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=[str(c) for c in classes]))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    if export_prefix:
        print(f"\nExporting embeddings for Java...")
        export_embeddings(embeddings, y, classes, export_prefix)
        print(f"Saving model artifacts for live single-sample inference...")
        save_model_artifacts(export_prefix, model, X_std, y, scaler, feature_cols, classes,
                              k_graph, weighted_edges, zero_as_missing_cols, impute_medians)

    return model, embeddings, (acc, f1)


if __name__ == "__main__":
    BEAN_FEATURES = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
                      "ConvexArea", "EquivDiameter", "ShapeFactor1", "ShapeFactor3"]
    DIABETES_FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                          "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    DIABETES_ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "bean"

    if target == "bean":
        run_pipeline(
            csv_path="TestBeanDataset1.txt",
            feature_cols=BEAN_FEATURES,
            label_col="Class",
            sep=r"\s+",
            k_graph=3,
            weighted_edges=False,
            polling_k=10,
            polling_method=4,
            eval_mode="kfold",  # switch to "loo" to match the paper's exact protocol (slower)
            export_prefix="bean",
        )
    elif target == "diabetes":
        run_pipeline(
            csv_path="pima_diabetes.csv",
            feature_cols=DIABETES_FEATURES,
            label_col="Outcome",
            sep=",",
            zero_as_missing_cols=DIABETES_ZERO_AS_MISSING,
            k_graph=3,
            weighted_edges=False,
            polling_k=10,
            polling_method=4,
            eval_mode="kfold",
            export_prefix="diabetes",
        )