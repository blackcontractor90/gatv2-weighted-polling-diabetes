# ============================================================
# nested_cv_missingness.py
#
# Sensitivity analysis for Reviewer 3, point 4 (median-imputed zeros may create artificial clusters
# in the feature-similarity graph).
#
# The primary nested-CV protocol is run again, unchanged (same folds, same 24 candidates, same
# accuracy criterion, transductive graph), with ONE change: after the usual training-fold median
# imputation and z-scoring, a 0/1 missingness indicator is appended for every feature that has
# missing values (Glucose, BloodPressure, SkinThickness, Insulin, BMI -> 5 extra columns, 13 in
# total). Every model, graph-based or not, receives the same 13 columns, and the kNN graph is built
# on all 13, so the indicators can separate imputed rows from measured ones. The indicators are not
# z-scored (they stay 0/1), so a rare indicator (BMI, 11 rows) does not become a large outlier.
#
# It reuses nested_cv_comparison.py unchanged by replacing its `preprocess` function, so the
# command line is the same:
#   python nested_cv_missingness.py run --out nested_cv_out_miss
#   python nested_cv_missingness.py summary --out nested_cv_out_miss
#   python posthoc_metrics.py --out nested_cv_out_miss        (optional)
# One process runs the three repeats in sequence (about 3 hours); a finished repeat is skipped if
# the command is run again.
# ============================================================

import numpy as np

import nested_cv_comparison as ncc
from gatv2_cv_validation import preprocess as _base_preprocess


def preprocess_with_indicators(X_raw, train_idx):
    """Training-fold median imputation + z-score (as in the primary protocol), then 0/1 missingness
    indicators for the features that contain missing values."""
    X_std = _base_preprocess(X_raw, train_idx)
    missing = np.isnan(X_raw)
    cols = [c for c in range(X_raw.shape[1]) if missing[:, c].any()]
    return np.hstack([X_std, missing[:, cols].astype(float)])


ncc.preprocess = preprocess_with_indicators

if __name__ == "__main__":
    ncc.main()