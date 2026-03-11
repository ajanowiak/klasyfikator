# train_time_agnostic.py

# train_motif_enrichment.py
#
# Trains time-agnostic Random Forest classifiers using the motif enrichment
# difference matrix as features (neural_labels or unfiltered, controlled by --filter_labels).
#
# Key differences from train.ipynb:
#   - Features: motif enrichment difference matrix (neural or unfiltered), NOT data_diff
#   - Missing loops (rows with any NaN) are dropped before training
#   - Only Random Forest, only time-agnostic models
#   - Tissues are processed in parallel (ProcessPoolExecutor)
#   - Records mean CV AUCROC to a summary CSV for later comparison
#
# Usage:
#   python train_motif_enrichment.py --filter_labels True   # neural_labels features
#   python train_motif_enrichment.py --filter_labels False  # unfiltered features

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score

from utils import print_timestamp


WINDOWS   = ["06-08", "10-12", "14-16"]
TISSUES   = ["Neuroblasts", "Neurons", "Glia"]
N_SPLITS  = 10
RF_PARAMS = dict(n_estimators=300, random_state=0, n_jobs=-1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def compose_windows_enrichment(tissue: str, feature_dir_template: str, label_tag: str) -> tuple:
    """
    Concatenate enrichment matrices across windows, align with tissue labels,
    drop NaN rows, and build a composite stratification vector.

    Args:
        tissue: tissue label, e.g. "Neuroblasts"
        feature_dir_template: f-string template with {window} placeholder,
                              e.g. "results/training_data/neural_labels/hrs{window}"

    Returns:
        X (pd.DataFrame), y (pd.Series), composite (np.ndarray of codes)
    """
    Xs, ys = [], []

    for idx, w in enumerate(WINDOWS):
        feature_dir  = feature_dir_template.format(window=w)
        enrich_path  = os.path.join(feature_dir, f"motif_enrichment_hrs{w}.csv")
        y_path       = f"results/training_data/{label_tag}/hrs{w}/y_{tissue}.csv"

        if not os.path.exists(enrich_path):
            raise FileNotFoundError(f"Enrichment file not found: {enrich_path}")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"Label file not found: {y_path}")

        X_w = pd.read_csv(enrich_path, index_col=0)
        y_w = pd.read_csv(y_path,      index_col=0).iloc[:, 0]

        # Align on shared loops
        shared = X_w.index.intersection(y_w.index)
        X_w = X_w.loc[shared]
        y_w = y_w.loc[shared]

        # Drop loops (rows) that have any NaN feature
        before = len(X_w)
        X_w = X_w.dropna(axis=0)
        after  = len(X_w)
        if before != after:
            print_timestamp(
                f"  [{tissue}] hrs{w}: dropped {before - after} loops with NaN features "
                f"({after} remaining)"
            )
        y_w = y_w.loc[X_w.index]

        # Tag with window index for stratification
        X_w["_window"] = idx
        Xs.append(X_w)
        ys.append(y_w)

    X = pd.concat(Xs, axis=0)
    y = pd.concat(ys, axis=0)

    composite = pd.Categorical(list(zip(X["_window"], y))).codes
    X = X.drop(columns=["_window"])

    return X, y, composite


# ---------------------------------------------------------------------------
# Training for a single tissue
# ---------------------------------------------------------------------------

def train_tissue(
    tissue: str,
    feature_dir_template: str,
    label_tag: str,          # "neural_labels" or "unfiltered" — used in paths
    n_splits: int = N_SPLITS,
    params: dict = RF_PARAMS,
) -> dict:
    """
    Cross-validated time-agnostic RF training for one tissue.

    Returns a dict with tissue name and CV metrics.
    """
    print_timestamp(f"[{tissue}] Starting training (label_tag={label_tag})...")

    X, y, composite = compose_windows_enrichment(tissue, feature_dir_template, label_tag)
    print_timestamp(f"[{tissue}] Data shape after NaN drop: {X.shape}, positives: {y.sum()}")

    classifier = RandomForestClassifier(**params)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    probs, trues, accs = [], [], []
    fold_index = {}

    for i, (train_idx, test_idx) in enumerate(skf.split(X, composite)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        fold_index[i] = (train_idx, test_idx)

        clf = clone(classifier)
        clf.fit(X_train, y_train)

        p = clf.predict_proba(X_test)[:, 1]
        probs.append(p)
        trues.append(y_test.values)
        accs.append(accuracy_score(y_test, (p > 0.5).astype(int)))

    # --- ROC metrics ---
    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)

    fprs, tprs, roc_aucs = [], [], []
    for true, prob in zip(trues, probs):
        fpr, tpr, _ = roc_curve(true, prob)
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(auc(fpr, tpr))

    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs = []
    for idx in range(n_splits):
        interp_tpr = np.interp(mean_fpr, fprs[idx], tprs[idx])
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

    mean_tpr      = np.mean(interp_tprs, axis=0)
    mean_tpr[-1]  = 1.0
    mean_auc      = auc(mean_fpr, mean_tpr)
    std_auc       = np.std(roc_aucs)
    tprs_upper    = np.minimum(mean_tpr + np.std(interp_tprs, axis=0), 1)
    tprs_lower    = np.maximum(mean_tpr - np.std(interp_tprs, axis=0), 0)

    # --- ROC figure ---
    _, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        mean_fpr, mean_tpr,
        label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
        lw=1, alpha=0.8,
    )
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=0.2, label="± 1 std. dev.")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.grid(axis="both")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=(
            f"Time-agnostic RF ROC ({tissue})\n"
            f"Features: {label_tag}\n"
            f"AUC = {mean_auc:.3f} ± {std_auc:.3f}  |  Acc = {mean_acc:.3f} ± {std_acc:.3f}"
        ),
    )
    ax.legend(loc="lower right")

    fig_dir = f"results/figures/AUCROC_for_tissue_filtering/{label_tag}/RF"
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"roc_RF_{tissue}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print_timestamp(f"[{tissue}] ROC figure saved to {fig_path}")

    # --- Save all-data model ---
    all_clf = clone(classifier)
    all_clf.fit(X, y)

    all_dir  = f"results/models/time_agnostic_with_filtering/{label_tag}"
    os.makedirs(all_dir, exist_ok=True)
    all_path = os.path.join(all_dir, f"RF_{tissue}.pkl")
    pickle.dump(all_clf, open(all_path, "wb"))

    print_timestamp(
        f"[{tissue}] Done — mean AUC={mean_auc:.4f} ± {std_auc:.4f}, "
        f"mean Acc={mean_acc:.4f} ± {std_acc:.4f}"
    )

    return {
        "tissue":    tissue,
        "mean_auc":  mean_auc,
        "std_auc":   std_auc,
        "mean_acc":  mean_acc,
        "std_acc":   std_acc,
        "mean_fpr":  mean_fpr,
        "mean_tpr":  mean_tpr,
        "tprs_upper": tprs_upper,
        "tprs_lower": tprs_lower,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train time-agnostic Random Forest on motif enrichment features."
    )
    parser.add_argument(
        "--filter_labels",
        type=lambda x: x.lower() == "true",
        required=True,
        help="Use neural-label-filtered enrichment (True) or unfiltered (False)",
    )
    args = parser.parse_args()

    label_tag = "neural_labels" if args.filter_labels else "unfiltered"
    feature_dir_template = f"results/training_data/{label_tag}/hrs{{window}}"

    print_timestamp(f"=== Training time-agnostic RF | features: {label_tag} ===")
    print_timestamp(f"Feature directory template: {feature_dir_template}")

    # Process tissues in parallel
    results = {}
    with ProcessPoolExecutor(max_workers=len(TISSUES)) as executor:
        futures = {
            executor.submit(
                train_tissue, t, feature_dir_template, label_tag, N_SPLITS
            ): t
            for t in TISSUES
        }
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                results[t] = fut.result()
            except Exception as e:
                print_timestamp(f"[{t}] FAILED: {e}")
                raise

    # --- Summary AUCROC table ---
    rows = []
    for t in TISSUES:
        if t not in results:
            continue
        r = results[t]
        rows.append({
            "tissue":         t,
            "label_tag":      label_tag,
            "mean_auc":       round(r["mean_auc"], 6),
            "std_auc":        round(r["std_auc"],  6),
            "mean_acc":       round(r["mean_acc"], 6),
            "std_acc":        round(r["std_acc"],  6),
        })

    summary_df = pd.DataFrame(rows)
    summary_dir = f"results/time_agnostic_with_filtering/{label_tag}"
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "cv_aucroc_summary_RF.csv")
    summary_df.to_csv(summary_path, index=False)
    print_timestamp(f"Summary table saved to {summary_path}")

    print("\n" + summary_df.to_string(index=False))
    print_timestamp("=== All done ===")


if __name__ == "__main__":
    main()