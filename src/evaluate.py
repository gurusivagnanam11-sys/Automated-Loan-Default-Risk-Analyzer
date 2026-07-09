"""
evaluate.py
-----------
Comprehensive evaluation of a trained model:
  - Classification report (precision, recall, F1)
  - ROC-AUC score
  - Confusion matrix heatmap
  - ROC curve plot
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    ConfusionMatrixDisplay,
)

MODELS_DIR = "models"


# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, X_test: np.ndarray, y_test: np.ndarray,
             save_dir: str = MODELS_DIR) -> dict:
    """
    Compute all standard classification metrics and produce plots.

    Returns
    -------
    metrics : dict with keys accuracy, precision, recall, f1, roc_auc
    """
    os.makedirs(save_dir, exist_ok=True)

    y_pred      = model.predict(X_test)
    y_prob      = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )

    # ── Scalar metrics ────────────────────────────────────────────────────────
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred),           4),
        "precision": round(precision_score(y_test, y_pred,  zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred,     zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred,         zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob),            4),
    }

    # ── Console report ────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  EVALUATION REPORT")
    print("="*55)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print("="*55)
    print("\nDetailed Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Fully Paid (0)", "Default (1)"],
        zero_division=0,
    ))

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    _plot_confusion_matrix(y_test, y_pred, save_dir)

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    _plot_roc_curve(y_test, y_prob, metrics["roc_auc"], save_dir)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
def _plot_confusion_matrix(y_true, y_pred, save_dir: str):
    """Save a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fully Paid", "Default"],
        yticklabels=["Fully Paid", "Default"],
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot]  Confusion matrix saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
def _plot_roc_curve(y_true, y_prob, auc_score: float, save_dir: str):
    """Save ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#2196F3")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("ROC Curve — Loan Default Prediction", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()

    path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot]  ROC curve saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import joblib
    from preprocess import preprocess

    _, X_te, _, y_te, _ = preprocess("data/loan_data.csv")
    model = joblib.load("models/loan_default_model.pkl")
    evaluate(model, X_te, y_te)
