"""
train_model.py
--------------
Trains four classifiers (Logistic Regression, Decision Tree,
Random Forest, XGBoost), runs GridSearchCV for hyperparameter
tuning, and saves the best model.
"""

import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from xgboost                 import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics         import roc_auc_score

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR = "models"
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
def get_models() -> dict:
    """Return dict of model_name → (estimator, param_grid)."""
    models = {
        "Logistic Regression": (
            LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            {
                "C":       [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l2"],
                "solver":  ["lbfgs"],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "max_depth":        [4, 6, 8, None],
                "min_samples_leaf": [5, 10, 20],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": [100, 200],
                "max_depth":    [6, 10, None],
                "max_features": ["sqrt", "log2"],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
                n_jobs=-1,
            ),
            {
                "n_estimators":  [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth":     [4, 6],
                "subsample":     [0.8, 1.0],
            },
        ),
    }
    return models


# ─────────────────────────────────────────────────────────────────────────────
def train_and_select(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
) -> tuple:
    """
    Train every model with GridSearchCV, compare ROC-AUC, return best model.

    Parameters
    ----------
    X_train   : preprocessed training features
    y_train   : training labels
    cv_folds  : StratifiedKFold splits
    scoring   : metric to maximise during search

    Returns
    -------
    best_name     : str
    best_model    : fitted estimator
    results_table : dict  {model_name: {best_params, cv_score}}
    """
    cv       = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    models   = get_models()
    results  = {}
    best_score = -1
    best_name  = None
    best_model = None

    print(f"\n{'='*60}")
    print(f" Model Training  (CV={cv_folds}, scoring={scoring})")
    print(f"{'='*60}\n")

    for name, (estimator, param_grid) in models.items():
        print(f"▶  Training: {name} ...")
        gs = GridSearchCV(
            estimator,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        gs.fit(X_train, y_train)

        best_cv_score = gs.best_score_
        results[name] = {
            "best_params": gs.best_params_,
            "cv_roc_auc":  round(float(best_cv_score), 4),
        }
        print(f"   Best params : {gs.best_params_}")
        print(f"   CV ROC-AUC  : {best_cv_score:.4f}\n")

        if best_cv_score > best_score:
            best_score = best_cv_score
            best_name  = name
            best_model = gs.best_estimator_

    print(f"🏆  Best model: {best_name}  (CV ROC-AUC = {best_score:.4f})\n")
    return best_name, best_model, results


# ─────────────────────────────────────────────────────────────────────────────
def plot_model_comparison(results: dict, save_dir: str = MODELS_DIR):
    """Bar chart comparing CV ROC-AUC scores of all models."""
    names  = list(results.keys())
    scores = [results[n]["cv_roc_auc"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, scores, color=["#2196F3", "#4CAF50", "#FF9800", "#E91E63"])
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("CV ROC-AUC Score", fontsize=12)
    ax.set_title("Model Comparison — Cross-Validated ROC-AUC", fontsize=13, fontweight="bold")

    # Annotate bars
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}", va="center", fontsize=10,
        )

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot]  Model comparison chart saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list, save_dir: str = MODELS_DIR):
    """Plot top-20 feature importances (tree-based models only)."""
    if not hasattr(model, "feature_importances_"):
        print("[info]  Feature importance not available for this model type.")
        return

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:20]  # top 20

    top_names  = [feature_names[i] for i in indices]
    top_values = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_names[::-1], top_values[::-1], color="#2196F3")
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title("Top-20 Feature Importances", fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(save_dir, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot]  Feature importance chart saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
def save_model(model, name: str, results: dict, save_dir: str = MODELS_DIR):
    """Persist model + metadata to disk."""
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "loan_default_model.pkl")
    meta_path  = os.path.join(save_dir, "training_results.json")

    joblib.dump(model, model_path)
    with open(meta_path, "w") as f:
        json.dump({"best_model": name, "results": results}, f, indent=2)

    print(f"[save]  Model saved    → {model_path}")
    print(f"[save]  Metadata saved → {meta_path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from preprocess import preprocess

    X_tr, X_te, y_tr, y_te, prep = preprocess("data/loan_data.csv")

    best_name, best_model, results = train_and_select(X_tr, y_tr)

    plot_model_comparison(results)

    # Attempt to reconstruct feature names for importance plot
    try:
        num_names = prep.transformers_[0][2]
        cat_enc   = prep.transformers_[1][1]
        cat_names = list(cat_enc.get_feature_names_out(prep.transformers_[1][2]))
        all_names = num_names + cat_names
        plot_feature_importance(best_model, all_names)
    except Exception as e:
        print(f"[warn]  Could not plot feature importance: {e}")

    save_model(best_model, best_name, results)
    print("\n✅  Training complete.")
