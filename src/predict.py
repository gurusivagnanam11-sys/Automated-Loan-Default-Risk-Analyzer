"""
predict.py
----------
Command-line and programmatic prediction interface.
Load a saved model + preprocessor and classify new loan applications.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Paths
MODEL_PATH       = os.path.join("models", "loan_default_model.pkl")
PREPROCESSOR_PATH = os.path.join("models", "preprocessor.pkl")

# ── Feature engineering (must mirror preprocess.py) ──────────────────────────
def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)
    df["payment_ratio"]  = df["installment"] / (df["annual_inc"] / 12 + 1)
    df["fico_band"]      = pd.cut(
        df["fico_score"],
        bins=[0, 620, 680, 740, 800, 900],
        labels=["Very Poor", "Fair", "Good", "Very Good", "Excellent"],
    ).astype(str)
    return df


# ─────────────────────────────────────────────────────────────────────────────
def load_artifacts():
    """Load model and preprocessor from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}.")

    model        = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


# ─────────────────────────────────────────────────────────────────────────────
def predict_single(borrower_data: dict) -> dict:
    """
    Predict default risk for a single borrower.

    Parameters
    ----------
    borrower_data : dict with keys matching the loan feature columns

    Returns
    -------
    dict : {label, probability, risk_level}
    """
    model, preprocessor = load_artifacts()

    df = pd.DataFrame([borrower_data])
    df = _add_engineered_features(df)

    X      = preprocessor.transform(df)
    label  = int(model.predict(X)[0])
    prob   = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else None

    risk_level = "🔴 HIGH RISK — Likely to Default" if label == 1 else "🟢 LOW RISK — Likely to Fully Pay"

    return {
        "label":       label,
        "probability": round(prob, 4) if prob is not None else "N/A",
        "risk_level":  risk_level,
    }


# ─────────────────────────────────────────────────────────────────────────────
def interactive_cli():
    """Walk the user through entering borrower details in the terminal."""
    print("\n" + "═"*55)
    print("  🏦  Loan Default Risk Analyzer — CLI Mode")
    print("═"*55 + "\n")

    def _get(prompt, cast=float, default=None):
        raw = input(f"  {prompt}: ").strip()
        if not raw and default is not None:
            return default
        try:
            return cast(raw)
        except ValueError:
            print(f"  ⚠  Invalid input, using default: {default}")
            return default

    borrower = {
        "loan_amnt":      _get("Loan Amount (e.g. 15000)",            float, 15000),
        "term":           input("  Term [ 36 months / 60 months ]: ").strip() or " 36 months",
        "int_rate":       _get("Interest Rate % (e.g. 12.5)",         float, 12.5),
        "installment":    _get("Monthly Installment (e.g. 450)",      float, 450),
        "annual_inc":     _get("Annual Income (e.g. 60000)",          float, 60000),
        "dti":            _get("Debt-to-Income Ratio (e.g. 18)",      float, 18),
        "fico_score":     _get("FICO Credit Score (580–850, e.g. 700)", int, 700),
        "revol_bal":      _get("Revolving Balance (e.g. 8000)",       float, 8000),
        "revol_util":     _get("Revolving Utilization % (e.g. 45)",   float, 45),
        "total_acc":      _get("Total Accounts (e.g. 12)",            int,   12),
        "home_ownership": input("  Home Ownership [RENT/OWN/MORTGAGE/OTHER]: ").strip().upper() or "RENT",
        "purpose":        input("  Loan Purpose (e.g. debt_consolidation): ").strip() or "debt_consolidation",
        "emp_length":     input("  Employment Length (e.g. 5 years): ").strip() or "5 years",
    }

    result = predict_single(borrower)

    print("\n" + "─"*55)
    print(f"  PREDICTION : {result['risk_level']}")
    print(f"  Default Probability : {result['probability']:.2%}" if isinstance(result["probability"], float) else "")
    print("─"*55 + "\n")
    return result


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # If a JSON arg is passed, use it; otherwise go interactive
    if len(sys.argv) > 1:
        data = json.loads(sys.argv[1])
        res  = predict_single(data)
        print(json.dumps(res, indent=2))
    else:
        interactive_cli()
