"""
preprocess.py
-------------
Handles all data cleaning, feature engineering, encoding, and splitting.
Returns train/test splits ready for model training.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_COL   = "loan_status"
RANDOM_STATE = 42
TEST_SIZE    = 0.20

# Feature groups (adjust if columns differ)
CAT_FEATURES = ["home_ownership", "purpose", "emp_length", "term"]
NUM_FEATURES = [
    "loan_amnt", "int_rate", "installment", "annual_inc",
    "dti", "fico_score", "revol_bal", "revol_util", "total_acc",
]


# ─────────────────────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV file and perform sanity checks."""
    df = pd.read_csv(filepath)
    print(f"[load]  {df.shape[0]} rows × {df.shape[1]} columns loaded.")
    assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' not found!"
    return df


# ─────────────────────────────────────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
      - Numeric  → median (robust to outliers)
      - Categoric → mode  (most frequent category)
    """
    df = df.copy()

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=["object", "str"]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    print(f"[missing]  After imputation → nulls = {df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features that may improve model performance.
      - loan_to_income  : how large the loan is relative to annual income
      - fico_band       : bucketed credit score (coarse categorical)
      - payment_ratio   : monthly installment / monthly income
    """
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
def build_preprocessor() -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer:
      - Numeric columns  → StandardScaler
      - Categoric columns → OneHotEncoder (unknown categories → ignore)
    Additional engineered features are included.
    """
    extended_num = NUM_FEATURES + ["loan_to_income", "payment_ratio"]
    extended_cat = CAT_FEATURES + ["fico_band"]

    numeric_transformer  = StandardScaler()
    categoric_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,  extended_num),
            ("cat", categoric_transformer, extended_cat),
        ],
        remainder="drop",
    )
    return preprocessor


# ─────────────────────────────────────────────────────────────────────────────
def preprocess(
    filepath: str,
    apply_smote: bool = True,
    save_preprocessor: bool = True,
    models_dir: str = "models",
):
    """
    Full preprocessing pipeline:
      1. Load data
      2. Handle missing values
      3. Engineer features
      4. Split into X / y, then train / test
      5. Fit ColumnTransformer on train set, transform both
      6. Optionally apply SMOTE to balance the training set
      7. Save fitted preprocessor for later inference

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    preprocessor                      : fitted ColumnTransformer
    """
    df = load_data(filepath)
    df = handle_missing(df)
    df = engineer_features(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    # ── Train / test split (stratified to preserve class ratio) ──────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"[split]  Train={len(X_train)}  Test={len(X_test)}")
    print(f"[split]  Class balance (train) → 0: {(y_train==0).sum()}  1: {(y_train==1).sum()}")

    # ── Fit & transform ───────────────────────────────────────────────────────
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)
    print(f"[preprocess]  Feature matrix shape after encoding: {X_train_proc.shape}")

    # ── SMOTE oversampling on training data ───────────────────────────────────
    if apply_smote:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_proc, y_train = smote.fit_resample(X_train_proc, y_train)
        print(f"[SMOTE]  After resampling → shape: {X_train_proc.shape}  "
              f"0: {(y_train==0).sum()}  1: {(y_train==1).sum()}")

    # ── Persist preprocessor for inference use ────────────────────────────────
    if save_preprocessor:
        os.makedirs(models_dir, exist_ok=True)
        path = os.path.join(models_dir, "preprocessor.pkl")
        joblib.dump(preprocessor, path)
        print(f"[save]  Preprocessor saved → {path}")

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, prep = preprocess("data/loan_data.csv")
    print("Preprocessing complete.")
