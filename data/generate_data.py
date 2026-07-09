"""
generate_data.py
----------------
Generates a realistic synthetic loan dataset for the Loan Default Risk Analyzer.
Run this once to produce data/loan_data.csv before training.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000  # number of samples

# ── Helper arrays ────────────────────────────────────────────────────────────
home_ownership_opts = ["RENT", "OWN", "MORTGAGE", "OTHER"]
purpose_opts        = [
    "debt_consolidation", "credit_card", "home_improvement",
    "other", "major_purchase", "small_business", "medical",
    "car", "vacation", "moving",
]
emp_length_opts = [
    "< 1 year", "1 year", "2 years", "3 years", "4 years",
    "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years",
]

# ── Raw feature draws ────────────────────────────────────────────────────────
loan_amnt   = np.random.randint(1_000, 40_001, N)
term        = np.random.choice([" 36 months", " 60 months"], N, p=[0.70, 0.30])
int_rate    = np.round(np.random.uniform(5.5, 28.0, N), 2)
annual_inc  = np.random.lognormal(mean=10.8, sigma=0.6, size=N).clip(15_000, 300_000)
dti         = np.round(np.random.uniform(0, 40, N), 2)
fico_score  = np.random.randint(580, 851, N)
revol_bal   = np.random.randint(0, 50_001, N)
revol_util  = np.round(np.random.uniform(0, 100, N), 1)
total_acc   = np.random.randint(2, 40, N)
home_ownership = np.random.choice(home_ownership_opts, N, p=[0.40, 0.15, 0.42, 0.03])
purpose        = np.random.choice(purpose_opts, N)
emp_length     = np.random.choice(emp_length_opts, N)

# Installment derived from loan amount + rate + term
months      = np.where(term == " 36 months", 36, 60)
r           = int_rate / 100 / 12
installment = np.where(
    r == 0,
    loan_amnt / months,
    np.round(loan_amnt * (r * (1 + r) ** months) / ((1 + r) ** months - 1), 2),
)

# ── Realistic default probability ───────────────────────────────────────────
# Higher int_rate, higher dti, lower fico → higher default probability
log_odds = (
    -4.0
    + 0.06  * int_rate
    + 0.04  * dti
    - 0.008 * (fico_score - 650)
    + 0.000_004 * revol_bal
    + 0.005 * revol_util
    - 0.000_003 * annual_inc
    + 0.3   * (term == " 60 months").astype(int)
)
prob_default = 1 / (1 + np.exp(-log_odds))
loan_status  = (np.random.rand(N) < prob_default).astype(int)

# ── Assemble DataFrame ───────────────────────────────────────────────────────
df = pd.DataFrame({
    "loan_amnt":      loan_amnt,
    "term":           term,
    "int_rate":       int_rate,
    "installment":    installment,
    "annual_inc":     annual_inc.round(2),
    "dti":            dti,
    "fico_score":     fico_score,
    "revol_bal":      revol_bal,
    "revol_util":     revol_util,
    "total_acc":      total_acc,
    "home_ownership": home_ownership,
    "purpose":        purpose,
    "emp_length":     emp_length,
    "loan_status":    loan_status,
})

# Inject ~3 % random NaN values into numeric columns for realism
for col in ["annual_inc", "dti", "revol_util", "emp_length"]:
    mask = np.random.rand(N) < 0.03
    df.loc[mask, col] = np.nan

df.to_csv("loan_data.csv", index=False)
print(f"✅  loan_data.csv saved  ({N} rows, default rate = {loan_status.mean():.2%})")
