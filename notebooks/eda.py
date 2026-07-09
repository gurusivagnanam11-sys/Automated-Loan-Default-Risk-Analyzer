"""
eda.py  ←→  eda.ipynb
---------------------
Exploratory Data Analysis for the Loan Default dataset.
Run as a script, or convert to Jupyter notebook with:
    pip install jupytext
    jupytext --to notebook eda.py
"""

# %% [markdown]
# # 📊 Loan Default — Exploratory Data Analysis
# This notebook walks through the key EDA steps:
# 1. Dataset overview
# 2. Missing value analysis
# 3. Target class distribution
# 4. Numeric feature distributions
# 5. Correlation heatmap
# 6. Categorical breakdowns

# %%
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 130, "axes.titlesize": 13})

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("../data/loan_data.csv")
print(f"Shape: {df.shape}")
print(df.head())

# %% [markdown]
# ## 1. Basic Overview

# %%
print(df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget distribution:\n", df["loan_status"].value_counts(normalize=True).round(3))

# %% [markdown]
# ## 2. Class Imbalance

# %%
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["loan_status"].value_counts()
ax.bar(["Fully Paid (0)", "Default (1)"],
       counts.values,
       color=["#4CAF50", "#F44336"], width=0.4)
ax.set_title("Target Class Distribution")
ax.set_ylabel("Count")
for i, v in enumerate(counts.values):
    ax.text(i, v + 20, f"{v:,}\n({v/len(df):.1%})", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("../models/eda_class_dist.png"); plt.close()
print("Saved: eda_class_dist.png")

# %% [markdown]
# ## 3. Numeric Feature Distributions

# %%
num_cols = ["loan_amnt", "int_rate", "annual_inc", "dti",
            "fico_score", "revol_util", "installment"]

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    axes[i].hist(df[col].dropna(), bins=40, color="#2196F3", edgecolor="white", alpha=0.85)
    axes[i].set_title(col)
    axes[i].set_xlabel("")

axes[-1].axis("off")
plt.suptitle("Numeric Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("../models/eda_distributions.png"); plt.close()
print("Saved: eda_distributions.png")

# %% [markdown]
# ## 4. Correlation Heatmap

# %%
corr_cols = ["loan_amnt", "int_rate", "installment", "annual_inc",
             "dti", "fico_score", "revol_bal", "revol_util",
             "total_acc", "loan_status"]

corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, linewidths=0.4,
    vmin=-1, vmax=1, ax=ax,
)
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../models/eda_correlation.png"); plt.close()
print("Saved: eda_correlation.png")

# %% [markdown]
# ## 5. Default Rate by Categorical Features

# %%
cat_cols = ["home_ownership", "purpose", "term"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col in zip(axes, cat_cols):
    rates = df.groupby(col)["loan_status"].mean().sort_values(ascending=False)
    rates.plot(kind="bar", ax=ax, color="#E91E63", edgecolor="white", alpha=0.85)
    ax.set_title(f"Default Rate by {col}")
    ax.set_ylabel("Default Rate")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)

plt.suptitle("Default Rate by Category", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("../models/eda_categorical.png"); plt.close()
print("Saved: eda_categorical.png")

# %% [markdown]
# ## 6. FICO Score vs Default

# %%
fig, ax = plt.subplots(figsize=(7, 4))
df.boxplot(column="fico_score", by="loan_status", ax=ax,
           boxprops=dict(color="#2196F3"),
           medianprops=dict(color="#F44336", linewidth=2))
ax.set_title("FICO Score by Loan Status")
ax.set_xlabel("Loan Status (0=Paid, 1=Default)")
ax.set_ylabel("FICO Score")
plt.suptitle("")
plt.tight_layout()
plt.savefig("../models/eda_fico_vs_status.png"); plt.close()
print("Saved: eda_fico_vs_status.png")

print("\n✅  EDA complete — all charts saved to models/")
