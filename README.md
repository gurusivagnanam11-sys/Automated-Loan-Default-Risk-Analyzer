# 🏦 Automated Loan Default Risk Analyzer

An end-to-end machine learning project that predicts whether a borrower is likely to **default on a loan** using financial and demographic features. Includes a full Flask web app with an interactive risk dashboard.

---

## 📁 Project Structure

```
loan_default_project/
│
├── data/
│   ├── loan_data.csv          ← Generated synthetic dataset (5 000 rows)
│   └── generate_data.py       ← Script to regenerate dataset
│
├── notebooks/
│   └── eda.py                 ← EDA script (converts to .ipynb via jupytext)
│
├── src/
│   ├── preprocess.py          ← Data cleaning, encoding, SMOTE, pipeline
│   ├── train_model.py         ← Model training + GridSearchCV comparison
│   ├── evaluate.py            ← Metrics, confusion matrix, ROC curve
│   └── predict.py             ← Single-sample prediction (CLI + programmatic)
│
├── models/
│   ├── loan_default_model.pkl  ← Best trained model (auto-saved)
│   ├── preprocessor.pkl        ← Fitted ColumnTransformer (auto-saved)
│   ├── model_comparison.png    ← CV ROC-AUC bar chart
│   ├── feature_importance.png  ← Top-20 feature importance plot
│   ├── confusion_matrix.png    ← Test-set confusion matrix
│   ├── roc_curve.png           ← ROC curve with AUC
│   └── training_results.json   ← Best params per model
│
├── templates/
│   ├── index.html              ← Input form (dark finance aesthetic)
│   └── result.html             ← Risk verdict with animated probability
│
├── app.py                      ← Flask web application
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone <repo-url>
cd loan_default_project
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
cd data
python generate_data.py
cd ..
```

### 3. Preprocess + Train

```bash
cd src
python preprocess.py      # Cleans data, applies SMOTE, saves preprocessor
python train_model.py     # Trains all 4 models, saves best model + plots
python evaluate.py        # Prints metrics, saves confusion matrix + ROC curve
cd ..
```

### 4. Run the Web App

```bash
python app.py
# Open → http://127.0.0.1:5000
```

### 5. CLI Prediction (optional)

```bash
cd src
python predict.py        # Interactive terminal input
```

---

## 🧠 Models Trained

| Model               | CV ROC-AUC (typical) |
|---------------------|----------------------|
| Logistic Regression | ~0.82                |
| Decision Tree       | ~0.78                |
| Random Forest       | ~0.87                |
| **XGBoost**         | **~0.89**            |

Best model is selected automatically by 5-fold cross-validated ROC-AUC.

---

## 📊 Features Used

| Feature         | Type        | Description                          |
|-----------------|-------------|--------------------------------------|
| `loan_amnt`     | Numeric     | Requested loan amount (USD)          |
| `term`          | Categorical | 36 or 60 months                      |
| `int_rate`      | Numeric     | Annual interest rate (%)             |
| `installment`   | Numeric     | Monthly payment amount               |
| `annual_inc`    | Numeric     | Borrower annual income               |
| `dti`           | Numeric     | Debt-to-income ratio                 |
| `fico_score`    | Numeric     | FICO credit score (580–850)          |
| `revol_bal`     | Numeric     | Total revolving balance              |
| `revol_util`    | Numeric     | Revolving credit utilization (%)     |
| `total_acc`     | Numeric     | Total credit accounts                |
| `home_ownership`| Categorical | RENT / OWN / MORTGAGE / OTHER        |
| `purpose`       | Categorical | Loan purpose category                |
| `emp_length`    | Categorical | Years of employment                  |

**Engineered features** (added automatically):
- `loan_to_income` — loan amount / annual income
- `payment_ratio`  — installment / monthly income
- `fico_band`      — bucketed credit score category

---

## 📈 Evaluation Metrics (sample output)

```
======================================================
  EVALUATION REPORT
======================================================
  Accuracy  : 0.9020
  Precision : 0.7143
  Recall    : 0.6800
  F1-Score  : 0.6967
  ROC-AUC   : 0.8912
======================================================

Detailed Classification Report:
                  precision  recall  f1-score  support
Fully Paid (0)     0.94      0.95     0.94       896
Default (1)        0.71      0.68     0.70       104
```

---

## 🌐 API Usage (JSON)

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 15000, "term": " 36 months", "int_rate": 18.5,
    "installment": 540, "annual_inc": 45000, "dti": 25,
    "fico_score": 640, "revol_bal": 12000, "revol_util": 72,
    "total_acc": 8, "home_ownership": "RENT",
    "purpose": "debt_consolidation", "emp_length": "2 years"
  }'
```

Response:
```json
{
  "label": 1,
  "probability": 0.7832,
  "risk_level": "🔴 HIGH RISK — Likely to Default"
}
```

---

## 🔬 Technical Details

- **Imbalanced classes** handled via SMOTE (Synthetic Minority Oversampling)
- **Hyperparameter tuning** via `GridSearchCV` with 5-fold stratified CV
- **Pipeline architecture** via `sklearn.compose.ColumnTransformer`
- **Categorical encoding** via `OneHotEncoder` (unknown categories handled gracefully)
- **Feature scaling** via `StandardScaler` (numeric features only)
- **Model persistence** via `joblib` (both model and preprocessor saved)

---

## 📦 Dependencies

See `requirements.txt`. Core:
- `scikit-learn`, `xgboost`, `imbalanced-learn`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `flask`, `joblib`

---

*Built as an industry-level ML project demonstrating end-to-end modeling, evaluation, and deployment.*
