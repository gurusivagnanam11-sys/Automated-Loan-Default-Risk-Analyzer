"""
app.py
------
Flask web application for the Loan Default Risk Analyzer.
Run with:  python app.py
"""

import os
import sys

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, render_template, request, jsonify
from predict import predict_single

app = Flask(__name__)


# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Render the main prediction form."""
    return render_template("index.html")


# ─────────────────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept form data, run prediction, return result.
    Supports both HTML form posts and JSON API calls.
    """
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        # Cast to correct types
        borrower = {
            "loan_amnt":      float(data.get("loan_amnt",   15000)),
            "term":           str(data.get("term",          " 36 months")),
            "int_rate":       float(data.get("int_rate",    12.5)),
            "installment":    float(data.get("installment", 450)),
            "annual_inc":     float(data.get("annual_inc",  60000)),
            "dti":            float(data.get("dti",         18)),
            "fico_score":     int(float(data.get("fico_score", 700))),
            "revol_bal":      float(data.get("revol_bal",   8000)),
            "revol_util":     float(data.get("revol_util",  45)),
            "total_acc":      int(float(data.get("total_acc", 12))),
            "home_ownership": str(data.get("home_ownership", "RENT")).upper(),
            "purpose":        str(data.get("purpose",       "debt_consolidation")),
            "emp_length":     str(data.get("emp_length",    "5 years")),
        }

        result = predict_single(borrower)

        if request.is_json:
            return jsonify(result)

        return render_template(
            "result.html",
            label=result["label"],
            probability=result["probability"],
            risk_level=result["risk_level"],
            borrower=borrower,
        )

    except Exception as e:
        error_msg = str(e)
        if request.is_json:
            return jsonify({"error": error_msg}), 500
        return render_template("index.html", error=error_msg)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  🏦  Loan Default Risk Analyzer")
    print("  ─────────────────────────────────")
    print("  Open  →  http://127.0.0.1:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
