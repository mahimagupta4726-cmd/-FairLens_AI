import pandas as pd
import numpy as np
from utils import detect_sensitive_columns, compute_demographic_parity, compute_disparate_impact
def analyze_bias(df: pd.DataFrame) -> dict:
    """
    Main bias analysis function.
    Takes a pandas DataFrame and returns a full fairness report as a dict.
    """
    df.columns = [c.strip().lower() for c in df.columns]  # normalize column names

    sensitive_cols = detect_sensitive_columns(df)
    results = {
        "total_rows": len(df),
        "columns_detected": list(df.columns),
        "sensitive_columns": sensitive_cols,
        "bias_analysis": {},
        "fairness_score": 100,
        "alerts": [],
        "verdict": ""
    }

    if not sensitive_cols:
        results["alerts"].append("⚠️ No sensitive columns detected (gender, race, age).")
        results["verdict"] = "No bias analysis possible — no sensitive attributes found."
        return results

    # Find the outcome/decision column
    outcome_col = None
    for candidate in ["decision", "outcome", "label", "hired", "approved", "result"]:
        if candidate in df.columns:
            outcome_col = candidate
            break

    if outcome_col is None:
        # Try last column as fallback
        outcome_col = df.columns[-1]
        results["alerts"].append(f"ℹ️ No 'decision' column found. Using last column '{outcome_col}' as outcome.")

    # Binarize outcome column if needed
    outcome_series = df[outcome_col]
    unique_vals = outcome_series.dropna().unique()
    
    if outcome_series.dtype == object or set(str(v).lower() for v in unique_vals) & {"yes","no","true","false","y","n"}:
        mapping = {"yes":1,"y":1,"true":1,"1":1,"no":0,"n":0,"false":0,"0":0}
        outcome_series = outcome_series.astype(str).str.strip().str.lower().map(mapping).fillna(0).astype(int)
    else:
        # Use median as threshold
        median_val = outcome_series.median()
        outcome_series = (outcome_series >= median_val).astype(int)

    penalty = 0  # used to calculate overall fairness score

    for col in sensitive_cols:
        col_result = {}
        groups = df[col].dropna().unique()
        col_result["groups_found"] = [str(g) for g in groups]

        # Demographic Parity
        dp = compute_demographic_parity(df[col], outcome_series)
        col_result["demographic_parity"] = dp

        # Disparate Impact
        di = compute_disparate_impact(df[col], outcome_series)
        col_result["disparate_impact"] = di

        # Bias flag: Disparate Impact < 0.8 is the "80% rule" (EEOC standard)
        bias_detected = any(v < 0.8 for v in di.values() if isinstance(v, float))
        col_result["bias_detected"] = bias_detected

        if bias_detected:
            penalty += 25
            results["alerts"].append(
                f"🚨 Bias detected in '{col}': Disparate Impact below 0.8 threshold (EEOC 80% Rule violated)."
            )

        # Parity gap: max difference between group positive rates
        if dp:
            rates = list(dp.values())
            parity_gap = round(max(rates) - min(rates), 4)
            col_result["parity_gap"] = parity_gap
            if parity_gap > 0.1:
                penalty += 10
                results["alerts"].append(
                    f"⚠️ High parity gap in '{col}': {parity_gap:.2%} difference in positive decision rates."
                )

        results["bias_analysis"][col] = col_result

    # Final fairness score (0–100)
    fairness_score = max(0, 100 - penalty)
    results["fairness_score"] = fairness_score

    # Verdict
    if fairness_score >= 80:
        results["verdict"] = "✅ FAIR — Model shows acceptable fairness levels."
    elif fairness_score >= 50:
        results["verdict"] = "⚠️ MODERATE BIAS — Some fairness issues detected. Review sensitive attributes."
    else:
        results["verdict"] = "🚨 HIGH BIAS — Significant discrimination patterns found. Immediate review required."

    return results