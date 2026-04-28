import pandas as pd
import numpy as np

# Keywords used to auto-detect sensitive columns
SENSITIVE_KEYWORDS = ["gender", "sex", "race", "ethnicity", "age", "religion", "nationality"]

def detect_sensitive_columns(df: pd.DataFrame) -> list:
    """
    Automatically detects sensitive/protected attribute columns
    by matching column names against known keywords.
    """
    detected = []
    for col in df.columns:
        for keyword in SENSITIVE_KEYWORDS:
            if keyword in col.lower():
                detected.append(col)
                break
    return detected


def compute_demographic_parity(sensitive_col: pd.Series, outcome: pd.Series) -> dict:
    """
    Demographic Parity: For each group, what fraction received a positive outcome?
    
    Formula: P(Y=1 | group=g) for each group g
    Returns: dict of {group_name: positive_rate}
    """
    df_temp = pd.DataFrame({"group": sensitive_col, "outcome": outcome}).dropna()
    parity = {}
    for group, grp_df in df_temp.groupby("group"):
        rate = round(grp_df["outcome"].mean(), 4)
        parity[str(group)] = rate
    return parity


def compute_disparate_impact(sensitive_col: pd.Series, outcome: pd.Series) -> dict:
    """
    Disparate Impact: Ratio of positive outcome rate for each group vs. the BEST group.
    
    Formula: P(Y=1 | group=g) / P(Y=1 | group=best)
    A value < 0.8 means the group is treated unfairly (EEOC 80% Rule).
    Returns: dict of {group_name: disparate_impact_ratio}
    """
    dp = compute_demographic_parity(sensitive_col, outcome)
    if not dp:
        return {}

    max_rate = max(dp.values())
    if max_rate == 0:
        return {g: 1.0 for g in dp}

    di = {}
    for group, rate in dp.items():
        di[group] = round(rate / max_rate, 4)
    return di