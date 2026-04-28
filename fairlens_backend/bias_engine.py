"""
FairLens AI — Real-Time Bias Detection Engine
=============================================
Hackathon-ready: works with CSV/JSON uploads, live streaming rows,
and a Flask API that the index.html frontend talks to.

Usage:
    pip install flask flask-cors pandas numpy
    python bias_engine.py
    → API running at http://localhost:5000
"""

import time
import threading
import json
from collections import deque

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow the HTML frontend on any port to call this API

# ──────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

SENSITIVE_KEYWORDS = [
    "gender", "sex", "race", "ethnicity", "age",
    "religion", "nationality", "zip", "zipcode",
]

OUTCOME_KEYWORDS = [
    "decision", "outcome", "label", "approved", "hired",
    "accepted", "result", "target", "y",
]

# 80 % rule threshold (EEOC / OFCCP)
DI_THRESHOLD = 0.80

# ──────────────────────────────────────────────────────────────────────────────
#  COLUMN DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def detect_sensitive_columns(df: pd.DataFrame) -> list[str]:
    """
    Match column names against SENSITIVE_KEYWORDS (case-insensitive substring).
    Returns list of detected column names.
    """
    detected = []
    for col in df.columns:
        col_lower = col.lower()
        for kw in SENSITIVE_KEYWORDS:
            if kw in col_lower:
                detected.append(col)
                break
    return detected


def detect_outcome_column(df: pd.DataFrame) -> str | None:
    """
    Find the most likely binary outcome column.
    Prefers columns whose names match OUTCOME_KEYWORDS,
    then falls back to any binary 0/1 column.
    """
    # 1. keyword match
    for col in df.columns:
        col_lower = col.lower()
        for kw in OUTCOME_KEYWORDS:
            if kw in col_lower:
                return col

    # 2. fallback: first binary-ish column
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if set(unique_vals).issubset({0, 1, "0", "1", True, False,
                                       "yes", "no", "Yes", "No",
                                       "approved", "rejected",
                                       "Approved", "Rejected"}):
            return col

    return None


def binarize_outcome(series: pd.Series) -> pd.Series:
    """
    Convert string outcomes (Approved/Rejected, yes/no …) to 0/1.
    """
    pos = {"1", "yes", "true", "approved", "hired", "accepted", "high risk", 1, True}
    def _convert(v):
        if pd.isna(v):
            return np.nan
        return 1 if str(v).strip().lower() in {str(p).lower() for p in pos} else 0
    return series.map(_convert).astype(float)


# ──────────────────────────────────────────────────────────────────────────────
#  BIAS METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_demographic_parity(
    sensitive_col: pd.Series, outcome: pd.Series
) -> dict[str, float]:
    """
    P(Y=1 | group=g) for every group g.
    Returns {group_name: positive_rate}.
    """
    df_tmp = pd.DataFrame({"group": sensitive_col, "outcome": outcome}).dropna()
    result = {}
    for group, grp in df_tmp.groupby("group"):
        result[str(group)] = round(float(grp["outcome"].mean()), 4)
    return result


def compute_disparate_impact(
    sensitive_col: pd.Series, outcome: pd.Series
) -> dict[str, float]:
    """
    DI ratio: P(Y=1|group) / P(Y=1|best_group).
    Value < 0.8  →  legally actionable under EEOC 80 % rule.
    """
    dp = compute_demographic_parity(sensitive_col, outcome)
    if not dp:
        return {}

    max_rate = max(dp.values())
    if max_rate == 0:
        return {g: 1.0 for g in dp}

    return {g: round(v / max_rate, 4) for g, v in dp.items()}


def compute_equalized_odds(
    sensitive_col: pd.Series, outcome: pd.Series, predicted: pd.Series | None = None
) -> dict[str, float]:
    """
    True-positive rate per group (approximated when no predicted column available).
    """
    if predicted is None:
        predicted = outcome  # treat outcome itself as prediction for now

    df_tmp = pd.DataFrame(
        {"group": sensitive_col, "outcome": outcome, "pred": predicted}
    ).dropna()

    result = {}
    for group, grp in df_tmp.groupby("group"):
        positives = grp[grp["outcome"] == 1]
        if len(positives) == 0:
            result[str(group)] = 0.0
        else:
            result[str(group)] = round(float(positives["pred"].mean()), 4)
    return result


def compute_fairness_score(bias_analysis: dict) -> int:
    """
    Composite fairness score 0-100.
    Penalty: each column with DI < 0.8 reduces score.
    """
    if not bias_analysis:
        return 100

    penalties = []
    for col_data in bias_analysis.values():
        di_vals = list(col_data.get("disparate_impact", {}).values())
        if di_vals:
            min_di = min(di_vals)
            # DI=1.0 → no penalty; DI=0.0 → full penalty
            penalties.append(max(0.0, 1.0 - min_di))

    if not penalties:
        return 100

    avg_penalty = sum(penalties) / len(penalties)
    score = max(0, min(100, round(100 * (1 - avg_penalty))))
    return score


def generate_alerts(bias_analysis: dict, fairness_score: int) -> list[str]:
    """
    Build human-readable alert strings from computed bias metrics.
    """
    alerts = []
    for col, data in bias_analysis.items():
        di_vals = data.get("disparate_impact", {})
        dp_vals = data.get("demographic_parity", {})

        for group, di in di_vals.items():
            if di < DI_THRESHOLD:
                rate = dp_vals.get(group, "?")
                alerts.append(
                    f"🚨 {col.capitalize()} bias — group '{group}': "
                    f"DI={di:.2f} (below 0.80 threshold). "
                    f"Positive rate={rate}. EEOC 80% Rule violated."
                )

        if dp_vals:
            rates = list(dp_vals.values())
            spread = max(rates) - min(rates) if rates else 0
            if spread > 0.2:
                best = max(dp_vals, key=dp_vals.get)
                worst = min(dp_vals, key=dp_vals.get)
                alerts.append(
                    f"⚠️ {col.capitalize()} parity gap: "
                    f"'{best}'={dp_vals[best]:.2f} vs '{worst}'={dp_vals[worst]:.2f} "
                    f"({spread:.0%} spread)."
                )

    if fairness_score < 50:
        alerts.insert(0, "🚨 CRITICAL: Overall fairness score below 50 — immediate remediation needed.")
    elif fairness_score < 70:
        alerts.insert(0, "⚠️ WARNING: Fairness score below 70 — significant bias detected.")

    return alerts


# ──────────────────────────────────────────────────────────────────────────────
#  CORE ANALYSIS FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def analyze_dataframe(df: pd.DataFrame) -> dict:
    """
    Full bias analysis pipeline on a pandas DataFrame.
    Returns a dict ready to be serialised as JSON for the frontend.
    """
    # 1. Detect columns
    sensitive_cols = detect_sensitive_columns(df)
    outcome_col    = detect_outcome_column(df)

    if outcome_col is None:
        return {
            "error": "No outcome column detected. "
                     "Rename your target column to 'decision', 'outcome', or 'label'.",
            "columns": list(df.columns),
        }

    outcome = binarize_outcome(df[outcome_col])

    # 2. Compute per-column metrics
    bias_analysis: dict[str, dict] = {}
    for col in sensitive_cols:
        dp = compute_demographic_parity(df[col], outcome)
        di = compute_disparate_impact(df[col], outcome)
        eo = compute_equalized_odds(df[col], outcome)

        bias_analysis[col] = {
            "demographic_parity": dp,
            "disparate_impact":   di,
            "equalized_odds":     eo,
            "bias_detected":      any(v < DI_THRESHOLD for v in di.values()),
            "groups_count":       int(df[col].nunique()),
        }

    # 3. Overall score & alerts
    fairness_score = compute_fairness_score(bias_analysis)
    alerts         = generate_alerts(bias_analysis, fairness_score)

    return {
        "fairness_score":   fairness_score,
        "total_rows":       len(df),
        "outcome_column":   outcome_col,
        "sensitive_columns": sensitive_cols,
        "bias_analysis":    bias_analysis,
        "alerts":           alerts,
        "positive_rate_overall": round(float(outcome.mean()), 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  REAL-TIME STREAMING  (Server-Sent Events)
# ──────────────────────────────────────────────────────────────────────────────

# Global state for live streaming
_stream_lock   = threading.Lock()
_stream_buffer: deque[dict] = deque(maxlen=500)   # last 500 analysed rows
_stream_stats  = {
    "total": 0, "bias_count": 0,
    "fairness_history": deque(maxlen=60),          # last 60 ticks
}


def _analyse_row(row: pd.Series, sensitive_cols: list[str], outcome_col: str) -> dict:
    """Light-weight single-row bias check for the live stream."""
    outcome_val = binarize_outcome(pd.Series([row.get(outcome_col)]))[0]
    groups      = {col: str(row.get(col, "?")) for col in sensitive_cols}

    # Flag when a minority group gets a negative outcome
    # (simplified heuristic — real product would compare vs baseline rate)
    flagged = False
    if outcome_val == 0 and groups:
        flagged = True   # any negative outcome on a sensitive group is highlighted

    return {
        "ts":       time.strftime("%H:%M:%S"),
        "groups":   groups,
        "outcome":  int(outcome_val) if not np.isnan(outcome_val) else None,
        "flagged":  flagged,
    }


def stream_dataframe(df: pd.DataFrame, delay: float = 0.05):
    """
    Background thread: walk through a DataFrame row-by-row,
    push results into _stream_buffer and update _stream_stats.
    Simulates a real-time decision stream.
    """
    sensitive_cols = detect_sensitive_columns(df)
    outcome_col    = detect_outcome_column(df)
    if outcome_col is None:
        return

    for _, row in df.iterrows():
        result = _analyse_row(row, sensitive_cols, outcome_col)
        with _stream_lock:
            _stream_buffer.appendleft(result)
            _stream_stats["total"] += 1
            if result["flagged"]:
                _stream_stats["bias_count"] += 1

            total = _stream_stats["total"]
            bias  = _stream_stats["bias_count"]
            tick_fairness = round(100 * (1 - bias / total), 1) if total else 100
            _stream_stats["fairness_history"].appendleft(tick_fairness)

        time.sleep(delay)


# ──────────────────────────────────────────────────────────────────────────────
#  FLASK API ROUTES
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /analyze
    Accepts: multipart/form-data with field 'file' (CSV or JSON)
    Returns: JSON bias analysis report
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send field name 'file'."}), 400

    f = request.files["file"]
    filename = f.filename.lower()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(f)
        elif filename.endswith(".json"):
            df = pd.read_json(f)
        else:
            return jsonify({"error": "Only .csv and .json files are supported."}), 400
    except Exception as exc:
        return jsonify({"error": f"Could not parse file: {exc}"}), 400

    result = analyze_dataframe(df)

    # Kick off background stream (non-blocking)
    threading.Thread(
        target=stream_dataframe, args=(df,), kwargs={"delay": 0.08}, daemon=True
    ).start()

    return jsonify(result)


@app.route("/stream", methods=["GET"])
def stream():
    """
    GET /stream
    Server-Sent Events endpoint.
    The frontend connects here to get a live feed of decisions.

    JavaScript example:
        const src = new EventSource('http://localhost:5000/stream');
        src.onmessage = e => console.log(JSON.parse(e.data));
    """
    def event_generator():
        last_sent = 0
        while True:
            with _stream_lock:
                new_items = list(_stream_buffer)[:10]   # send up to 10 latest rows
                stats = {
                    "total":            _stream_stats["total"],
                    "bias_count":       _stream_stats["bias_count"],
                    "fairness_history": list(_stream_stats["fairness_history"])[:30],
                }

            payload = {"rows": new_items, "stats": stats}
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(1.0)   # push update every second

    return Response(
        event_generator(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",     # disable Nginx buffering
        },
    )


@app.route("/stream/reset", methods=["POST"])
def stream_reset():
    """Clear the live stream buffer — useful when switching datasets."""
    with _stream_lock:
        _stream_buffer.clear()
        _stream_stats["total"] = 0
        _stream_stats["bias_count"] = 0
        _stream_stats["fairness_history"].clear()
    return jsonify({"status": "reset"})


@app.route("/health", methods=["GET"])
def health():
    """Quick liveness check — call this on page load to test connectivity."""
    return jsonify({
        "status": "ok",
        "stream_rows": len(_stream_buffer),
        "total_analysed": _stream_stats["total"],
    })


@app.route("/demo", methods=["GET"])
def demo():
    """
    GET /demo
    Generates a synthetic biased hiring dataset and runs analysis.
    Useful for testing the frontend without uploading a real file.
    """
    np.random.seed(42)
    n = 400
    gender = np.random.choice(["Male", "Female"], n, p=[0.55, 0.45])
    race   = np.random.choice(["White", "Black", "Asian", "Latino"], n,
                               p=[0.50, 0.20, 0.18, 0.12])
    age    = np.random.randint(22, 60, n)
    score  = np.random.randint(50, 100, n)

    # Inject deliberate bias: females approved less, Black/Latino approved less
    prob = np.full(n, 0.70)
    prob[gender == "Female"] -= 0.22
    prob[race == "Black"]    -= 0.18
    prob[race == "Latino"]   -= 0.14
    prob = np.clip(prob, 0.05, 0.95)
    decision = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "gender": gender, "race": race, "age": age,
        "score": score, "decision": decision,
    })
    result = analyze_dataframe(df)

    # Start streaming the demo data
    threading.Thread(
        target=stream_dataframe, args=(df,), kwargs={"delay": 0.15}, daemon=True
    ).start()

    return jsonify(result)


# ──────────────────────────────────────────────────────────────────────────────
#  CLI HELPER  — run analysis directly from the terminal
# ──────────────────────────────────────────────────────────────────────────────

def cli_analyze(filepath: str):
    """Quick terminal analysis without starting the Flask server."""
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_json(filepath)

    result = analyze_dataframe(df)

    print("\n" + "═" * 60)
    print("  FAIRLENS AI  —  Bias Analysis Report")
    print("═" * 60)
    print(f"  Rows analysed : {result['total_rows']}")
    print(f"  Fairness Score: {result['fairness_score']} / 100")
    print(f"  Outcome column: {result['outcome_column']}")
    print(f"  Sensitive cols: {', '.join(result['sensitive_columns']) or 'none detected'}")
    print()

    for col, data in result["bias_analysis"].items():
        print(f"  ── {col.upper()} ──")
        for group, di in data["disparate_impact"].items():
            flag = "⚠ BIAS" if di < DI_THRESHOLD else "✓ OK"
            dp   = data["demographic_parity"].get(group, "?")
            print(f"     {group:15s}  DI={di:.3f}  DP={dp:.3f}  {flag}")
        print()

    if result["alerts"]:
        print("  ALERTS:")
        for a in result["alerts"]:
            print(f"    {a}")
    print("═" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        # python bias_engine.py data.csv
        cli_analyze(sys.argv[1])
    else:
        # python bias_engine.py  → start the Flask server
        print("╔══════════════════════════════════════╗")
        print("║   FairLens AI  — Bias Engine v1.0    ║")
        print("╠══════════════════════════════════════╣")
        print("║  POST /analyze   → upload CSV/JSON   ║")
        print("║  GET  /stream    → live SSE feed     ║")
        print("║  GET  /demo      → synthetic dataset ║")
        print("║  GET  /health    → liveness check    ║")
        print("╚══════════════════════════════════════╝")
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
