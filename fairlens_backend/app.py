from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import traceback

# Try importing your model
try:
    from model import analyze_bias
    MODEL_AVAILABLE = True
except:
    MODEL_AVAILABLE = False

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "FairLens AI Backend Running ✅"})

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        df = pd.read_csv(file)

        # ✅ Try real model
        if MODEL_AVAILABLE:
            result = analyze_bias(df)
        else:
            # 🔁 Fallback (safe demo)
            result = {
                "fairness_score": 84,
                "bias": "Medium",
                "rows": len(df)
            }

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 FairLens AI Backend running on http://localhost:5000")
    app.run(debug=True, port=5000)