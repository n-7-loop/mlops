import sys
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.config import MODELS_DIR
from pipeline.preprocessing import CATEGORICAL_COLS, NUMERIC_COLS

app = Flask(__name__)

MODEL_PATH = MODELS_DIR / "RandomForest.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODER_PATH = MODELS_DIR / "feature_encoders.pkl"

EXPECTED_FEATURE_COUNT = len(CATEGORICAL_COLS) + len(NUMERIC_COLS)


def _sales_artifacts_ok(encoders, scaler) -> bool:
    if not isinstance(encoders, dict):
        return False
    if not all(col in encoders for col in CATEGORICAL_COLS):
        return False
    n_in = getattr(scaler, "n_features_in_", None)
    if n_in is not None and int(n_in) != EXPECTED_FEATURE_COUNT:
        return False
    return True


model = scaler = feature_encoders = None
_load_error = None
try:
    _model = joblib.load(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)
    _enc = joblib.load(ENCODER_PATH)
    if not _sales_artifacts_ok(_enc, _scaler):
        _load_error = (
            "Saved models/scaler/encoders do not match the sales pipeline "
            f"(expected encoder keys {list(CATEGORICAL_COLS)} and {EXPECTED_FEATURE_COUNT} features). "
            "Re-run: python scripts/run_pipeline.py"
        )
        print(_load_error)
    else:
        model, scaler, feature_encoders = _model, _scaler, _enc
        print("Models loaded successfully.")
except Exception as e:
    _load_error = str(e)
    print(f"Error loading models: {e}")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/form-options")
def form_options():
    if feature_encoders is None:
        msg = _load_error or "Encoders not loaded."
        # 200 so the client can read JSON; terminal is not flooded with 500 for a known state.
        return jsonify({"success": False, "error": msg})
    try:
        out = {}
        for col in CATEGORICAL_COLS:
            le = feature_encoders[col]
            out[col] = sorted(le.classes_.tolist(), key=str.lower)
        return jsonify({"success": True, "options": out})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None or scaler is None or feature_encoders is None:
            err = _load_error or "Model artifacts not loaded. Run scripts/run_pipeline.py first."
            return jsonify({"success": False, "error": err})

        data = request.json
        row = []
        for col in CATEGORICAL_COLS:
            raw = str(data[col]).strip()
            enc = feature_encoders[col]
            if raw not in enc.classes_:
                return jsonify(
                    {
                        "success": False,
                        "error": f"Unknown {col}: {raw!r}. Pick a value seen in training.",
                    }
                )
            row.append(float(enc.transform([raw])[0]))

        row.append(float(data["unit_price"]))
        row.append(float(data["quantity"]))

        features_scaled = scaler.transform(np.array([row]))
        predicted = float(model.predict(features_scaled)[0])

        return jsonify({"success": True, "prediction": round(predicted, 2)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
