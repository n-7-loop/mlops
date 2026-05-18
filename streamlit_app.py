import joblib
import numpy as np
import streamlit as st
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.config import MODELS_DIR
from pipeline.preprocessing import CATEGORICAL_COLS, NUMERIC_COLS

MODEL_PATH = MODELS_DIR / "RandomForest.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODER_PATH = MODELS_DIR / "feature_encoders.pkl"


@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        enc = joblib.load(ENCODER_PATH)
        return model, scaler, enc, None
    except Exception as e:
        return None, None, None, str(e)


model, scaler, encoders, load_err = load_artifacts()

st.set_page_config(page_title="Sale total estimator", layout="centered")
st.title("Predict checkout total — Developed by Ayma")

if load_err:
    st.error(f"Error loading model artifacts: {load_err}")
    st.stop()

with st.form("prediction_form"):
    cols = st.columns(2)
    inputs = {}
    # Populate selects from encoders
    for i, col in enumerate(CATEGORICAL_COLS):
        with cols[i % 2]:
            vals = sorted(getattr(encoders[col], "classes_", []).tolist(), key=str.lower)
            inputs[col] = st.selectbox(col.replace("_", " ").title(), options=[""] + vals, index=0)

    with cols[0]:
        inputs["unit_price"] = st.number_input("Unit price (USD)", min_value=0.0, format="%.2f")
    with cols[1]:
        inputs["quantity"] = st.number_input("Quantity", min_value=1, step=1)

    submitted = st.form_submit_button("Estimate total")

if submitted:
    try:
        # Validate selections
        for col in CATEGORICAL_COLS:
            if not inputs.get(col):
                st.warning(f"Please select a value for {col}.")
                raise ValueError("Missing categorical input")

        row = []
        for col in CATEGORICAL_COLS:
            raw = str(inputs[col]).strip()
            le = encoders[col]
            if raw not in le.classes_:
                st.error(f"Unknown {col}: {raw!r}. Pick a value seen in training.")
                raise ValueError("Unknown categorical value")
            row.append(float(le.transform([raw])[0]))

        row.append(float(inputs["unit_price"]))
        row.append(float(inputs["quantity"]))

        features_scaled = scaler.transform(np.array([row]))
        predicted = float(model.predict(features_scaled)[0])
        st.success(f"Estimated total: ${predicted:,.2f}")
    except Exception as e:
        if not isinstance(e, ValueError):
            st.error(f"Prediction failed: {e}")