import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model and scaler
# -----------------------------
MODEL_PATH  = "knn_model.pkl"
SCALER_PATH = "scaler.pkl"

loaded_model  = joblib.load(MODEL_PATH)
loaded_scaler = joblib.load(SCALER_PATH)

# -----------------------------
# Feature ranges
# -----------------------------
FEATURE_RANGES = {
    "fixed acidity":       (4.6,   15.9),
    "volatile acidity":    (0.12,   1.58),
    "citric acid":         (0.00,   1.00),
    "residual sugar":      (0.9,   15.5),
    "chlorides":           (0.012,  0.611),
    "free sulfur dioxide": (1.0,   72.0),
    "total sulfur dioxide":(6.0,  289.0),
    "density":             (0.99007, 1.00369),
    "pH":                  (2.74,   4.01),
    "sulphates":           (0.33,   2.00),
    "alcohol":             (8.4,   14.9)
}

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🍷 Red Wine Quality Predictor")
st.markdown("Enter the wine's **chemical properties** below. "
            "Inputs are shown two per row to minimize scrolling.")

input_data = {}
feature_list = list(FEATURE_RANGES.items())

# Iterate two features at a time
for i in range(0, len(feature_list), 2):
    cols = st.columns(2)
    for j in range(2):
        if i + j < len(feature_list):
            feature, (min_val, max_val) = feature_list[i + j]
            default_val = round((min_val + max_val) / 2, 2)
            with cols[j]:
                input_data[feature] = st.number_input(
                    label=feature.title(),
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=default_val,
                    step=0.01,
                    format="%.3f"
                )

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict Quality"):
    new_sample = pd.DataFrame([input_data])
    scaled_sample = loaded_scaler.transform(new_sample)
    prediction = loaded_model.predict(scaled_sample)[0]

    if prediction == 1:
        st.success("✅ **Good Red Wine** predicted!")
    else:
        st.error("❌ **Not a Good Red Wine** predicted.")
