import streamlit as st
import pandas as pd
import joblib

MODEL_PATH  = "knn_model.pkl"
SCALER_PATH = "scaler.pkl"

loaded_model  = joblib.load(MODEL_PATH)
loaded_scaler = joblib.load(SCALER_PATH)

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

st.title("🍷 Red Wine Quality Predictor")

st.markdown(
    "Enter the wine's **chemical properties** below. "
    "Values are restricted to the dataset's observed min/max."
)

# Collect inputs
input_data = {}
for feature, (min_val, max_val) in FEATURE_RANGES.items():
    default_val = round((min_val + max_val) / 2, 2)
    input_data[feature] = st.number_input(
        label=f"{feature.title()}",
        min_value=float(min_val),
        max_value=float(max_val),
        value=default_val,
        step=0.01,
        format="%.3f"
    )

if st.button("Predict Quality"):
    # Convert to DataFrame
    new_sample = pd.DataFrame([input_data])

    # Scale features
    scaled_sample = loaded_scaler.transform(new_sample)

    # Predict
    prediction = loaded_model.predict(scaled_sample)[0]

    # Display result
    if prediction == 1:
        st.success("✅ **Good Red Wine** predicted!")
    else:
        st.error("❌ **Not a Good Red Wine** predicted.")
