"""
demo_app.py
-----------
Interactive Streamlit demo for the breast cancer diagnosis model.
Runs entirely locally - no API key, no paid service, no internet required.

Run with:
    streamlit run demo_app.py
"""

import streamlit as st

from src.predict import DiagnosisModel

st.set_page_config(page_title="Breast Cancer Diagnosis Demo", page_icon="🩺")

st.title("🩺 Breast Cancer Diagnosis Demo")
st.write(
    "Adjust the cell-nuclei measurements below (or keep the defaults, which "
    "are pre-filled with realistic average values) and click **Predict** to "
    "see the model's diagnosis. Model: RandomForest, trained on the "
    "Breast Cancer Wisconsin (Diagnostic) dataset."
)


@st.cache_resource
def load_model() -> DiagnosisModel:
    return DiagnosisModel()


try:
    model = load_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

# Reasonable default values (approx. dataset means) so the form is usable
# without the user having to know real measurements.
DEFAULTS = {
    "mean radius": 14.1, "mean texture": 19.3, "mean perimeter": 92.0,
    "mean area": 655.0, "mean smoothness": 0.096, "mean compactness": 0.104,
    "mean concavity": 0.089, "mean concave points": 0.049,
    "mean symmetry": 0.181, "mean fractal dimension": 0.063,
    "radius error": 0.405, "texture error": 1.217, "perimeter error": 2.866,
    "area error": 40.3, "smoothness error": 0.007, "compactness error": 0.025,
    "concavity error": 0.032, "concave points error": 0.012,
    "symmetry error": 0.021, "fractal dimension error": 0.0038,
    "worst radius": 16.3, "worst texture": 25.7, "worst perimeter": 107.3,
    "worst area": 880.6, "worst smoothness": 0.132, "worst compactness": 0.254,
    "worst concavity": 0.272, "worst concave points": 0.115,
    "worst symmetry": 0.290, "worst fractal dimension": 0.0839,
}

with st.form("input_form"):
    st.subheader("Cell-nuclei measurements")
    cols = st.columns(3)
    values = {}
    for i, feature in enumerate(model.feature_names):
        default = DEFAULTS.get(feature, 0.0)
        with cols[i % 3]:
            values[feature] = st.number_input(
                feature, value=float(default), format="%.4f"
            )
    submitted = st.form_submit_button("Predict")

if submitted:
    result = model.predict(values)
    label = result["prediction"]
    probs = result["probabilities"]

    if label == "malignant":
        st.error(f"Prediction: **{label.upper()}**")
    else:
        st.success(f"Prediction: **{label.upper()}**")

    st.write("Class probabilities:")
    st.bar_chart(probs)

st.caption(
    "This demo is for educational/portfolio purposes only and is not a "
    "medical diagnostic tool."
)
