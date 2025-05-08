import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime
import xgboost as xgb

# Page Configuration
st.set_page_config(
    page_title="🏠 Smart Price Forecaster",
    page_icon="📈",
    layout="wide"
)

# Load model and data
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("model.pkl")
        features = joblib.load("features.pkl")
        sample_data = pd.read_csv("house_price-dataset.csv")
        return model, features, sample_data
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.stop()

model, features, sample_data = load_artifacts()

# Sidebar
with st.sidebar:
    st.title("Settings")
    prediction_type = st.radio(
        "Prediction Mode",
        ["Single Property", "Batch Upload"]
    )
    st.markdown("---")
    st.markdown("### About")
    st.markdown(f"""
    This app predicts house prices using machine learning.
    - *Model*: XGBoost Regressor
    - *Last Updated*: {datetime.now().strftime("%B %d, %Y")}
    """)

# App Title
st.title("🏠 Smart Price Forecaster")
st.markdown("Predict accurate home values using our AI-powered valuation tool")

if prediction_type == "Single Property":
    st.header("Property Details")
    col1, col2 = st.columns(2)

    user_input = {}
    for i, feature in enumerate(features):
        with (col1 if i < len(features) / 2 else col2):
            user_input[feature] = st.number_input(f"{feature}", value=0.0)

    if st.button("Predict Price", type="primary"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)
        st.success(f"### Estimated Value: ${prediction[0]:,.2f}")
else:
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        try:
            batch_data = pd.read_csv(uploaded_file)
            missing = set(features) - set(batch_data.columns)
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                if st.button("Predict Batch", type="primary"):
                    preds = model.predict(batch_data[features])
                    batch_data["PredictedPrice"] = preds
                    st.dataframe(batch_data.head())
                    csv = batch_data.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Results", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Sample Data Format
with st.expander("💡 Sample Data Format"):
    st.write("Expected format for batch predictions:")
    st.dataframe(sample_data.head())
