import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('california_housing_model.pkl')
        feature_engineer = joblib.load('feature_engineer.pkl')
        return model, feature_engineer
    except:
        st.error("Model files not found.")
        return None, None

# Preprocess input data
def preprocess_input_data(input_data, feature_engineer):
    df = pd.DataFrame([input_data])
    df_engineered = feature_engineer.transform(df)
    return df_engineered

# Main app
def main():
    st.title("California Housing Price Predictor")
    model, feature_engineer = load_models()
    
    if model is None:
        st.stop()
    
    with st.form("property_details"):
        longitude = st.slider("Longitude", -124.35, -114.31, -118.24, step=0.01)
        latitude = st.slider("Latitude", 32.54, 41.95, 34.05, step=0.01)
        housing_median_age = st.slider("Median Housing Age (years)", 1, 52, 25)
        total_rooms = st.number_input("Total Rooms", min_value=1, value=1000, step=10)
        total_bedrooms = st.number_input("Total Bedrooms", min_value=1, value=200, step=10)
        population = st.number_input("Population", min_value=1, value=500, step=10)
        households = st.number_input("Households", min_value=1, value=300, step=10)
        median_income = st.slider("Median Income (tens of thousands)", 0.5, 15.0, 3.0, step=0.1)
        
        submitted = st.form_submit_button("Predict House Price")
        
        if submitted:
            input_data = {
                'longitude': longitude,
                'latitude': latitude,
                'housing_median_age': housing_median_age,
                'total_rooms': total_rooms,
                'total_bedrooms': total_bedrooms,
                'population': population,
                'households': households,
                'median_income': median_income
            }
            
            processed_data = preprocess_input_data(input_data, feature_engineer)
            prediction = model.predict(processed_data)[0]
            
            st.metric("Predicted House Value", f"${prediction:,.0f}")

if __name__ == "__main__":
    main()
