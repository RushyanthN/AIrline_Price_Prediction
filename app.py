import streamlit as st
import pandas as pd
import numpy as np
from model import load_and_preprocess_data, train_xgboost_model, predict_price

# Load and preprocess data
df_economy, df_business, onehot, df_model = load_and_preprocess_data()

# Train models
model_economy = train_xgboost_model(df_economy)
model_business = train_xgboost_model(df_business)

# Streamlit app
st.title('Flight Price Prediction')

# User inputs
flight_class = st.selectbox('Class', ['Economy', 'Business'])
day_of_week = st.selectbox('Travel Day', ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
arr_time_window = st.selectbox('Arrival Time Window', ['00:00-04:00', '04:00-08:00', '08:00-12:00', '12:00-16:00', '16:00-20:00', '20:00-00:00'])
stops = st.selectbox('Number of Stops', [0, 1, 2])
airline = st.selectbox('Airline', ['Vistara', 'Air India'])
duration_in_hours = st.number_input('Duration in Hours', value=5.0, step=0.5)

# Convert user inputs to model format
if flight_class == 'Economy':
    input_data = {col: 0 for col in df_economy.columns if col != 'log_price' and col != 'class_Business'}
else:
    input_data = {col: 0 for col in df_business.columns if col != 'log_price' and col != 'class_Economy'}

input_data['Duration_in_hours'] = duration_in_hours
input_data['stop'] = stops
input_data['arr_time__window'] = int(arr_time_window.split(':')[0]) // 4
input_data[f'day_of_week_{day_of_week}'] = 1
input_data[f'airline_{airline}'] = 1

input_df = pd.DataFrame([input_data])

predicted_price = predict_price(model_economy if flight_class == 'Economy' else model_business, input_df)
st.write(f"Predicted Flight Price: ₹{predicted_price}")