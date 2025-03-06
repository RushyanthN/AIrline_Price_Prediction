

# 🛫 Airline Price Prediction

## 🎯 Project Overview

This project aims to predict airline ticket prices using machine learning techniques. It leverages historical flight fare data and applies advanced modeling techniques such as XGBoost with cross-validation and hyperparameter tuning to achieve accurate price predictions.

## 📌 Key Features

* **Data Preprocessing and Feature Engineering:** Handling missing values, encoding categorical features, and scaling numerical variables.
* **Hyperparameter Tuning and Cross-Validation:** Optimizing the model for better performance.
* **Model Evaluation with Diagnostic Plots:** Assessing model accuracy and behavior.
* **Web-Based Interface:** Using Streamlit for easy interaction and prediction.

## 🏗️ Model Training & Evaluation

* The dataset is preprocessed to prepare it for modeling.
* XGBoost is used as the primary predictive model.
* Hyperparameter tuning is performed to optimize the XGBoost model.
* Cross-validation is implemented to ensure robust model performance.
* Model performance is evaluated using metrics such as RMSE, R², and MAE.
* Visual diagnostics are employed to understand feature importance and model behavior.

## 📊 Visualizations

* Feature correlation heatmaps to understand relationships between variables.
* Model prediction vs. actual price plots to visualize model accuracy.
* Residual analysis to identify potential model biases.

## 🛠️ Technologies Used

* **Python:**
    * Pandas (for data manipulation)
    * NumPy (for numerical operations)
    * Scikit-Learn (for model training, evaluation, and preprocessing)
    * XGBoost (for the predictive model)
* **Streamlit:** (for creating the interactive web application)
* **Matplotlib & Seaborn:** (for data visualization)

## 🎯 Future Improvements

* Implementing deep learning models (e.g., LSTMs for sequential data) to capture temporal patterns.
* Adding real-time API-based airline fare data to provide up-to-date predictions.
* Enhancing UI/UX of the Streamlit web app for a better user experience.
