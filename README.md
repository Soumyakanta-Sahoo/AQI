# Air Quality Index Prediction and Forecasting using Machine Learning

This project presents a complete machine learning pipeline to analyze, predict, and forecast the **Air Quality Index (AQI)** across Indian cities, using historical data and time series models.

---


## 📁 Project Structure
```
AQI/ 
├── data/ # Raw dataset (e.g. city_day.csv) 
├── src/
│ ├── data_preprocessing.py 
│ ├── eda.py 
│ ├── models.py 
│ ├── forecast.py 
├── output/ # All model outputs and plots 
├── main.py # Main driver script 
├── report.md # Project write-up in markdown 
├── report.tex # LaTeX version of the report 
└── README.md # This file
```
---

## 📊 Features

- **Data Cleaning & Preprocessing**
  - Handles missing values
  - Feature engineering (year, month, day, weekday)
  - AQI category labeling

- **Exploratory Data Analysis**
  - AQI distribution
  - Category counts
  - Correlation heatmap
  - Time and city-wise trends

- **Machine Learning Models**
  - Regression: Linear, Tree-based, Gradient Boosting, XGBoost
  - Classification: Random Forest for AQI category prediction
  - Feature importance visualization

- **Time Series Forecasting**
  - [✔] Prophet-based AQI forecasting
  - [🛑] ARIMA (attempted but excluded due to instability)

- **Output Folder Structure**
  - Forecasts saved as CSV + PNG
  - Classification results exported
  - EDA plots stored for reporting

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
