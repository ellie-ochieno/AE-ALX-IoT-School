'''
* ML PREDICTION & ANOMALIES DETECTION ------------------
*  # Devices:
*     -> DHT22 sensor
*     -> ESP32
*  # Technologies:
*     -> Protocol: HTTP
*     -> Platform: ThingSpeak + Google Colab
*     -> ML Tools: Prediction & Regression - LinearRegression
*  # Activity:
*    -> Using ML for data processing, prediction and anomalies detection
*
*   ANGAZA ELIMU&ALX - IOT SCHOOL: Cohort 1, 2024
* --------------------------------------------------
'''
#DoHardThings01
# Import necessary libraries
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ThingSpeak Configuration
CHANNEL_ID = ''
READ_API_KEY = ''

# Section 1: Data Collection from ThingSpeak
def fetch_data():
    url = f'https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=1000'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['feeds'])
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['temperature'] = pd.to_numeric(df['field1'], errors='coerce')
    df['humidity'] = pd.to_numeric(df['field2'], errors='coerce')
    print("Data fetched successfully.")
    return df[['created_at', 'temperature', 'humidity']]

# Section 2: Data Preparation
def prepare_data(df):
    df=df.dropna() #drops null and non-applicable datasets in api payload
    df['time_index'] = (df['created_at'] - df['created_at'].min()).dt.total_seconds()
    X = df[['time_index']]
    y_temp = df['temperature']
    y_hum = df['humidity']
    print("Data prepared successfully.")
    return X, y_temp, y_hum

# Section 3: Train Temperature and Humidity Prediction Models
def train_regression_models(X, y_temp, y_hum):
    # Split data for training and testing
    X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
    _, _, y_hum_train, y_hum_test = train_test_split(X, y_hum, test_size=0.2, random_state=42)

    # Train temperature prediction model
    temp_model = LinearRegression()
    temp_model.fit(X_train, y_temp_train)
    temp_predictions = temp_model.predict(X_test)
    temp_mse = mean_squared_error(y_temp_test, temp_predictions)

    # Train humidity prediction model
    hum_model = LinearRegression()
    hum_model.fit(X_train, y_hum_train)
    hum_predictions = hum_model.predict(X_test)
    hum_mse = mean_squared_error(y_hum_test, hum_predictions)

    print(f"Temperature Prediction MSE: {temp_mse}")
    print(f"Humidity Prediction MSE: {hum_mse}")
    return temp_model, hum_model

# Section 4.1: Anomaly Detection for Humidity
def humid_anomaly_detection(df):
    hum_mean, hum_std = df['humidity'].mean(), df['humidity'].std()
    threshold_upper = hum_mean + 2 * hum_std
    threshold_lower = hum_mean - 2 * hum_std
    df['humidity_anomaly'] = df['humidity'].apply(lambda x: 'Anomaly' if x > threshold_upper or x < threshold_lower else 'Normal')
    print("Anomaly detection completed.")
    return threshold_upper, threshold_lower

# Section 4.2: Anomaly Detection for Temperature
def temp_anomaly_detection(df):
    temp_mean, temp_std = df['temperature'].mean(), df['temperature'].std()
    threshold_upper = temp_mean + 2 * temp_std
    threshold_lower = temp_mean - 2 * temp_std
    df['temperature_anomaly'] = df['temperature'].apply(lambda x: 'Anomaly' if x > threshold_upper or x < threshold_lower else 'Normal')
    print("Anomaly detection completed.")
    return threshold_upper, threshold_lower

# Section 5: Predict and Test on New Data
def test_new_data(temp_model, hum_model, temp_thresholds, hum_thresholds):
    # Fetch new data
    new_data = fetch_data()
    new_data['time_index'] = (new_data['created_at'] - new_data['created_at'].min()).dt.total_seconds()
    X_new = new_data[['time_index']]

    # Make predictions
    new_data['predicted_temperature'] = temp_model.predict(X_new)
    new_data['predicted_humidity'] = hum_model.predict(X_new)

    # Apply anomaly detection for humidity
    hum_upper, hum_lower = hum_thresholds
    new_data['humidity_anomaly'] = new_data['humidity'].apply(lambda x: 'Anomaly' if x > hum_upper or x < hum_lower else 'Normal')

    # Apply anomaly detection for temperature
    temp_upper, temp_lower = temp_thresholds
    new_data['temperature_anomaly'] = new_data['temperature'].apply(lambda x: 'Anomaly' if x > temp_upper or x < temp_lower else 'Normal')

    print("Predictions and anomaly detection on new data:")
    print(new_data[['created_at', 'temperature', 'predicted_temperature', 'temperature_anomaly',
                    'humidity', 'predicted_humidity', 'humidity_anomaly']].tail())

# Main Execution Flow
if __name__ == "__main__":
    # Step 1: Fetch and prepare the data
    df = fetch_data()
    X, y_temp, y_hum = prepare_data(df)

    # Step 2: Train the ML models for temperature and humidity
    temp_model, hum_model = train_regression_models(X, y_temp, y_hum)

    # Step 3: Set up and apply anomaly detection for humidity and temperature
    temp_thresholds = temp_anomaly_detection(df)
    hum_thresholds = humid_anomaly_detection(df)

    # Step 4: Test and use the models on new data
    test_new_data(temp_model, hum_model, temp_thresholds, hum_thresholds)
