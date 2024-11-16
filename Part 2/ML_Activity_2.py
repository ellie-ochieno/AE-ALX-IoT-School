'''
* ML CLASSIFICATION OF DATA TRENDS ------------------
*  # Devices:
*     -> DHT22 sensor
*     -> ESP32
*  # Technologies:
*     -> Protocol: HTTP
*     -> Platform: ThingSpeak + Google Colab
*     -> ML Tools: Data trends classification - RandomForestClassifier
*  # Activity:
*    -> Building a classification model to classify temperature trends (e.g., "Rising", "Falling", or "Stable")
*       based on the dataset collected from a DHT22 sensor.
*
*   ANGAZA ELIMU&ALX - IOT SCHOOL: Cohort 1, 2024
* --------------------------------------------------
'''

# Import necessary libraries
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ThingSpeak Configuration
CHANNEL_ID = ''
READ_API_KEY = ''

# Section 1: Data Collection from ThingSpeak
def fetch_data():
    """
    Fetches temperature and humidity data from a ThingSpeak channel.
    Returns a DataFrame with timestamp, temperature, and humidity columns.
    """
    url = f'https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=1000'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['feeds'])
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['temperature'] = pd.to_numeric(df['field1'], errors='coerce')
    df['humidity'] = pd.to_numeric(df['field2'], errors='coerce')
    print("Data fetched successfully.")
    return df[['created_at', 'temperature', 'humidity']]

# Section 2: Preprocess and label data for trends
def label_temperature_trends(df):
    """
    Label temperature trends:
    - Rising: if temperature at t > temperature at t-1
    - Falling: if temperature at t < temperature at t-1
    - Stable: if temperature at t == temperature at t-1
    """
    df['temperature_diff'] = df['temperature'].diff()
    df['trend'] = df['temperature_diff'].apply(
        lambda x: 'Rising' if x > 0 else ('Falling' if x < 0 else 'Stable')
    )
    df = df.dropna()  # Drop the first row since it will have NaN for diff
    print("Temperature trends labeled successfully.")
    return df

# Section 3: Split data into features and labels
def prepare_data(df):
    # Features: Use time-based index and previous temperature values
    df=df.dropna() #drops null and non-applicable datasets in api payload
    df['time_index'] = (df['created_at'] - df['created_at'].min()).dt.total_seconds()
    X = df[['time_index', 'temperature']]
    y = df['trend']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Section 4: Train a classification model
def train_classification_model(X_train, y_train):
    clf = RandomForestClassifier(random_state=42, n_estimators=100)
    clf.fit(X_train, y_train)
    print("Classification model trained successfully.")
    return clf

# Section 5: Test and evaluate the model
def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return y_pred

# Section 6: Visualize classification results
def visualize_results(df, y_pred, X_test):
    df_test = pd.DataFrame(X_test)
    df_test['predicted_trend'] = y_pred
    plt.figure(figsize=(10, 6))
    for trend in df_test['predicted_trend'].unique():
        subset = df_test[df_test['predicted_trend'] == trend]
        plt.scatter(subset['time_index'], subset['temperature'], label=trend)
    plt.xlabel("Time Index")
    plt.ylabel("Temperature")
    plt.title("Temperature Trend Classification")
    plt.legend()
    plt.show()

# Section 7: Predict and classify trends on new data
def classify_new_data(clf):
    new_data = fetch_data()
    new_data['time_index'] = (new_data['created_at'] - new_data['created_at'].min()).dt.total_seconds()
    X_new = new_data[['time_index', 'temperature']]
    new_data['predicted_trend'] = clf.predict(X_new)
    print("Predicted trends on new data:")
    print(new_data[['created_at', 'temperature', 'predicted_trend']].tail())
    return new_data

# Main execution flow
if __name__ == "__main__":
    # Step 1: Fetch and preprocess the data
    df = fetch_data()
    df = label_temperature_trends(df)

    # Step 2: Split the data
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Step 3: Train the classification model
    clf = train_classification_model(X_train, y_train)

    # Step 4: Evaluate the model
    y_pred = evaluate_model(clf, X_test, y_test)

    # Step 5: Visualize classification results
    visualize_results(df, y_pred, X_test)

    # Step 6: Predict on new data
    classify_new_data(clf)
