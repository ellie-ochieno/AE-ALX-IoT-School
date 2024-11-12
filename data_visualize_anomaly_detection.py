'''
* DATA MONITORING & VISUALIZATION------------------
*  # Devices:
*     -> DHT22 sensor
*     -> ESP32
*  # Technologies:
*     -> Protocol: HTTP
*     -> Platform: ThingSpeak + Google Colab
*  # Activity:
*    -> Data transmission, monitoring and visualization
*       - Anomalies detection in temperature&humidity data trends
*
*   ANGAZA ELIMU&ALX - IOT SCHOOL: Cohort 1, 2024
* --------------------------------------------------
'''

'''
 Support libraries -------------------------------------------------------------
'''
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt

# ---- ThingSpeak API details
CHANNEL_ID = "2739538"
API_KEY = "HIWSB0KUZ7UM29KU"  # Read API Key
THINGSPEAK_URL = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={API_KEY}&results=100"

# ---- Fetch data from ThingSpeak
response = requests.get(THINGSPEAK_URL)
data = response.json()

# ---- Extract data
timestamps = [entry['created_at'] for entry in data['feeds']]
temperature = [float(entry['field1']) for entry in data['feeds']]
humidity = [float(entry['field2']) for entry in data['feeds']]

# ---- Dataframe setup
df = pd.DataFrame({'Timestamp': timestamps, 'Temperature': temperature, 'Humidity': humidity})
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# ---- Anomaly thresholds definition
# compute the average mean and std deviation
temp_mean, temp_std = np.mean(df['Temperature']), np.std(df['Temperature'])
hum_mean, hum_std = np.mean(df['Humidity']), np.std(df['Humidity'])

# define lower and upper limit thresholds
temp_threshold_upper = temp_mean + 2 * temp_std
temp_threshold_lower = temp_mean - 2 * temp_std
hum_threshold_upper = hum_mean + 2 * hum_std
hum_threshold_lower = hum_mean - 2 * hum_std

# ---- Flag anomalies
df['Temp_Anomaly'] = (df['Temperature'] > temp_threshold_upper) | (df['Temperature'] < temp_threshold_lower)
df['Hum_Anomaly'] = (df['Humidity'] > hum_threshold_upper) | (df['Humidity'] < hum_threshold_lower)

# ---- Plot data and anomalies
plt.figure(figsize=(12, 8))

# ---- Plot temperature data with mean and threshold lines
plt.subplot(2, 1, 1)
plt.plot(df['Timestamp'], df['Temperature'], label='Temperature', color='blue')
plt.axhline(temp_mean, color='green', linestyle='--', label='Temperature Mean')
plt.axhline(temp_threshold_upper, color='red', linestyle='--', label='Temperature Upper Threshold')
plt.axhline(temp_threshold_lower, color='orange', linestyle='--', label='Temperature Lower Threshold')
plt.scatter(df[df['Temp_Anomaly']]['Timestamp'], df[df['Temp_Anomaly']]['Temperature'], color='red', label='Temperature Anomaly', s=50, zorder=5)
plt.fill_between(df['Timestamp'], temp_threshold_lower, temp_threshold_upper, color='lightblue', alpha=0.1, label='Normal Temperature Range')
plt.xlabel('Timestamp')
plt.ylabel('Temperature (°C)')
plt.legend(loc='upper left')
plt.title('Anomaly Detection in Temperature')

# ---- Plot humidity data with mean and threshold lines
plt.subplot(2, 1, 2)
plt.plot(df['Timestamp'], df['Humidity'], label='Humidity', color='purple')
plt.axhline(hum_mean, color='green', linestyle='--', label='Humidity Mean')
plt.axhline(hum_threshold_upper, color='red', linestyle='--', label='Humidity Upper Threshold')
plt.axhline(hum_threshold_lower, color='orange', linestyle='--', label='Humidity Lower Threshold')
plt.scatter(df[df['Hum_Anomaly']]['Timestamp'], df[df['Hum_Anomaly']]['Humidity'], color='purple', label='Humidity Anomaly', s=50, zorder=5)
plt.fill_between(df['Timestamp'], hum_threshold_lower, hum_threshold_upper, color='lightgreen', alpha=0.1, label='Normal Humidity Range')
plt.xlabel('Timestamp')
plt.ylabel('Humidity (%)')
plt.legend(loc='upper left')
plt.title('Anomaly Detection in Humidity')

plt.tight_layout()
plt.show()
