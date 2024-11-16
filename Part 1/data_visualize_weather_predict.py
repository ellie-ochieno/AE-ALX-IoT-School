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
*       - Time series basic weather forecasting and prediction
*
*   ANGAZA ELIMU&ALX - IOT SCHOOL: Cohort 1, 2024
* --------------------------------------------------
'''

'''
 Support libraries -------------------------------------------------------------
'''
import requests
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# ---- ThingSpeak API details
CHANNEL_ID = "2739538"
API_KEY = "HIWSB0KUZ7UM29KU" # Read API Key
THINGSPEAK_URL = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={API_KEY}&results=100"

# ---- Fetch data from ThingSpeak
response = requests.get(THINGSPEAK_URL)
data = response.json()

# ---- Parse data
timestamps = [entry['created_at'] for entry in data['feeds']]
temperature = [float(entry['field1']) for entry in data['feeds']]
humidity = [float(entry['field2']) for entry in data['feeds']]

# ---- Create DataFrame
df = pd.DataFrame({'Timestamp': timestamps, 'Temperature': temperature, 'Humidity': humidity})
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# ---- Prepare data for Prophet - the future forecasting handler lib
temperature_df = df[['Timestamp', 'Temperature']].rename(columns={'Timestamp': 'ds', 'Temperature': 'y'})
humidity_df = df[['Timestamp', 'Humidity']].rename(columns={'Timestamp': 'ds', 'Humidity': 'y'})

# ---- Remove timezone from 'ds' column - timezone not required for the prophet future forecasting lib handler lib
temperature_df['ds'] = temperature_df['ds'].dt.tz_localize(None)
humidity_df['ds'] = humidity_df['ds'].dt.tz_localize(None)

# ---- Fit Prophet model for Temperature Forecasting
temp_model = Prophet(interval_width=0.95)  # Set prediction interval to 95%
temp_model.fit(temperature_df)

# ---- Forecasting 24 hours into the future
future_temp = temp_model.make_future_dataframe(periods=24, freq='h')
temp_forecast = temp_model.predict(future_temp)

# ---- Plot Temperature Forecast
plt.figure(figsize=(10, 5))
temp_model.plot(temp_forecast, xlabel='Time', ylabel='Temperature (°C)')
plt.title('24-Hour Temperature Forecast with Historical Trends')
plt.grid(True)

# ---- Highlight observed data period
plt.fill_between(temp_forecast['ds'], temp_forecast['yhat_lower'], temp_forecast['yhat_upper'], color='skyblue', alpha=0.2)
plt.annotate('Future Predicted Range', xy=(temp_forecast['ds'].iloc[-5], temp_forecast['yhat'].iloc[-5]), color='blue')

# ---- Fit Prophet model for Humidity Forecasting
humidity_model = Prophet(interval_width=0.95)
humidity_model.fit(humidity_df)

# ---- Forecasting 24 hours into the future
future_humidity = humidity_model.make_future_dataframe(periods=24, freq='h')
humidity_forecast = humidity_model.predict(future_humidity)

# ---- Plot Humidity Forecast
plt.figure(figsize=(10, 5))
humidity_model.plot(humidity_forecast, xlabel='Time', ylabel='Humidity (%)')
plt.title('24-Hour Humidity Forecast with Historical Trends')
plt.grid(True)

# ---- Highlight observed data period
plt.fill_between(humidity_forecast['ds'], humidity_forecast['yhat_lower'], humidity_forecast['yhat_upper'], color='lightgreen', alpha=0.2)
plt.annotate('Future Predicted Range', xy=(humidity_forecast['ds'].iloc[-5], humidity_forecast['yhat'].iloc[-5]), color='green')

# ---- Plot the forecast
plt.show()
