/*
* DATA MONITORING & VISUALIZATION------------------
*  # Devices:
*     -> DHT22 sensor
*     -> ESP32
*  # Technologies:
*     -> Protocol: HTTP
*     -> Platform: ThingSpeak
*  # Activity:
*    -> Data transmission and visualization
*       - ThingSpeak + Google Colab
*
*   ANGAZA ELIMU&ALX - IOT SCHOOL: Cohort 1, 2024
* --------------------------------------------------
*/

/*
 Support libraries -------------------------------------------------------------
*/
#include <WiFi.h>
#include "DHT.h"
#include <HTTPClient.h>
#include "secrets.h"      // For connection authentication credentials
// -----------------------------------------------------------------------------

/*
 Control Variables -------------------------------------------------------------
*/
//---- DHT settings
#define DHTPIN 7  // ESP32 GPIO pin connected to DHT22
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE); // dht object

//---- Onboard LED
#define ONBOARD_LED_PIN 13   // Used as a signal LED for wifi connection
// -----------------------------------------------------------------------------

/*
 Control Functions -------------------------------------------------------------
*/
//---- Initialize main instances execution
void setup() {

  Serial.begin(115200); // sets data exchange baude rate
  Serial.print("\nConnecting to ");
  Serial.println(WIFI_SSID);

  //---- Initialize WiFi connection
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  //---- Initialize signal LED mode
  pinMode(ONBOARD_LED_PIN, OUTPUT);
  digitalWrite(ONBOARD_LED_PIN, HIGH);      // Turn ON onboard LED

  //---- Initialize WiFi connection loader and signal LED
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
    digitalWrite(ONBOARD_LED_PIN, LOW);
    delay(250);
    digitalWrite(ONBOARD_LED_PIN, HIGH);
    delay(250);
  }

  Serial.println("\nWiFi connected\nIP address: ");
  Serial.println(WiFi.localIP());           //---- Print WiFi IP address

  //---- Turn OFF signal LED on WiFi connection
  digitalWrite(ONBOARD_LED_PIN, LOW);

  //---- Initialize DHT sensor
  dht.begin();
}

//---- iterative execution handler for main control instances
void loop() {
  //---- Read DHT22 sensor data
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  //---- Validate sensor reading
  if (!isnan(temperature) && !isnan(humidity)) {
    String data = "Temperature: " + String(temperature) + " °C, Humidity: " + String(humidity) + " %";
    Serial.println(data);
  } else{
    Serial.println("\nFailed to read from DHT sensor");
  }


  //---- Push data to ThingSpeak on successful WiFi connection
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http; // Create http object
    String url = String(THINGSPEAK_URL) + "?api_key=" + THINGSPEAK_API_KEY + "&field1=" + String(temperature) + "&field2=" + String(humidity);

    //---- Initialize and persist data transmission
    http.begin(url);
    int httpCode = http.GET();
    if (httpCode > 0) {
      Serial.println("Data sent to ThingSpeak successfully!\n");
    } else {
      Serial.println("Error in HTTP request");
    }
    http.end(); // end connection and release resources
  }

  delay(20000);  // Collect and send data every 20 seconds
}
