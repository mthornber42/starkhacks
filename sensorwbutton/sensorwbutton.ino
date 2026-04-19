#include <Arduino.h>
#include <WiFi.h>
#include <Adafruit_MPU6050.h>
#include <Wire.h>
#include <ArduinoJson.h>

// ---------------- WIFI ----------------
const char* ssid = "TP-Link_0869";
const char* password = "34747193";
// const char* ssid = "Green Machine";
// const char* password = "tbomrone";
// const char* ssid = "StarkHacks-2";
// const char* password = "StarkHacks2026";


// ---------------- API ----------------
const char* host = "192.168.0.102";  // your server IP
const int port = 8000;

// ---------------- BUTTON ----------------
const int BUTTON_PIN = 10;



// ---------------- STATE ----------------
enum State { IDLE, RECORDING };
State state = IDLE;

bool lastButtonState = HIGH;
bool stopRequested = false;

unsigned long startTime = 0;
unsigned long lastSampleTime = 0;

// ---------------- JSON ----------------
DynamicJsonDocument doc(65536);
JsonArray samples;

// ---------------- SENSOR --------------
Adafruit_MPU6050 mpu;

// =====================================================
// SETUP
// =====================================================
void setup() {
  Serial.begin(115200);
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (true) {
      delay(10);
    }
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  delay(100);


  pinMode(BUTTON_PIN, INPUT);

  connectWiFi();
}

// =====================================================
// LOOP
// =====================================================
void loop() {
  checkButton();

  if (state == RECORDING) {
    recordIMU();
  }
}

// =====================================================
// WIFI
// =====================================================
void connectWiFi() {
  WiFi.begin(ssid, password);

  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected!");
}

// =====================================================
// BUTTON HANDLER W EDGE DETECTION
// =====================================================
void checkButton() {
  bool current = digitalRead(BUTTON_PIN);

  // detect press
  if (lastButtonState == LOW && current == HIGH) {

    if (state == IDLE) {
      startRecording();
    } else if (state == RECORDING) {
      stopRequested = true;
    }
  }

  lastButtonState = current;
}

// =====================================================
// START RECORDING
// =====================================================
void startRecording() {

  Serial.println("Recording started");

  doc.clear();
  doc["samples"] = JsonArray();
  samples = doc["samples"].to<JsonArray>();

  startTime = millis();
  lastSampleTime = 0;

  stopRequested = false;
  state = RECORDING;
}

// =====================================================
// RECORD IMU
// =====================================================
void recordIMU() {
  unsigned long now = millis();

  // stop conditions
  if (now - startTime >= 2000 || stopRequested) {
    finishRecording();
    return;
  }

  // sample rate at 50Hz
  if (now - lastSampleTime < 20) return;
  lastSampleTime = now;

  // IMU READ
  JsonObject sample = samples.createNestedObject();

  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  sample["a_x"] = a.acceleration.x;
  sample["a_y"] = a.acceleration.y;
  sample["a_z"] = a.acceleration.z;
  sample["g_x"] = g.gyro.x;
  sample["g_y"] = g.gyro.y;
  sample["g_z"] = g.gyro.z;
}

// =====================================================
// FINISH + SEND
// =====================================================
void finishRecording() {
  Serial.println("Recording finished");

  state = IDLE;

  String payload;
  serializeJson(doc, payload);

  Serial.println("Payload:");
  Serial.println(payload);

  sendToAPI(payload);
}

// =====================================================
// HTTP POST
// =====================================================
void sendToAPI(String json) {
  WiFiClient client;

  if (!client.connect(host, port)) {
    Serial.println("Connection failed");
    return;
  }

  client.print("POST /imu HTTP/1.1\r\n");
  client.print("Host: ");
  client.print(host);
  client.print("\r\n");

  client.print("Content-Type: application/json\r\n");
  client.print("Connection: close\r\n");
  client.print("Content-Length: ");
  client.print(json.length());
  client.print("\r\n");

  client.print("\r\n");   // END HEADERS

  client.print(json);     // IMPORTANT: NO println

  delay(10);
  client.stop();

  Serial.println("Sent to API");
}














