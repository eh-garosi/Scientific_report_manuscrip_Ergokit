/*
 * Intelligent Wearable Insole System — ESP32 Embedded Classifier
 * ===============================================================
 * Manuscript: "An intelligent wearable insole system for machine learning-based
 *              detection of high-risk load-lifting postures"
 * Journal   : Scientific Reports
 *
 * Description
 * -----------
 * This firmware runs on an ESP32-WROOM-32 microcontroller and performs
 * real-time lifting-posture classification using a Logistic Regression
 * classifier embedded as a C header file (model.h).
 *
 * Sensor pipeline (5 Hz):
 *   1. Read 12 FSR plantar-pressure channels via CD74HC4067 multiplexer.
 *   2. Read sagittal trunk flexion angle from MPU6050 IMU (T8 placement).
 *   3. Accumulate samples over a 1-second window (5 samples).
 *   4. Average the window → 13-dimensional feature vector.
 *   5. Run on-board inference with the embedded Logistic Regression model.
 *   6. Trigger feedback:
 *        Green LED → low-risk posture (Label 1)
 *        Red LED + Buzzer → high-risk posture (Label 2)
 *
 * Pin Assignment (matches Supplementary Table S2.2)
 * --------------------------------------------------
 * MPU6050 SDA  → GPIO 21  (4.7 kΩ pull-up)
 * MPU6050 SCL  → GPIO 22  (4.7 kΩ pull-up)
 * MUX S0       → GPIO 13  (LSB select)
 * MUX S1       → GPIO 12
 * MUX S2       → GPIO 14
 * MUX S3       → GPIO 27  (MSB select)
 * MUX SIG      → GPIO 34  (ADC1_CH6, input only)
 * MUX EN       → GPIO 32  (active LOW)
 * Green LED    → GPIO 25  (330 Ω series resistor)
 * Red LED      → GPIO 26  (330 Ω series resistor)
 * Buzzer       → GPIO 33  (PWM capable)
 *
 * FSR Channel Mapping (CD74HC4067 multiplexer channels 0–15)
 * -----------------------------------------------------------
 * MUX CH 0  → FSR1  (Left heel)
 * MUX CH 1  → FSR2  (Left navicular / mid-foot)
 * MUX CH 2  → FSR3  (Left 1st metatarsal head)
 * MUX CH 3  → FSR4  (Left 5th metatarsal head)
 * MUX CH 4  → FSR5  (Left hallux IP joint)
 * MUX CH 5  → FSR6  (Left 3rd toe MTP joint)
 * MUX CH 6  → FSR7  (Right heel)
 * MUX CH 7  → FSR8  (Right navicular / mid-foot)
 * MUX CH 8  → FSR9  (Right 1st metatarsal head)
 * MUX CH 9  → FSR10 (Right 5th metatarsal head)
 * MUX CH 10 → FSR11 (Right hallux IP joint)
 * MUX CH 11 → FSR12 (Right 3rd toe MTP joint)
 * MUX CH 12–15: unused
 *
 * Model Output Mapping
 * --------------------
 * The embedded Logistic Regression (model.h, generated via micromlgen /
 * Eloquent ML) returns a zero-based class index:
 *   classIdx = 0 → Label 1 (low-risk  posture) → Green LED
 *   classIdx = 1 → Label 2 (high-risk posture) → Red LED + Buzzer
 *
 * Authors: M. Vafadar, A.H. Jafari, F. Karbasi, E. Ghaffari
 * Ethics Approval: IR.IUMS.REC.1402.966
 */

#include "BluetoothSerial.h"
#include <Wire.h>
#include <MPU6050.h>
#include "model.h"   // Embedded Logistic Regression classifier

// ── Multiplexer control pins (CD74HC4067) ─────────────────────────────────
const int MUX_S0  = 13;   // LSB address select
const int MUX_S1  = 12;
const int MUX_S2  = 14;
const int MUX_S3  = 27;   // MSB address select
const int MUX_SIG = 34;   // Analog signal input (ADC1_CH6)
const int MUX_EN  = 32;   // Active LOW enable

// ── Feedback output pins ───────────────────────────────────────────────────
const int LED_GREEN = 25;  // Low-risk  indicator (330 Ω series resistor)
const int LED_RED   = 26;  // High-risk indicator (330 Ω series resistor)
const int BUZZER    = 33;  // High-risk auditory alert (PWM capable)

// ── Timing constants ───────────────────────────────────────────────────────
// Sampling interval: 200 000 µs = 200 ms → 5 Hz (satisfies Nyquist for
// quasi-static postural signals up to 2.5 Hz; see manuscript §4).
const unsigned long SAMPLE_INTERVAL_US     = 200000UL;
// Classification window: 1 000 000 µs = 1 s → 5 samples per decision.
const unsigned long PREDICTION_INTERVAL_US = 1000000UL;

// ── Number of FSR channels used ────────────────────────────────────────────
const int NUM_FSR = 12;

// ── Global state ──────────────────────────────────────────────────────────
BluetoothSerial SerialBT;
MPU6050 mpu;
Eloquent::ML::Port::LogisticRegression classifier;

unsigned long prevSampleMicros     = 0;
unsigned long prevPredictionMicros = 0;

float fsrSums[NUM_FSR] = {0.0f};
int   sampleCount      = 0;

// ── Helper: select multiplexer channel ────────────────────────────────────
float readMuxChannel(int channel) {
    // Write 4-bit address to S0–S3
    digitalWrite(MUX_S0, (channel >> 0) & 0x01);
    digitalWrite(MUX_S1, (channel >> 1) & 0x01);
    digitalWrite(MUX_S2, (channel >> 2) & 0x01);
    digitalWrite(MUX_S3, (channel >> 3) & 0x01);
    delayMicroseconds(10);               // brief settling time
    return (float)analogRead(MUX_SIG);
}

// ── Helper: compute trunk flexion angle from MPU6050 raw data ─────────────
float getTrunkAngle() {
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    // Sagittal (pitch) angle from accelerometer components
    return atan2f((float)ay, (float)az) * 180.0f / PI;
}

// ═══════════════════════════════════════════════════════════════════════════
void setup() {
    // Multiplexer pins
    pinMode(MUX_S0,  OUTPUT);
    pinMode(MUX_S1,  OUTPUT);
    pinMode(MUX_S2,  OUTPUT);
    pinMode(MUX_S3,  OUTPUT);
    pinMode(MUX_EN,  OUTPUT);
    digitalWrite(MUX_EN, LOW);   // Enable multiplexer (active LOW)

    // Feedback pins
    pinMode(LED_GREEN, OUTPUT);
    pinMode(LED_RED,   OUTPUT);
    pinMode(BUZZER,    OUTPUT);
    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_RED,   LOW);
    digitalWrite(BUZZER,    LOW);

    Serial.begin(115200);
    SerialBT.begin("ESP32_SmartInsole");   // Bluetooth device name

    // IMU initialisation
    Wire.begin();
    mpu.initialize();
    if (!mpu.testConnection()) {
        Serial.println("[ERROR] MPU6050 connection failed — check wiring.");
        while (true) { delay(1000); }
    }
    Serial.println("[INFO] MPU6050 connected.");
    Serial.println("[INFO] Smart Insole system ready. Streaming at 5 Hz.");
}

// ═══════════════════════════════════════════════════════════════════════════
void loop() {
    unsigned long now = micros();

    // ── 5 Hz sensor sampling ────────────────────────────────────────────
    if (now - prevSampleMicros >= SAMPLE_INTERVAL_US) {
        prevSampleMicros = now;
        sampleSensors();
    }

    // ── 1 Hz classification ─────────────────────────────────────────────
    if (now - prevPredictionMicros >= PREDICTION_INTERVAL_US) {
        prevPredictionMicros = now;
        classifyPosture();
    }
}

// ── Sample all 12 FSR channels and accumulate for averaging ───────────────
void sampleSensors() {
    float angleNow = getTrunkAngle();

    // Read FSR channels 0–11 (sequential; all 12 channels used)
    for (int ch = 0; ch < NUM_FSR; ch++) {
        float val = readMuxChannel(ch);
        fsrSums[ch] += val;
    }
    sampleCount++;

    // Stream real-time data over Serial and Bluetooth (CSV format)
    // Format: FSR1,FSR2,...,FSR12,TrunkAngle
    for (int i = 0; i < NUM_FSR; i++) {
        float instantVal = readMuxChannel(i);
        Serial.print(instantVal);
        SerialBT.print(instantVal);
        if (i < NUM_FSR - 1) { Serial.print(","); SerialBT.print(","); }
    }
    Serial.print(",");   SerialBT.print(",");
    Serial.print(angleNow);    SerialBT.print(angleNow);
    Serial.println();          SerialBT.println();
}

// ── Average window, run classifier, trigger feedback ──────────────────────
void classifyPosture() {
    if (sampleCount == 0) return;

    // Build 13-dimensional feature vector [FSR1..FSR12, TrunkAngle]
    float features[13];
    for (int i = 0; i < NUM_FSR; i++) {
        features[i] = fsrSums[i] / (float)sampleCount;
        fsrSums[i]  = 0.0f;   // reset accumulator
    }
    sampleCount   = 0;
    features[12]  = getTrunkAngle();   // current trunk angle for classification

    // Debug: print feature vector
    Serial.print("[FEATURES] ");
    for (int i = 0; i < 13; i++) {
        Serial.print(features[i], 2);
        if (i < 12) Serial.print(", ");
    }
    Serial.println();

    // ── On-board inference ─────────────────────────────────────────────
    // classifier.predict() returns a zero-based class index:
    //   0 → Label 1 (low-risk)   → Green LED on
    //   1 → Label 2 (high-risk)  → Red LED + Buzzer on
    int classIdx = classifier.predict(features);

    if (classIdx == 0) {
        // Low-risk posture
        digitalWrite(LED_GREEN, HIGH);
        digitalWrite(LED_RED,   LOW);
        noTone(BUZZER);
        Serial.println("[POSTURE] LOW-RISK  → Green LED");
    } else {
        // High-risk posture
        digitalWrite(LED_GREEN, LOW);
        digitalWrite(LED_RED,   HIGH);
        tone(BUZZER, 2000, 500);   // 2 kHz tone for 500 ms
        Serial.println("[POSTURE] HIGH-RISK → Red LED + Buzzer");
    }
}
