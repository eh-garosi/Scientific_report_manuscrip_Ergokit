# Scientific_report_manuscrip_Ergokit
Intelligent Wearable Insole System — Code Repository

**Manuscript:** "An intelligent wearable insole system for machine learning-based detection of high-risk load-lifting postures."  
**Journal:** *Scientific Reports*  
**Ethics Approval:** IR.IUMS.REC.1402.966  
**Authors:** M. Vafadar, A.H. Jafari, F. Karbasi, Ehsan Garosi* (eh.garoai@gmail.com)

---

## Overview

This repository contains all code used in the study. The system classifies lifting postures as **low-risk (Label 1)** or **high-risk (Label 2)** using a 13-dimensional feature vector:

- **FSR1–FSR12** — bilateral plantar-pressure readings from 12 force-sensitive resistors embedded in shoe insoles
- **TrunkAngle** — sagittal trunk flexion angle (degrees) measured by an MPU6050 IMU mounted at T8

Ground-truth labels were derived from the **UTAH back compressive force method** (threshold: 700 lbs / ≈ 3114 N).

---

## Repository Structure

```
├── Final_ML_Smartinsole.py     # Machine learning training and evaluation pipeline
├── sketch_smartinsole.ino      # ESP32 Arduino firmware for real-time classification
├── model.h                     # Embedded Logistic Regression classifier (C header)
├── datafinal.xlsx              # Dataset: 23 subjects × 37 conditions (not included — see Data Availability)
└── README.md
```

---

## File Descriptions

### `Final_ML_Smartinsole.py` — Machine Learning Pipeline

Trains and evaluates five supervised classifiers for binary posture classification.

**Models evaluated:**
| ID  | Algorithm             | Key hyperparameters (best via grid search) |
|-----|-----------------------|--------------------------------------------|
| LR  | Logistic Regression   | C = 0.1, L2 penalty, newton-cg solver      |
| SVM | Support Vector Machine| RBF kernel, C = 1, γ = 0.1                |
| KNN | K-Nearest Neighbors   | k = 3, Euclidean distance, distance-weighted |
| DT  | Decision Tree         | min_samples_split = 5, min_samples_leaf = 2 |
| RF  | Random Forest         | 100 estimators, min_samples_split = 5, min_samples_leaf = 2 |

**Study design implemented in code:**
- 23 subjects; participant-wise 78 %/22 % train–test split (18 subjects train, 5 subjects test)
- **5-fold stratified cross-validation** on the training subset
- Per-subject z-score normalization applied before splitting (to reflect realistic per-device calibration)
- Incremental feature addition from k = 1 to k = 13, ordered by Random Forest MDI importance
- Outputs saved to `results/` as CSV files

**Reported performance (Logistic Regression, k = 13 features, independent test set):**

| Metric      | Value   |
|-------------|---------|
| Accuracy    | 91.17 % |
| Sensitivity | 93.75 % |
| Specificity | 87.83 % |
| AUC         | 0.94    |

> **Note on reproducibility:** Results may vary slightly across runs because the participant assignment to train/test subsets is sensitive to random seed interactions with the small sample (n = 23). The reported values correspond to `SEED = 42`. The random seed is fixed throughout the pipeline to maximize reproducibility.

---

### `sketch_smartinsole.ino` — ESP32 Firmware

Real-time firmware for the assembled intelligent insole system.

**Hardware (ESP32-WROOM-32):**

| Function         | ESP32 Pin | Notes                        |
|------------------|-----------|------------------------------|
| MPU6050 SDA      | GPIO 21   | 4.7 kΩ pull-up               |
| MPU6050 SCL      | GPIO 22   | 4.7 kΩ pull-up               |
| MUX S0 (LSB)     | GPIO 13   |                              |
| MUX S1           | GPIO 12   |                              |
| MUX S2           | GPIO 14   |                              |
| MUX S3 (MSB)     | GPIO 27   |                              |
| MUX SIG          | GPIO 34   | ADC1_CH6, input only         |
| MUX EN           | GPIO 32   | Active LOW                   |
| Green LED        | GPIO 25   | 330 Ω series resistor        |
| Red LED          | GPIO 26   | 330 Ω series resistor        |
| Buzzer           | GPIO 33   | PWM capable                  |

**FSR channel mapping (CD74HC4067 multiplexer):**

| MUX Channel | Sensor | Anatomical Location (left foot sensors 1–6, right 7–12) |
|-------------|--------|----------------------------------------------------------|
| CH 0        | FSR1   | Left heel                                                |
| CH 1        | FSR2   | Left navicular / mid-foot                               |
| CH 2        | FSR3   | Left 1st metatarsal head                                |
| CH 3        | FSR4   | Left 5th metatarsal head                                |
| CH 4        | FSR5   | Left hallux IP joint                                    |
| CH 5        | FSR6   | Left 3rd toe MTP joint                                  |
| CH 6        | FSR7   | Right heel                                              |
| CH 7        | FSR8   | Right navicular / mid-foot                              |
| CH 8        | FSR9   | Right 1st metatarsal head                               |
| CH 9        | FSR10  | Right 5th metatarsal head                               |
| CH 10       | FSR11  | Right hallux IP joint                                   |
| CH 11       | FSR12  | Right 3rd toe MTP joint                                 |

**Operation:**
- Sensors sampled at **5 Hz** (200 ms interval)
- Classification performed every **1 second** (5-sample averaged window)
- Latency < 200 ms from sample to feedback
- Feedback: **Green LED** = low-risk; **Red LED + 2 kHz buzzer** = high-risk
- Data streamed in real time via Bluetooth (CSV: FSR1–FSR12, TrunkAngle)

**Required Arduino libraries:**
- `BluetoothSerial` (built into ESP32 Arduino core)
- `Wire` (built in)
- `MPU6050` by Electronic Cats (install via Arduino Library Manager)

---

### `model.h` — Embedded Classifier

A C header file containing the trained Logistic Regression model, generated using [micromlgen / Eloquent ML](https://eloquentarduino.com). It performs on-board inference without requiring floating-point libraries beyond standard C.

**Class index mapping:**

| `predict()` return value | Training label | Posture class | Feedback           |
|--------------------------|----------------|---------------|--------------------|
| `0`                      | Label 1        | Low-risk      | Green LED          |
| `1`                      | Label 2        | High-risk     | Red LED + Buzzer   |

The Eloquent ML library re-indexes training labels (1, 2) to zero-based indices (0, 1) in sorted order. This mapping is documented in both `model.h` and `sketch_smartinsole.ino`.

---

## How to Run the Machine Learning Pipeline

### Requirements

```
Python >= 3.9
numpy
pandas
scikit-learn >= 1.0
openpyxl
```

Install dependencies:

```bash
pip install numpy pandas scikit-learn openpyxl
```

### Usage

1. Place `datafinal.xlsx` in the same directory as `Final_ML_Smartinsole.py` (or update `DATA_PATH` in the script).
2. Run:

```bash
python Final_ML_Smartinsole.py
```

3. Results are saved to `results/`:

| File                        | Contents                                              |
|-----------------------------|-------------------------------------------------------|
| `feature_importance.csv`    | MDI importance scores for all 13 features             |
| `results_LR.csv`            | LR performance at k = 1 → 13 features                 |
| `results_SVM.csv`           | SVM performance at k = 1 → 13                         |
| `results_KNN.csv`           | KNN performance at k = 1 → 13                         |
| `results_DT.csv`            | DT performance at k = 1 → 13                          |
| `results_RF.csv`            | RF performance at k = 1 → 13                          |
| `mean_performance_by_k.csv` | Mean performance across all 5 models at each k        |
| `best_model_per_k.csv`      | Best model selected at each k (accuracy then AUC)     |

---

## How to Flash the Embedded Classifier

1. Open `sketch_smartinsole.ino` in the **Arduino IDE** (v2.x recommended).
2. Install the **ESP32 board package** via Board Manager (`esp32` by Espressif Systems).
3. Install the **MPU6050** library by Electronic Cats via Library Manager.
4. Ensure `model.h` is in the **same folder** as `sketch_smartinsole.ino`.
5. Select board: **ESP32 Dev Module** (or ESP32-WROOM-32).
6. Select the correct COM port and upload.

To update the embedded classifier after retraining, regenerate `model.h` using [micromlgen](https://github.com/eloquentarduino/micromlgen):

```python
from micromlgen import port
from sklearn.linear_model import LogisticRegression
# ... train your model ...
print(port(clf, classmap={0: 'low_risk', 1: 'high_risk'}))
```

---

## Data Availability

The dataset (`datafinal.xlsx`) contains plantar-pressure and trunk-angle recordings from 23 participants across 37 lifting conditions. Due to participant privacy, the data are available from the corresponding author upon reasonable request, as stated in the manuscript's Data Availability statement.

---

## Citation

If you use this code, please cite the original manuscript:

> Vafadar M, Jafari AH, Karbasi F, garosi E. An intelligent wearable insole system for machine learning-based detection of high-risk load-lifting postures. *Scientific Reports*. [Year]; [Volume]:[Article number]. 
---

## License

This code is released for academic and research use. For commercial applications, please contact the corresponding author.

---

## Contact

**Corresponding author:** Ehsan Garosi  
Iran University of Medical Sciences  
[Contact details as provided in the published manuscript]

