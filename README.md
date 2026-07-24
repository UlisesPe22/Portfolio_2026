# 🐝Beehive Anomaly Detection System
Beekeepers can't check every hive, every day — and by the time a problem is visible from the outside, it's often too late. This solution turns a hive into a connected device: a **BME280 sensor** that creates **temperature, humidity, and pressure** readings, and a trained **LSTM model** continuously analyzes the pattern of those readings to flag whether the colony looks healthy ("queen producing") or at risk ("queen not producing").

The result is a live web dashboard a beekeeper can check from their phone or laptop — remote hive health monitoring, without disturbing the bees.
## 📑 Table of Contents

- [The Problem: Why a Queenless Hive Matters](#the-problem-why-a-queenless-hive-matters)
- [Data & IoT Sensors](#data--iot-sensors)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
- [Identify patterns and properly encode the information](#identify-patterns-to-properly-encode-the-information)
  - [Feature Engineering](#feature-engineering)
- [Model: LSTM on Sliding Windows // Description of Supervised Training](#model-lstm-on-sliding-windows--description-of-supervised-training)
  - [Results](#results)
- [System Architecture](#system-architecture)
- [Getting Started](#getting-started)
 
## The Problem: Why a Queenless Hive Matters

A hive's queen is its entire reason for organizational cohesion. When she is missing or not accepted by the colony:

- **Hive collapse** — the colony's health, cohesion, and organization deteriorate rapidly
- **Financial and time costs** — every day the problem goes undetected shortens the window to save the hive; left too long, the colony becomes unrecoverableg

![Dashboard Demo](assets_readme/dashboard_anomaly_detection_app.gif)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Backend-000000?logo=flask&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite&logoColor=white)


# Data & IoT Sensors

- **Dataset:** Yang, A. (2023). *Beehive Sounds* [Data set]. Kaggle. https://www.kaggle.com/datasets/annajyang/beehive-sounds
- **IoT Device:** BME280 senor readings of: temperature / humidity / pressure sensor, mounted to stream live hive conditions

| State | Meaning |
|---|---|
| 🟢 **Normal** | The queen is accepted by the hive and satisfying the colony's needs |
| 🔴 **Anomaly** | The queen is either not present in the hive, or present but not accepted — and the hive's needs are not being met |

## Exploratory Data Analysis

A PCA analysis was run to understand which sensor readings drive the most variance in hive state. The results showed temperature and humidity inside the hive are the most significant featrues and therefore the anomlay predction system can be developed with that data.
**PCA — Key Features by Principal Component**

| Principal Component | Feature | Loading |
|---|---|---|
| PC1 | Weather Temperature | 0.48 |
| PC1 | Weather Humidity | 0.48 |
| PC1 | Hive Humidity | 0.37 |
| PC1 | Hive Temperature | 0.36 |
| PC1 | Wind Speed | 0.33 |
| PC2 | Hive Pressure | 0.63 |
| PC2 | Weather Pressure | 0.63 |
| PC2 | Hive Temperature | 0.33 |
| PC2 | Weather Humidity | 0.21 |
| PC2 | Weather Temperature | 0.19 |

**Conclusions from EDA:**
- `temperature`, `humidity` and 'pressure' were selected as the predictive features
- Target variable: `queen_status`
- **As shown in the image above, the information that differentiates the two states comes from the distance between points and from the height difference between normal and anomalous points.**

# Identify patterns to properly encode the information

<!-- 📊 normal/anomalous pattern detection over time -->
<img width="860" height="420" alt="image" src="https://github.com/user-attachments/assets/6194bdbf-260f-48a8-aae8-bc8dc9328a9a" />

## Feature Engineering

- **Scaling:** all features normalized with a **MinMax scaler**. This is used to preserve and encode the distance information between points.
- **Engineered features:** 3 additional features capturing short/mid/long-term drift are added to train the anomaly detection model — the Euclidean distance between the current point *x* and *x-n*, for **n = 3, 5, and 10**
- <img width="1100" height="120" alt="image" src="https://github.com/user-attachments/assets/da9d164d-b4dd-40ca-bd8b-109ac077f1f6" />


# Model: LSTM on Sliding Windows // Description of Supervised Training 

The dataset came with labeled ground truth (queen_status), so the model was trained in a supervised fashion — learning to map sequences of sensor readings to a known Normal/Anomaly label. This training phase is what produced hive_model.h5

- Each window contains **10 chronologically ordered points**
  - Window 1: indices 0–9, Window 2: indices 1–10, and so on
- **Labeling rule:** a window is labeled *anomaly* if the most recent point in that window is an anomaly
- **Tensor shape:** `(n_windows, window_size, features)`
  - Training tensor: `(179, 10, 6)`
  - Test tensor: `(87, 10, 6)`
- **Algorithm:** LSTM (Long Short-Term Memory) — chosen for its ability to learn temporal dependencies in sequential sensor data

## Results
`0` = Normal &nbsp;&nbsp;|&nbsp;&nbsp; `1` = Anomaly

<img width="400" alt="image" src="https://github.com/user-attachments/assets/5e086620-e16d-46b8-8340-7ea92bbbc9cc" /> <img width="400" alt="image" src="https://github.com/user-attachments/assets/5e0f31b8-2fcf-4715-8ad6-ed3e66cfd2d5" />

# System Architecture

| Component | Responsibility |
|---|---|
| `app.py` | Flask app factory, registers blueprints, initializes DB |
| `config.py` | Central config: DB path, host `0.0.0.0`, port `5000`, `MODEL_PATH` |
| `api/routes.py` | All HTTP endpoints|
| `buffer/buffer.py` | In-memory rolling window of the last 20 readings — the model's actual input source |
| `models/predict.py` | Builds the 10-step window, scales features, computes distance features, runs inference |
| `models/database.py` | SQLite connection helpers, table initialization |
| `templates/dashboard.html` | Live dashboard — polls the API every 2s, shows prediction, confidence, stats, recent readings |
| `tests/simulator.py` | Not a unit test — replays real historical rows into the live API to simulate a sensor stream |

# Getting Started

```
# 1. Create a virtual environment

# 2. Install dependencies
pip install requirements.txt

# 3. Run the app
python app.py

# 4. In a second terminal, feed it live-looking data from the historical DB
python tests/simulator.py
```
Then open `http://localhost:5000` to see the live dashboard.

