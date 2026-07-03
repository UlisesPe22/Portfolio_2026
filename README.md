# 🐝Beehive Anomaly Detection System

**Remote, real-time monitoring for beekeepers — catching queenless hives before they collapse.**

A full-stack IoT + Deep Learning system that ingests live beehive sensor data (temperature, humidity, pressure), runs it through an LSTM neural network, and gives beekeepers a live dashboard telling them whether their hive is healthy — without opening the box.

<!-- 🎥 DEMO: drop your GIF/video here, e.g. -->
<!-- ![Dashboard Demo](assets/dashboard-demo.gif) -->

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Backend-000000?logo=flask&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite&logoColor=white)
![Status](https://img.shields.io/badge/Status-Prototype-yellow)

---

## Table of Contents

- [Overview](#overview)
- [The Problem: Why a Queenless Hive Matters](#the-problem-why-a-queenless-hive-matters)
- [Data & IoT Sensors](#data--iot-sensors)
- [Normal vs. Anomaly](#normal-vs-anomaly)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model: LSTM on Sliding Windows](#model-lstm-on-sliding-windows)
- [Results](#results)
- [System Architecture](#system-architecture)
- [API Reference](#api-reference)
- [Getting Started](#getting-started)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Author](#author)

---

## Overview

Beekeepers can't check every hive, every day — and by the time a problem is visible from the outside, it's often too late. **Hive Guardian** turns a hive into a connected device: a **BME280 sensor** streams temperature, humidity, and pressure readings, and a trained **LSTM model** continuously analyzes the pattern of those readings to flag whether the colony looks healthy ("queen producing") or at risk ("queen not producing / not accepted").

The result is a live web dashboard a beekeeper can check from their phone or laptop — remote hive health monitoring, without disturbing the bees.

This project covers the full pipeline: exploratory data analysis → feature engineering → deep learning model → Flask API → live dashboard → a simulator that replays real historical data to demonstrate the system end-to-end.

## The Problem: Why a Queenless Hive Matters

A hive's queen is its entire reason for organizational cohesion. When she is missing or not accepted by the colony:

- **Hive collapse** — the colony's health, cohesion, and organization deteriorate rapidly
- **Financial and time costs** — every day the problem goes undetected shortens the window to save the hive; left too long, the colony becomes unrecoverable

Early detection is the difference between a quick intervention and losing the hive entirely — which is the entire premise for automating this with sensors + ML instead of relying on manual inspection.

## Data & IoT Sensors

- **Dataset:** Yang, A. (2023). *Beehive Sounds* [Data set]. Kaggle. https://www.kaggle.com/datasets/annajyang/beehive-sounds
- **IoT Device:** BME280 temperature / humidity / pressure sensor, mounted to stream live hive conditions

## Normal vs. Anomaly

| State | Meaning |
|---|---|
| 🟢 **Normal** | The queen is accepted by the hive and satisfying the colony's needs |
| 🔴 **Anomaly** | The queen is either not present in the hive, or present but not accepted — and the hive's needs are not being met |

## Exploratory Data Analysis

A PCA analysis was run to understand which sensor readings drive the most variance in hive state.

**Principal Component 1 (PC1) — Key Features**

| Feature | Loading |
|---|---|
| Weather Temperature | 0.48 |
| Weather Humidity | 0.48 |
| Hive Humidity | 0.37 |
| Hive Temperature | 0.36 |
| Wind Speed | 0.33 |

**Principal Component 2 (PC2) — Key Features**

| Feature | Loading |
|---|---|
| Hive Pressure | 0.63 |
| Weather Pressure | 0.63 |
| Hive Temperature | 0.33 |
| Weather Humidity | 0.21 |
| Weather Temperature | 0.19 |

<!-- 📊 IMAGE: Hive 3 — normal/anomalous pattern detection over time -->
<!-- ![Hive 3 Pattern Detection](assets/hive3-pattern-detection.png) -->

<!-- 📊 IMAGE: Hive 3 — sensor data registered outside the hive -->
<!-- ![Hive 3 External Sensors](assets/hive3-external-sensors.png) -->

**Conclusions from EDA:**
- `hive_temperature`, `hive_humidity`, and `hive_pressure` were selected as the predictive features
- Target variable: `queen_status`

## Feature Engineering

- **Scaling:** all features normalized with a **MinMax scaler**
- **Engineered features:** 3 additional features capturing short/mid/long-term drift — the Euclidean distance between the current point *x* and *x-n*, for **n = 3, 5, and 10**

## Model: LSTM on Sliding Windows

The model doesn't classify single readings — it classifies **patterns over time**, using a sliding window approach:

- Each window contains **10 chronologically ordered points**
  - Window 1: indices 0–9, Window 2: indices 1–10, and so on
- **Labeling rule:** a window is labeled *anomaly* if the most recent point in that window is an anomaly
- **Tensor shape:** `(n_windows, window_size, features)`
  - Training tensor: `(179, 10, 6)`
  - Test tensor: `(87, 10, 6)`
- **Algorithm:** LSTM (Long Short-Term Memory) — chosen for its ability to learn temporal dependencies in sequential sensor data

## Results

`0` = Normal &nbsp;&nbsp;|&nbsp;&nbsp; `1` = Anomaly

<!-- 📊 IMAGE: confusion matrix / accuracy plot -->
<!-- ![Model Results](assets/model-results.png) -->

<!-- TODO: add your accuracy / precision / recall / F1 numbers here once you have them written down, e.g.: -->
<!-- | Metric | Score |
|---|---|
| Accuracy | 0.xx |
| Precision | 0.xx |
| Recall | 0.xx |
| F1-score | 0.xx | -->

## System Architecture

```
simulator.py ──POST JSON──▶ /api/hive ──▶ RollingBuffer (last 20 readings)
                                              │
                                              ▼
                              predict.py (build 10-step window,
                              scale, compute Euclidean-distance
                              features) ──▶ hive_model.h5 ──▶ prediction
                                              │
   dashboard.html ◀──poll every 2s── /api/hive/latest + /history
```

**Project structure:**

```
hive_app/
├── app.py                  # Flask app factory, entry point
├── config.py                # Env-driven config (DB path, host, port, model path)
├── hive_data.db              # SQLite — 286 rows of historical sensor readings
├── api/
│   ├── __init__.py
│   └── routes.py             # POST /api/hive, GET /, /api/hive/latest, /api/hive/history
├── buffer/
│   ├── __init__.py
│   └── buffer.py              # RollingBuffer — thread-safe deque, last 20 readings
├── models/
│   ├── __init__.py
│   ├── predict.py             # Core ML logic: windowing, scaling, inference
│   └── hive_model.h5           # Trained Keras/TensorFlow LSTM model
├── templates/
│   └── dashboard.html          # Dark-themed live dashboard (HTML/CSS/vanilla JS)
└── tests/
    └── simulator.py             # Replays historical DB rows as a live sensor feed
```

| Component | Responsibility |
|---|---|
| `app.py` | Flask app factory, registers blueprints, initializes DB |
| `config.py` | Central config: DB path, host `0.0.0.0`, port `5000`, `MODEL_PATH` |
| `api/routes.py` | All HTTP endpoints (see [API Reference](#api-reference)) |
| `buffer/buffer.py` | In-memory rolling window of the last 20 readings — the model's actual input source |
| `models/predict.py` | Builds the 10-step window, scales features, computes distance features, runs inference |
| `models/database.py` | SQLite connection helpers, table initialization |
| `templates/dashboard.html` | Live dashboard — polls the API every 2s, shows prediction, confidence, stats, recent readings |
| `tests/simulator.py` | Not a unit test — replays real historical rows into the live API to simulate a sensor stream |

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/hive` | Receive a new sensor reading, push to buffer, run prediction |
| `GET` | `/` | Serve the dashboard |
| `GET` | `/api/hive/latest` | Latest reading + prediction (polled by the dashboard) |
| `GET` | `/api/hive/history?n=20` | Last N readings, for the recent-entries table |

## Getting Started

```powershell
# 1. Activate the virtual environment
C:\Users\perez\Documents\Bee_sensors\hive_venv\Scripts\Activate.ps1

# 2. Install dependencies (no requirements.txt yet — see Known Limitations)
pip install flask tensorflow numpy pandas requests

# 3. Run the app
python app.py

# 4. In a second terminal, feed it live-looking data from the historical DB
python tests/simulator.py
```

Then open `http://localhost:5000` to see the live dashboard.

> ⚠️ **Warm-up period:** the first ~10 posted readings won't produce a prediction ("Not enough rows to build a window") — this is expected while the rolling buffer fills up.

## Known Limitations

Being upfront about the current state, since this is a working prototype rather than a production system:

- **No `requirements.txt` yet** — dependencies must be installed manually
- **Config/code mismatch** — `app.py` defaults to `hive_model.tflite`, but `config.py` correctly points to `hive_model.h5` (the file that actually exists), so `config.MODEL_PATH` wins in practice
- **Readings aren't persisted** — `POST /api/hive` only writes to the in-memory buffer, not to SQLite; the DB is currently read-only, used only as the simulator's source data
- **TensorFlow cold-start** — the first request is slow while the model loads; tested under both Python 3.11 and 3.13, but 3.11 is the safer target for TensorFlow compatibility

## Roadmap

- [ ] Add `requirements.txt` / `pyproject.toml` for reproducible installs
- [ ] Persist incoming readings to SQLite (currently buffer-only)
- [ ] Deploy model as `.tflite` for lighter-weight inference
- [ ] Real IoT ingestion (replace simulator with live BME280 stream, e.g. via MQTT)
- [ ] Alerting (email/SMS/push notification when an anomaly is detected)
- [ ] Multi-hive dashboard view for beekeepers managing several colonies

## Author

**Ulises Ernesto Pérez Espinosa**
Applied AI & Data Science student | Data pipelines, ML, and full-stack development

<!-- Add: GitHub | LinkedIn | Portfolio link -->
