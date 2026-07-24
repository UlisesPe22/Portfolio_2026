# hive_app/models/predict.py
import os
import sqlite3
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from tensorflow.keras.models import load_model
from buffer.buffer import BUFFER
import config

# Defaults
MODEL_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "hive_model.h5")  
WINDOW_SIZE = 10
SENSOR_COLS_SCALED = ["hive_temp_scaled", "hive_humidity_scaled", "hive_pressure_scaled"]
EUCLID_LAGS = [3, 5, 10]


FALLBACK_SCALER = {
    "hive_temp": {"min": 15.5, "max": 55.62},
    "hive_humidity": {"min": 7.23, "max": 93.47},
    "hive_pressure": {"min": 1003.54, "max": 1015.97},
}


# ------------------------------
# Build DataFrame from buffer + payload
# ------------------------------
def build_df(current_payload: Dict[str, Any], buffer_items: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
    items = []
    if buffer_items:
        items.extend(buffer_items)
    if current_payload:
        if not items or items[-1].get("id") != current_payload.get("id"):
            items.append(current_payload)

    rows = []
    for it in items:
        try:
            rows.append({
                "hive number": int(it.get("hive_number")),
                "time": pd.to_datetime(it.get("time")),
                "hive_temp": float(it.get("hive_temp")),
                "hive_humidity": float(it.get("hive_humidity")),
                "hive_pressure": float(it.get("hive_pressure")),
                "status": int(it.get("hive_status")),
            })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    return df



# ------------------------------
# Scaling
# ------------------------------
def scaled_df(df: pd.DataFrame, scaler_params: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
    if scaler_params is None:
        scaler_params = FALLBACK_SCALER

    def _minmax_scale_series(series: pd.Series, xmin: float, xmax: float) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        denom = float(xmax) - float(xmin)
        if denom == 0.0:
            denom = 1.0
        scaled = (s - float(xmin)) / denom
        return scaled.clip(lower=0.0, upper=1.0)

    mapping = [
        ("hive_temp", "hive_temp", "hive_temp_scaled"),
        ("hive_humidity", "hive_humidity", "hive_humidity_scaled"),
        ("hive_pressure", "hive_pressure", "hive_pressure_scaled"),
    ]

    for df_col, param_key, out_col in mapping:
        if df_col in df.columns:
            params = scaler_params.get(param_key)
            if params is None or ("min" not in params) or ("max" not in params):
                continue
            xmin = params["min"]
            xmax = params["max"]
            df[out_col] = _minmax_scale_series(df[df_col], xmin, xmax)
            df[out_col.replace("_", " ")] = df[out_col]

    return df

# ------------------------------
# Windows builder
# ------------------------------
def build_hive_windows(df, window_size=WINDOW_SIZE, sensor_cols=None, euclid_lags=None):
    import numpy as np
    if sensor_cols is None:
        sensor_cols = SENSOR_COLS_SCALED
    if euclid_lags is None:
        euclid_lags = EUCLID_LAGS

    df_h = df.copy()
    df_h["time"] = pd.to_datetime(df_h["time"])
    df_h = df_h.sort_values("time").reset_index(drop=True)

    required_cols = sensor_cols 
    missing = [c for c in required_cols if c not in df_h.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")

    for lag in euclid_lags:
        shifted = df_h[sensor_cols].shift(lag)
        df_h[f"euclid_x_{lag}"] = np.sqrt(((df_h[sensor_cols] - shifted) ** 2).sum(axis=1))

    euclid_cols = [f"euclid_x_{lag}" for lag in euclid_lags]
    df_h = df_h.dropna(subset=euclid_cols).reset_index(drop=True)

    sensor_windows, dist_features, combined_windows = [], [], []
    n_rows = len(df_h)
    n_windows = n_rows - window_size + 1
    if n_windows <= 0:
        raise ValueError(f"Not enough rows to build a window. Need >= {window_size}, got {n_rows}.")

    for start in range(n_windows):
        end = start + window_size
        win = df_h.iloc[start:end]
        sensors_arr = win[sensor_cols].values.astype(float)
        last_row = win.iloc[-1]
        dists = last_row[euclid_cols].values.astype(float)
        dist_matrix = np.zeros((window_size, len(euclid_lags)), dtype=float)
        dist_matrix[-1, :] = dists
        combined = np.concatenate([sensors_arr, dist_matrix], axis=1)
        sensor_windows.append(sensors_arr)
        dist_features.append(dists)
        combined_windows.append(combined)

    X_sensors = np.stack(sensor_windows)
    X_dists = np.stack(dist_features)
    X_combined = np.stack(combined_windows)
 
    return X_sensors, X_dists, X_combined
# ------------------------------
# Load model
# ------------------------------
def load_model_path(model_path: str = None):
    model_path = model_path or MODEL_PATH_DEFAULT
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model not found at {model_path}")
    model = load_model(model_path)
    return model

# ------------------------------
# Predict function 
# ------------------------------
def predict_from_payload(current_payload: Dict[str, Any],
                         model_path: Optional[str] = None,
                         ) -> Dict[str, Any]:
    buffer_items = BUFFER.get_all()
    df = build_df(current_payload, buffer_items)
    
    df = scaled_df(df, scaler_params=FALLBACK_SCALER)


    X_sensors, X_dists, X_combined = build_hive_windows(df, window_size=WINDOW_SIZE)

    # pick last window as input
    input_tensor = X_combined[-1:] 

    model = load_model_path(model_path)
    prediction = model.predict(input_tensor)
    
    # Flatten prediction to scalar 
    try:
        prediction_value = float(prediction[0][0]) if prediction.ndim > 1 else float(prediction[0])
    except Exception:
        prediction_value = prediction.tolist()

    print("[predict] model raw output:", prediction.tolist(), "prediction_value:", prediction_value, flush=True)

    return {
        "model_path": model_path or MODEL_PATH_DEFAULT,
        "input_shape": list(input_tensor.shape),
        "output": prediction.tolist(),
        "prediction": prediction_value
    }
