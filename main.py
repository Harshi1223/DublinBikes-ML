# main.py — Dublin Bikes Shortage Prediction API
# XGBoost + Logistic Regression + Random Forest + LSTM active
#
# Fully live where possible: bikes/docks, lag features,
# cascade features, weather, and time features are all
# computed from live sources once enough data has
# accumulated - falling back to historical values only
# where live data genuinely isn't available yet.

from fastapi import FastAPI
import xgboost as xgb
import joblib
import pandas as pd
import numpy as np
import requests
import os
import pickle
import json
import threading
import time as time_module
from collections import defaultdict
from datetime import datetime
import tensorflow as tf
import keras

model = keras.Sequential()
app = FastAPI(title="Dublin Bikes Shortage Prediction API")

# ---------- Load models ----------

xgb_30 = xgb.Booster()
xgb_30.load_model("models/candidate_booster_30min.json")
xgb_60 = xgb.Booster()
xgb_60.load_model("models/booster_60min.json")

lr_30 = joblib.load("models/lr_best_30min.pkl")
lr_60 = joblib.load("models/lr_best_60min.pkl")
lr_scaler = joblib.load("models/lr_scaler.pkl")

rf_30 = joblib.load("models/random_forest_target_30min_final.pkl")
rf_60 = rf_30  # TEMPORARY: 60min RF file was corrupted, using 30min as fallback
RF_THRESHOLD = 0.7

lstm_30 = keras.models.load_model("models/lstm_target_30min_FINAL.keras")
lstm_60 = keras.models.load_model("models/lstm_target_60min_FINAL.keras")
lstm_scaler_30 = joblib.load("models/lstm_30min_scaler_FINAL.pkl")
lstm_scaler_60 = joblib.load("models/lstm_60min_scaler_FINAL.pkl")

with open("models/lstm_30min_feature_columns.json") as f:
    lstm_feature_cols = json.load(f)

with open("models/lstm_30min_best_thresholds.json") as f:
    lstm_threshold_30 = json.load(f)
with open("models/lstm_60min_best_thresholds.json") as f:
    lstm_threshold_60 = json.load(f)

LSTM_SEQ_LEN = 6

data = pd.read_csv("data/test_features_v2.csv")
feature_cols = [c for c in data.columns if c not in ['timestamp', 'target_now', 'target_30min', 'target_60min']]

with open("data/neighbour_map.pkl", "rb") as f:
    NEIGHBOUR_MAP = pickle.load(f)

CLASS_NAMES = ["Normal", "Bike Shortage", "Dock Shortage"]

BEST_MODEL_BY_METRIC = "xgboost"
BEST_MODEL_REASON = "Highest Macro F1 on held-out test set (0.8013 vs 0.7770 LR, 0.784 RF)"


def apply_rf_threshold(proba_row, threshold=RF_THRESHOLD):
    p_normal, p_bike, p_dock = proba_row
    flag_bike = p_bike >= threshold
    flag_dock = p_dock >= threshold
    if flag_bike and flag_dock:
        return 1 if p_bike >= p_dock else 2
    elif flag_bike:
        return 1
    elif flag_dock:
        return 2
    return 0


# ---------- Live JCDecaux integration ----------

JCDECAUX_API_KEY = os.getenv("JCDECAUX_API_KEY")
LIVE_STATION_CACHE = {}
LIVE_HISTORY_BUFFER = defaultdict(list)


def poll_jcdecaux_loop():
    while True:
        if JCDECAUX_API_KEY:
            try:
                url = f"https://api.jcdecaux.com/vls/v1/stations?contract=dublin&apiKey={JCDECAUX_API_KEY}"
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                stations = response.json()
                now = datetime.now()

                for s in stations:
                    station_id = s['number']
                    LIVE_STATION_CACHE[station_id] = {
                        'name': s['name'],
                        'available_bikes': s['available_bikes'],
                        'available_bike_stands': s['available_bike_stands'],
                        'last_update': s['last_update'],
                    }
                    LIVE_HISTORY_BUFFER[station_id].append({
                        'timestamp': now,
                        'bikes_available': s['available_bikes'],
                        'docks_available': s['available_bike_stands'],
                    })
                    cutoff = now.timestamp() - 70 * 60
                    LIVE_HISTORY_BUFFER[station_id] = [
                        r for r in LIVE_HISTORY_BUFFER[station_id]
                        if r['timestamp'].timestamp() > cutoff
                    ]
                print(f"[{now.strftime('%H:%M:%S')}] Polled {len(stations)} stations")
            except Exception as e:
                print(f"JCDecaux poll failed (will retry): {e}")
        time_module.sleep(120)


# ---------- Live weather integration ----------

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
LIVE_WEATHER_CACHE = {}


def poll_weather_loop():
    while True:
        if OPENWEATHER_API_KEY:
            try:
                url = f"https://api.openweathermap.org/data/2.5/weather?lat=53.428&lon=-6.241&appid={OPENWEATHER_API_KEY}&units=metric"
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                w = response.json()
                LIVE_WEATHER_CACHE['rain'] = w.get('rain', {}).get('1h', 0.0)
                LIVE_WEATHER_CACHE['temp'] = w['main']['temp']
                LIVE_WEATHER_CACHE['rhum'] = w['main']['humidity']
                LIVE_WEATHER_CACHE['wdsp'] = w['wind']['speed']
                LIVE_WEATHER_CACHE['vis'] = w.get('visibility', 10000)
                print("Live weather updated")
            except Exception as e:
                print(f"Weather poll failed (will retry): {e}")
        time_module.sleep(600)


threading.Thread(target=poll_jcdecaux_loop, daemon=True).start()
threading.Thread(target=poll_weather_loop, daemon=True).start()


def compute_live_lag_features(station_id: int):
    history = LIVE_HISTORY_BUFFER.get(station_id, [])
    if not history:
        return None
    now = datetime.now().timestamp()
    result = {}
    for window_min in [15, 30, 60]:
        cutoff = now - window_min * 60
        window_data = [r for r in history if r['timestamp'].timestamp() >= cutoff]
        if window_data:
            result[f'bikes_avg_{window_min}min'] = sum(r['bikes_available'] for r in window_data) / len(window_data)
            result[f'docks_avg_{window_min}min'] = sum(r['docks_available'] for r in window_data) / len(window_data)
        else:
            result[f'bikes_avg_{window_min}min'] = None
            result[f'docks_avg_{window_min}min'] = None
    return result


def compute_live_cascade_features(station_id: int):
    neighbour_ids = NEIGHBOUR_MAP.get(station_id, {}).get('neighbour_ids', [])
    if not neighbour_ids:
        return None

    shortage_flags = []
    bikes_15, bikes_30, bikes_60 = [], [], []
    docks_15, docks_30, docks_60 = [], [], []

    for nid in neighbour_ids:
        n_live = LIVE_STATION_CACHE.get(nid)
        if n_live:
            is_shortage = int(n_live['available_bikes'] == 0 or n_live['available_bike_stands'] == 0)
            shortage_flags.append(is_shortage)

        n_lags = compute_live_lag_features(nid)
        if n_lags:
            if n_lags['bikes_avg_15min'] is not None: bikes_15.append(n_lags['bikes_avg_15min'])
            if n_lags['bikes_avg_30min'] is not None: bikes_30.append(n_lags['bikes_avg_30min'])
            if n_lags['bikes_avg_60min'] is not None: bikes_60.append(n_lags['bikes_avg_60min'])
            if n_lags['docks_avg_15min'] is not None: docks_15.append(n_lags['docks_avg_15min'])
            if n_lags['docks_avg_30min'] is not None: docks_30.append(n_lags['docks_avg_30min'])
            if n_lags['docks_avg_60min'] is not None: docks_60.append(n_lags['docks_avg_60min'])

    if not shortage_flags and not bikes_15:
        return None

    return {
        'neighbour_shortage_rate': float(np.mean(shortage_flags)) if shortage_flags else None,
        'neighbour_avg_bikes_15min': float(np.mean(bikes_15)) if bikes_15 else None,
        'neighbour_avg_bikes_30min': float(np.mean(bikes_30)) if bikes_30 else None,
        'neighbour_avg_bikes_60min': float(np.mean(bikes_60)) if bikes_60 else None,
        'neighbour_avg_docks_15min': float(np.mean(docks_15)) if docks_15 else None,
        'neighbour_avg_docks_30min': float(np.mean(docks_30)) if docks_30 else None,
        'neighbour_avg_docks_60min': float(np.mean(docks_60)) if docks_60 else None,
    }


def get_risk_level(proba):
    shortage_prob = float(proba[1] + proba[2])
    if shortage_prob >= 0.7:
        return "High", "🔴"
    elif shortage_prob >= 0.4:
        return "Medium", "🟡"
    return "Low", "🟢"


def build_lstm_sequence(station_id: int, current_row_features: dict):
    history = LIVE_HISTORY_BUFFER.get(station_id, [])

    if len(history) >= LSTM_SEQ_LEN:
        recent = history[-LSTM_SEQ_LEN:]
        sequence = []
        for r in recent:
            step_features = dict(current_row_features)
            step_features['bikes_available'] = r['bikes_available']
            step_features['docks_available'] = r['docks_available']
            sequence.append([step_features[col] for col in lstm_feature_cols])
    else:
        sequence = [[current_row_features[col] for col in lstm_feature_cols] for _ in range(LSTM_SEQ_LEN)]

    return np.array(sequence, dtype=float).reshape(LSTM_SEQ_LEN, len(lstm_feature_cols))


def get_probabilities(station_id: int, horizon: str, model: str):
    matches = data[data['station_id'] == station_id]
    if matches.empty:
        return None
    row = matches.iloc[[0]].copy()

    now = datetime.now()
    row['hour'] = now.hour
    row['day_of_week'] = now.weekday()
    row['month'] = now.month
    row['is_weekend'] = int(now.weekday() in [5, 6])
    row['is_rush_hour'] = int(now.hour in [7, 8, 9, 17, 18, 19])

    live = LIVE_STATION_CACHE.get(station_id)
    if live:
        row['bikes_available'] = live['available_bikes']
        row['docks_available'] = live['available_bike_stands']

    live_lags = compute_live_lag_features(station_id)
    if live_lags:
        for key, value in live_lags.items():
            if value is not None:
                row[key] = value

    live_cascade = compute_live_cascade_features(station_id)
    if live_cascade:
        for key, value in live_cascade.items():
            if value is not None:
                row[key] = value

    if LIVE_WEATHER_CACHE:
        for key in ['rain', 'temp', 'rhum', 'wdsp', 'vis']:
            if key in LIVE_WEATHER_CACHE:
                row[key] = LIVE_WEATHER_CACHE[key]

    if model == "xgboost":
        X = row[feature_cols]
        booster = xgb_30 if horizon == "30min" else xgb_60
        proba = booster.predict(xgb.DMatrix(X))[0]
    elif model == "logistic_regression":
        X = row[feature_cols]
        lr_model = lr_30 if horizon == "30min" else lr_60
        X_scaled = lr_scaler.transform(X)
        proba = lr_model.predict_proba(X_scaled)[0]
    elif model == "random_forest":
        X = row[feature_cols]
        rf_model = rf_30 if horizon == "30min" else rf_60
        proba = rf_model.predict_proba(X)[0]
    elif model == "lstm":
        row_dict = row.iloc[0].to_dict()
        scaler = lstm_scaler_30 if horizon == "30min" else lstm_scaler_60
        model_obj = lstm_30 if horizon == "30min" else lstm_60

        seq = build_lstm_sequence(station_id, row_dict)
        seq_scaled = scaler.transform(seq).reshape(1, LSTM_SEQ_LEN, len(lstm_feature_cols))

        proba = model_obj.predict(seq_scaled, verbose=0)[0]
    else:
        return None

    return proba


def get_prediction_class(proba, model):
    if model == "random_forest":
        return CLASS_NAMES[apply_rf_threshold(proba)]
    return CLASS_NAMES[int(np.argmax(proba))]


@app.get("/")
def root():
    return {
        "status": "API is running",
        "available_models": ["xgboost", "logistic_regression", "random_forest", "lstm"],
        "live_stations_cached": len(LIVE_STATION_CACHE),
        "live_weather_available": bool(LIVE_WEATHER_CACHE),
    }


@app.get("/predict")
def predict(station_id: int, horizon: str = "30min", model: str = "xgboost"):
    proba = get_probabilities(station_id, horizon, model)
    if proba is None:
        return {"error": "station not found or unknown model"}

    prediction = get_prediction_class(proba, model)
    risk_label, risk_emoji = get_risk_level(proba)

    return {
        "station_id": station_id,
        "horizon": horizon,
        "model": model,
        "prediction": prediction,
        "probabilities": {
            "normal": float(proba[0]),
            "bike_shortage": float(proba[1]),
            "dock_shortage": float(proba[2]),
        },
        "risk_level": risk_label,
        "risk_emoji": risk_emoji,
    }


@app.get("/predict/compare")
def compare_all(station_id: int, horizon: str = "30min"):
    results = {}
    for model in ["xgboost", "logistic_regression", "random_forest", "lstm"]:
        proba = get_probabilities(station_id, horizon, model)
        if proba is None:
            continue
        results[model] = {
            "prediction": get_prediction_class(proba, model),
            "confidence": float(np.max(proba)),
        }

    if not results:
        return {"error": "station not found"}

    return {
        "station_id": station_id,
        "horizon": horizon,
        "results": results,
        "best_model": BEST_MODEL_BY_METRIC,
        "best_model_reason": BEST_MODEL_REASON,
    }


@app.get("/cascade")
def cascade_risk(station_id: int, horizon: str = "30min", model: str = "xgboost"):
    neighbour_ids = NEIGHBOUR_MAP.get(station_id, {}).get('neighbour_ids', [])

    own_proba = get_probabilities(station_id, horizon, model)
    if own_proba is None:
        return {"error": "station not found"}
    own_prediction = get_prediction_class(own_proba, model)

    neighbour_results = []
    shortage_count = 0
    for nid in neighbour_ids:
        n_proba = get_probabilities(int(nid), horizon, model)
        if n_proba is None:
            continue
        n_pred = get_prediction_class(n_proba, model)
        if n_pred != "Normal":
            shortage_count += 1
        neighbour_results.append({"station_id": int(nid), "prediction": n_pred})

    cascade_level = "High" if shortage_count >= 2 else ("Medium" if shortage_count == 1 else "Low")

    return {
        "station_id": station_id,
        "own_prediction": own_prediction,
        "cascade_risk": cascade_level,
        "neighbours": neighbour_results,
    }


@app.get("/recommend")
def recommend_alternatives(station_id: int, horizon: str = "30min", model: str = "xgboost"):
    neighbour_ids = NEIGHBOUR_MAP.get(station_id, {}).get('neighbour_ids', [])
    distances = NEIGHBOUR_MAP.get(station_id, {}).get('distances_km', [])

    recommendations = []
    for nid, dist in zip(neighbour_ids, distances):
        n_row = data[data['station_id'] == nid]
        if n_row.empty:
            continue
        n_row = n_row.iloc[0]

        n_proba = get_probabilities(int(nid), horizon, model)
        n_pred = get_prediction_class(n_proba, model)

        live = LIVE_STATION_CACHE.get(int(nid))
        bikes = live['available_bikes'] if live else int(n_row['bikes_available'])
        docks = live['available_bike_stands'] if live else int(n_row['docks_available'])

        recommendations.append({
            "station_id": int(nid),
            "distance_km": round(float(dist), 3),
            "bikes_available": bikes,
            "docks_available": docks,
            "predicted_status": n_pred,
        })

    recommendations.sort(key=lambda r: (r["predicted_status"] != "Normal", r["distance_km"]))

    return {"station_id": station_id, "recommendations": recommendations}


@app.get("/live-snapshot")
def get_live_snapshot(station_id: int):
    live_data = LIVE_STATION_CACHE.get(station_id)
    if not live_data:
        return {"error": "no live data available for this station yet"}
    return {"station_id": station_id, "live_data": live_data}


@app.get("/live-status")
def live_status(station_id: int = None):
    if station_id:
        history = LIVE_HISTORY_BUFFER.get(station_id, [])
        return {
            "station_id": station_id,
            "readings_collected": len(history),
            "oldest_reading": history[0]['timestamp'].isoformat() if history else None,
            "newest_reading": history[-1]['timestamp'].isoformat() if history else None,
        }
    return {
        "total_stations_with_history": len(LIVE_HISTORY_BUFFER),
        "live_stations_cached": len(LIVE_STATION_CACHE),
        "live_weather_available": bool(LIVE_WEATHER_CACHE),
    }