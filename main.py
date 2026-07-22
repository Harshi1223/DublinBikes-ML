# main.py — Dublin Bikes Shortage Prediction API
# XGBoost + Logistic Regression + Random Forest + LSTM

from fastapi import FastAPI
from pathlib import Path
import xgboost as xgb
import joblib
import pandas as pd
import numpy as np
import requests
import os
import pickle
import json
import logging
import threading
import time as time_module
from collections import defaultdict
from datetime import datetime
from tensorflow import keras

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

app = FastAPI(title="Dublin Bikes Shortage Prediction API")

# ---------- Constants ----------

HORIZONS = ["30min", "60min"]
MODELS = ["xgboost", "logistic_regression", "random_forest", "lstm"]
CLASS_NAMES = ["Normal", "Bike Shortage", "Dock Shortage"]

RF_THRESHOLD = 0.7
LSTM_SEQ_LEN = 6
LIVE_HISTORY_RETENTION_MIN = 70
JCDECAUX_POLL_INTERVAL_SEC = 120
WEATHER_POLL_INTERVAL_SEC = 600
WEATHER_RETRY_INTERVAL_SEC = 60          # shorter wait before retrying after a failed poll
WEATHER_CACHE_MAX_AGE_SEC = 1800         # weather older than this is treated as stale, not live
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.4

BEST_MODEL_OVERALL = "xgboost"
BEST_MODEL_OVERALL_REASON = "Highest Macro F1 on held-out test set (0.8013 vs 0.7770 LR, 0.784 RF)"

# ---------- Load models ----------

xgb_30 = xgb.Booster()
xgb_30.load_model(str(MODELS_DIR / "candidate_booster_30min.json"))
xgb_60 = xgb.Booster()
xgb_60.load_model(str(MODELS_DIR / "booster_60min.json"))

lr_30 = joblib.load(MODELS_DIR / "lr_best_30min.pkl")
lr_60 = joblib.load(MODELS_DIR / "lr_best_60min.pkl")
lr_scaler = joblib.load(MODELS_DIR / "lr_scaler.pkl")

rf_30 = joblib.load(MODELS_DIR / "random_forest_target_30min_final.pkl")
rf_60 = joblib.load(MODELS_DIR / "random_forest_target_60min_final.pkl")

lstm_30 = keras.models.load_model(MODELS_DIR / "lstm_target_30min_FINAL.keras")
lstm_60 = keras.models.load_model(MODELS_DIR / "lstm_target_60min_FINAL.keras")
lstm_scaler_30 = joblib.load(MODELS_DIR / "lstm_30min_scaler_FINAL.pkl")
lstm_scaler_60 = joblib.load(MODELS_DIR / "lstm_60min_scaler_FINAL.pkl")

with open(MODELS_DIR / "lstm_30min_feature_columns.json") as f:
    lstm_feature_cols = json.load(f)

data = pd.read_csv(DATA_DIR / "test_features_v2.csv")
feature_cols = [c for c in data.columns if c not in ['timestamp', 'target_now', 'target_30min', 'target_60min']]

# indexed once by station_id for fast repeated lookups
data_by_station = {sid: df.iloc[[0]] for sid, df in data.groupby('station_id')}

with open(DATA_DIR / "neighbour_map.pkl", "rb") as f:
    NEIGHBOUR_MAP = pickle.load(f)

logger.info(f"Loaded {len(data_by_station)} stations, all four models.")


def apply_rf_threshold(proba_row, threshold=RF_THRESHOLD):
    """Applies Random Forest's tuned per-class threshold (0.7) instead of
    plain argmax, since RF's default decision boundary over-predicts
    Normal for this dataset's class imbalance."""
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
    """Background loop: fetches current bike/dock availability for every
    Dublin Bikes station every JCDECAUX_POLL_INTERVAL_SEC, and appends
    each reading to a rolling per-station history buffer used to compute
    live lag and cascade features."""
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
                    cutoff = now.timestamp() - LIVE_HISTORY_RETENTION_MIN * 60
                    LIVE_HISTORY_BUFFER[station_id] = [
                        r for r in LIVE_HISTORY_BUFFER[station_id]
                        if r['timestamp'].timestamp() > cutoff
                    ]
                logger.info(f"JCDecaux poll OK — {len(stations)} stations updated")
            except requests.exceptions.RequestException as e:
                logger.warning(f"JCDecaux poll failed, will retry in {JCDECAUX_POLL_INTERVAL_SEC}s: {e}")
        else:
            logger.info("JCDECAUX_API_KEY not set — skipping poll")
        time_module.sleep(JCDECAUX_POLL_INTERVAL_SEC)


# ---------- Live weather integration ----------

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
LIVE_WEATHER_CACHE = {}


def poll_weather_loop():
    """Background loop: fetches current weather for Dublin Airport every
    WEATHER_POLL_INTERVAL_SEC. On failure, retries sooner
    (WEATHER_RETRY_INTERVAL_SEC) rather than waiting the full interval,
    so a transient outage doesn't leave the cache stale for 10 minutes.
    Stores extra descriptive fields (weather_main, weather_description,
    clouds) for display purposes only — these are not model features."""
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

                # display-only extras, not used as model features
                weather_list = w.get('weather', [{}])
                LIVE_WEATHER_CACHE['weather_main'] = weather_list[0].get('main')
                LIVE_WEATHER_CACHE['weather_description'] = weather_list[0].get('description')
                LIVE_WEATHER_CACHE['clouds'] = w.get('clouds', {}).get('all')

                LIVE_WEATHER_CACHE['last_updated'] = datetime.now()

                logger.info(f"Weather poll OK — {LIVE_WEATHER_CACHE['temp']}°C, {LIVE_WEATHER_CACHE['weather_main']}")
                time_module.sleep(WEATHER_POLL_INTERVAL_SEC)
            except requests.exceptions.RequestException as e:
                logger.warning(f"Weather poll failed, retrying in {WEATHER_RETRY_INTERVAL_SEC}s: {e}")
                time_module.sleep(WEATHER_RETRY_INTERVAL_SEC)
        else:
            logger.info("OPENWEATHER_API_KEY not set — skipping poll")
            time_module.sleep(WEATHER_POLL_INTERVAL_SEC)


def is_weather_fresh():
    """Returns True only if weather data exists AND is recent enough to
    trust as genuinely live, rather than a stale cache from a poll that
    stopped succeeding."""
    if 'last_updated' not in LIVE_WEATHER_CACHE:
        return False
    age_sec = (datetime.now() - LIVE_WEATHER_CACHE['last_updated']).total_seconds()
    return age_sec <= WEATHER_CACHE_MAX_AGE_SEC


threading.Thread(target=poll_jcdecaux_loop, daemon=True).start()
threading.Thread(target=poll_weather_loop, daemon=True).start()


def compute_live_lag_features(station_id: int):
    """Computes real 15/30/60-min rolling averages from the live history
    buffer for one station. Returns None entirely if no history exists
    yet; returns None per-window if that specific window has no readings."""
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
    """Computes live cascade signal (neighbour shortage rate and lag
    averages) by looking up a station's 3 nearest neighbours in the
    live history buffer — the same buffer JCDecaux polling fills for
    every station, not just the one being queried."""
    neighbour_ids = NEIGHBOUR_MAP.get(station_id, {}).get('neighbour_ids', [])
    if not neighbour_ids:
        return None

    shortage_flags, bikes_15, bikes_30, bikes_60 = [], [], [], []
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
    """Converts a probability vector into a coarse High/Medium/Low risk
    label, based on the combined bike+dock shortage probability."""
    shortage_prob = float(proba[1] + proba[2])
    if shortage_prob >= HIGH_RISK_THRESHOLD:
        return "High", "🔴"
    elif shortage_prob >= MEDIUM_RISK_THRESHOLD:
        return "Medium", "🟡"
    return "Low", "🟢"


def build_lstm_sequence(station_id: int, current_row_features: dict):
    """Builds a (SEQ_LEN, n_features) sequence for LSTM input, using the
    live history buffer's most recent readings where at least SEQ_LEN
    exist; otherwise repeats the current row as a fallback sequence."""
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
    """
    Computes class probabilities for a given station, horizon, and model,
    blending live data (bikes/docks, lag features, cascade signal, weather,
    time features) with historical fallback values where live data isn't
    yet available or has gone stale.
    """
    if station_id not in data_by_station:
        return None
    row = data_by_station[station_id].copy()

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

    if is_weather_fresh():
        for key in ['rain', 'temp', 'rhum', 'wdsp', 'vis']:
            if key in LIVE_WEATHER_CACHE:
                row[key] = LIVE_WEATHER_CACHE[key]

    try:
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
    except Exception as e:
        logger.error(f"Prediction failed for station={station_id}, horizon={horizon}, model={model}: {e}")
        return None

    return proba


def get_prediction_class(proba, model):
    """Converts a probability vector into a class label, applying RF's
    tuned threshold specifically; all other models use plain argmax
    since their thresholds are already baked into training/tuning."""
    if model == "random_forest":
        return CLASS_NAMES[apply_rf_threshold(proba)]
    return CLASS_NAMES[int(np.argmax(proba))]


# ---------- Endpoints ----------

@app.get("/")
def root():
    """Health check and API summary."""
    return {
        "status": "API is running",
        "available_models": MODELS,
        "live_stations_cached": len(LIVE_STATION_CACHE),
        "live_weather_available": is_weather_fresh(),
    }


@app.get("/predict")
def predict(station_id: int, horizon: str = "30min", model: str = "xgboost"):
    """Predict shortage class and probabilities for one station, horizon, and model."""
    if horizon not in HORIZONS or model not in MODELS:
        return {"error": "invalid horizon or model"}

    proba = get_probabilities(station_id, horizon, model)
    if proba is None:
        return {"error": "station not found or prediction failed"}

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
    """Compare predictions across all models. Separates the statistically
    best model overall (fixed, from test-set evaluation) from the model
    most confident on this specific live prediction (genuinely dynamic)."""
    results = {}
    for model in MODELS:
        proba = get_probabilities(station_id, horizon, model)
        if proba is None:
            continue
        results[model] = {
            "prediction": get_prediction_class(proba, model),
            "confidence": float(np.max(proba)),
        }

    if not results:
        return {"error": "station not found"}

    most_confident_model = max(results, key=lambda m: results[m]["confidence"])

    return {
        "station_id": station_id,
        "horizon": horizon,
        "results": results,
        "best_model_overall": BEST_MODEL_OVERALL,
        "best_model_overall_reason": BEST_MODEL_OVERALL_REASON,
        "most_confident_this_prediction": most_confident_model,
        "most_confident_score": results[most_confident_model]["confidence"],
    }


@app.get("/cascade")
def cascade_risk(station_id: int, horizon: str = "30min", model: str = "xgboost"):
    """Cascade risk from a station's 3 nearest neighbours."""
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
    """Rank nearby stations as alternatives, worst-status-last, closest-first."""
    neighbour_ids = NEIGHBOUR_MAP.get(station_id, {}).get('neighbour_ids', [])
    distances = NEIGHBOUR_MAP.get(station_id, {}).get('distances_km', [])

    recommendations = []
    for nid, dist in zip(neighbour_ids, distances):
        if nid not in data_by_station:
            continue
        n_row = data_by_station[nid].iloc[0]

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
    """Returns the latest raw JCDecaux reading for one station."""
    live_data = LIVE_STATION_CACHE.get(station_id)
    if not live_data:
        return {"error": "no live data available for this station yet"}
    return {"station_id": station_id, "live_data": live_data}


@app.get("/live-weather")
def get_live_weather():
    """Returns the current live weather values being blended into
    predictions. Response shape unchanged from before — rain_mm,
    temperature_c, humidity_pct, wind_speed, visibility — with new
    optional fields (weather_main, weather_description, clouds,
    last_updated) appended, so existing frontend code keeps working."""
    if not is_weather_fresh():
        return {"error": "live weather not yet available"}
    return {
        "rain_mm": LIVE_WEATHER_CACHE.get('rain'),
        "temperature_c": LIVE_WEATHER_CACHE.get('temp'),
        "humidity_pct": LIVE_WEATHER_CACHE.get('rhum'),
        "wind_speed": LIVE_WEATHER_CACHE.get('wdsp'),
        "visibility": LIVE_WEATHER_CACHE.get('vis'),
        "weather_main": LIVE_WEATHER_CACHE.get('weather_main'),
        "weather_description": LIVE_WEATHER_CACHE.get('weather_description'),
        "clouds_pct": LIVE_WEATHER_CACHE.get('clouds'),
        "last_updated": LIVE_WEATHER_CACHE.get('last_updated').isoformat() if LIVE_WEATHER_CACHE.get('last_updated') else None,
    }


@app.get("/live-status")
def live_status(station_id: int = None):
    """Diagnostic endpoint: how much live history has accumulated,
    for a specific station or overall."""
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
        "live_weather_available": is_weather_fresh(),
    }