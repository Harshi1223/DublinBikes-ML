import streamlit as st
import requests
import pandas as pd
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 5

st.set_page_config(page_title="Dublin Bikes — Shortage Predictor", layout="centered")

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "lstm": "LSTM",
}


def api_get(endpoint: str, params: dict = None):
    """Shared helper for backend calls: timeout + error handling."""
    try:
        response = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.Timeout:
        return None, "Request timed out — the prediction service may be busy or unreachable."
    except requests.exceptions.ConnectionError:
        return None, f"Could not reach the prediction service at {API_BASE}."
    except requests.exceptions.HTTPError as e:
        return None, f"Prediction service returned an error: {e}"
    except requests.exceptions.RequestException as e:
        return None, f"Unexpected error contacting prediction service: {e}"


st.title("Dublin Bikes — Shortage Predictor")


@st.cache_data
def load_stations():
    ids_df = pd.read_csv(DATA_DIR / "test_features_v2.csv", usecols=['station_id'])
    names_df = pd.read_csv(DATA_DIR / "station_names.csv")
    merged = ids_df.drop_duplicates().merge(names_df, on='station_id', how='left')
    merged['station_name'] = merged['station_name'].fillna("Unknown")
    merged['display'] = merged['station_name'] + "  ·  #" + merged['station_id'].astype(str)
    return merged.sort_values('station_name')


@st.cache_data
def load_station_coords():
    df = pd.read_csv(DATA_DIR / "test_features_v2.csv", usecols=['station_id', 'latitude', 'longitude'])
    return df.drop_duplicates('station_id').set_index('station_id')


station_lookup_df = load_stations()
station_coords = load_station_coords()


def get_station_name(sid):
    match = station_lookup_df[station_lookup_df['station_id'] == sid]
    return match.iloc[0]['station_name'] if not match.empty else "Unknown"


# ---------- Controls ----------
col_a, col_b = st.columns([2, 1])
with col_a:
    station_display = st.selectbox("Station", station_lookup_df['display'])
    station_id = int(station_display.split("#")[-1])
with col_b:
    horizon = st.radio("Horizon", ["30min", "60min"], horizontal=True)

view_mode = st.radio("View", ["Single Model", "Compare All Models"], horizontal=True)

model_choice = None
if view_mode == "Single Model":
    model_choice = st.selectbox("Model", list(MODEL_LABELS.keys()), format_func=lambda x: MODEL_LABELS[x])

predict_clicked = st.button("Predict", type="primary", use_container_width=True)

if predict_clicked:

    # ---------- Current Status ----------
    with st.spinner("Fetching live station status..."):
        live_result, live_err = api_get("/live-snapshot", {"station_id": station_id})

    st.subheader("Current Status")
    if live_err:
        st.warning(f"Live data unavailable ({live_err}) — showing historical baseline")
    elif live_result and "live_data" in live_result:
        ld = live_result["live_data"]
        bikes = ld["available_bikes"]
        docks = ld["available_bike_stands"]
        capacity = bikes + docks
        occupancy_pct = round((bikes / capacity) * 100) if capacity > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Bikes Available", bikes)
        col2.metric("Docks Available", docks)
        col3.metric("Occupancy", f"{occupancy_pct}%")
        st.caption(f"🟢 Live · updated {ld['last_update']}")
    else:
        st.warning("Live data not yet available — showing historical baseline")

    # ---------- Weather ----------
    root_info, _ = api_get("/")
    st.subheader("Weather")
    if root_info and root_info.get("live_weather_available"):
        st.success("🟢 Live weather feed active — temperature, rainfall, humidity and wind speed are blended into this prediction.")
    else:
        st.info("Using the most recent historical weather values as a fallback.")

    # ---------- Prediction ----------
    if view_mode == "Single Model":
        with st.spinner("Predicting..."):
            result, err = api_get("/predict", {"station_id": station_id, "horizon": horizon, "model": model_choice})

        if err:
            st.error(err)
        elif "error" in result:
            st.error(result["error"])
        else:
            risk = result['risk_level']
            emoji = result['risk_emoji']
            probs = result['probabilities']

            if risk == "High":
                st.error(f"{emoji} **{result['prediction']}** — {MODEL_LABELS[model_choice]}, {horizon} forecast")
            elif risk == "Medium":
                st.warning(f"{emoji} **{result['prediction']}** — {MODEL_LABELS[model_choice]}, {horizon} forecast")
            else:
                st.success(f"{emoji} **{result['prediction']}** — {MODEL_LABELS[model_choice]}, {horizon} forecast")

            st.subheader("Probability Breakdown")
            st.write("Normal")
            st.progress(int(probs['normal'] * 100), text=f"{probs['normal']*100:.1f}%")
            st.write("Bike Shortage")
            st.progress(int(probs['bike_shortage'] * 100), text=f"{probs['bike_shortage']*100:.1f}%")
            st.write("Dock Shortage")
            st.progress(int(probs['dock_shortage'] * 100), text=f"{probs['dock_shortage']*100:.1f}%")

    else:
        with st.spinner("Comparing all models..."):
            compare_result, err = api_get("/predict/compare", {"station_id": station_id, "horizon": horizon})

        if err:
            st.error(err)
        elif "results" in compare_result:
            st.subheader("Model Comparison")
            comparison_data = [{
                "Model": MODEL_LABELS.get(m, m),
                "Prediction": r["prediction"],
                "Confidence": f"{r['confidence']*100:.1f}%"
            } for m, r in compare_result["results"].items()]
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("🏆 Best Overall", MODEL_LABELS.get(compare_result['best_model_overall']))
                st.caption(compare_result['best_model_overall_reason'])
            with col2:
                st.metric("⚡ Most Confident Now", MODEL_LABELS.get(compare_result['most_confident_this_prediction']),
                          f"{compare_result['most_confident_score']*100:.1f}%")
        model_choice = "xgboost"

    # ---------- Cascade ----------
    st.subheader("Cascade Risk — Neighbouring Stations")
    with st.spinner("Checking neighbouring stations..."):
        cascade_result, cascade_err = api_get("/cascade", {"station_id": station_id, "horizon": horizon, "model": model_choice})

    if cascade_err:
        st.error(cascade_err)
    elif "error" not in cascade_result:
        risk = cascade_result['cascade_risk']
        if risk == "High":
            st.error(f"Cascade Risk: {risk}")
        elif risk == "Medium":
            st.warning(f"Cascade Risk: {risk}")
        else:
            st.success(f"Cascade Risk: {risk}")

        neighbour_rows = [{"Station": get_station_name(n['station_id']), "Status": n['prediction']} for n in cascade_result['neighbours']]
        if neighbour_rows:
            st.dataframe(pd.DataFrame(neighbour_rows), use_container_width=True, hide_index=True)

        try:
            map_points = [{"lat": station_coords.loc[station_id]['latitude'], "lon": station_coords.loc[station_id]['longitude']}]
            for n in cascade_result['neighbours']:
                if n['station_id'] in station_coords.index:
                    c = station_coords.loc[n['station_id']]
                    map_points.append({"lat": c['latitude'], "lon": c['longitude']})
            st.map(pd.DataFrame(map_points))
        except Exception as e:
            st.warning(f"Could not render map: {e}")

    # ---------- Recommendations ----------
    st.subheader("Alternative Stations")
    with st.spinner("Generating recommendations..."):
        rec_result, rec_err = api_get("/recommend", {"station_id": station_id, "horizon": horizon, "model": model_choice})

    if rec_err:
        st.error(rec_err)
    elif "recommendations" in rec_result and rec_result["recommendations"]:
        rec_rows = [{
            "Station": get_station_name(r['station_id']),
            "Distance (km)": r['distance_km'],
            "Bikes": r['bikes_available'],
            "Docks": r['docks_available'],
            "Status": r['predicted_status'],
        } for r in rec_result["recommendations"]]
        st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No nearby station data available")