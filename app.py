import streamlit as st
import requests
import pandas as pd
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 5

st.set_page_config(page_title="Dublin Bikes — Shortage Board", page_icon="🚲", layout="centered")

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "lstm": "LSTM",
}
RISK_COLORS = {"High": "#D9534F", "Medium": "#E8A33D", "Low": "#4CAF7D"}


def api_get(endpoint: str, params: dict = None):
    """Shared helper for all backend calls: timeout + error handling,
    so a backend outage shows a clear message instead of freezing or
    crashing the UI."""
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


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.board-header { background: #16273B; border-left: 5px solid #0E7C7C; padding: 18px 24px; border-radius: 4px; margin-bottom: 24px; }
.board-header h1 { font-family: 'Oswald', sans-serif; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; font-size: 26px; color: #EDEEF0; margin: 0; }
.board-header p { color: #8A97A8; font-size: 13px; margin: 4px 0 0 0; font-family: 'IBM Plex Mono', monospace; }
.live-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #4CAF7D; margin-right: 6px; animation: pulse 2s infinite; }
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
.status-card { background: #16273B; border: 1px solid #223349; border-radius: 4px; padding: 16px 20px; margin-bottom: 12px; }
.status-card .label { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8A97A8; text-transform: uppercase; letter-spacing: 1px; }
.status-card .value { font-family: 'IBM Plex Mono', monospace; font-size: 28px; color: #EDEEF0; font-weight: 600; }
.announcement { border-radius: 4px; padding: 22px 26px; margin: 16px 0; border-left: 6px solid; }
.announcement .eyebrow { font-family: 'IBM Plex Mono', monospace; font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.85; }
.announcement .headline { font-family: 'Oswald', sans-serif; font-size: 30px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin: 6px 0; }
.section-label { font-family: 'Oswald', sans-serif; text-transform: uppercase; letter-spacing: 1px; font-size: 15px; color: #0E7C7C; border-bottom: 1px solid #223349; padding-bottom: 6px; margin: 28px 0 12px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="board-header">
    <h1>🚲 Dublin Bikes — Shortage Board</h1>
    <p><span class="live-dot"></span>MULTI-HORIZON CASCADE-AWARE PREDICTION · 30&nbsp;/&nbsp;60 MIN</p>
</div>
""", unsafe_allow_html=True)


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


col_a, col_b = st.columns([2, 1])
with col_a:
    station_display = st.selectbox("STATION", station_lookup_df['display'])
    station_id = int(station_display.split("#")[-1])
with col_b:
    horizon = st.radio("HORIZON", ["30min", "60min"], horizontal=True)

view_mode = st.radio("VIEW", ["Single Model", "Compare All Models"], horizontal=True)

model_choice = None
if view_mode == "Single Model":
    model_choice = st.selectbox("MODEL", list(MODEL_LABELS.keys()), format_func=lambda x: MODEL_LABELS[x])

predict_clicked = st.button("▶  RUN PREDICTION", type="primary", use_container_width=True)

if predict_clicked:
    with st.spinner("Fetching live station status..."):
        live_result, live_err = api_get("/live-snapshot", {"station_id": station_id})

    st.markdown('<div class="section-label">CURRENT STATUS</div>', unsafe_allow_html=True)
    if live_err:
        st.warning(f"Live data unavailable ({live_err}) — showing historical baseline")
    elif live_result and "live_data" in live_result:
        ld = live_result["live_data"]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="status-card"><div class="label">Bikes Available</div><div class="value">{ld["available_bikes"]}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="status-card"><div class="label">Docks Available</div><div class="value">{ld["available_bike_stands"]}</div></div>', unsafe_allow_html=True)
        st.caption(f"🟢 Live · updated {ld['last_update']}")
    else:
        st.warning("Live data not yet available — showing historical baseline")

    if view_mode == "Single Model":
        with st.spinner("Predicting..."):
            result, err = api_get("/predict", {"station_id": station_id, "horizon": horizon, "model": model_choice})

        if err:
            st.error(err)
        elif "error" in result:
            st.error(result["error"])
        else:
            risk = result['risk_level']
            color = RISK_COLORS.get(risk, "#4CAF7D")
            probs = result['probabilities']
            st.markdown(f"""
            <div class="announcement" style="background:{color}22; border-color:{color};">
                <div class="eyebrow" style="color:{color};">{MODEL_LABELS[model_choice]} · {horizon.upper()} FORECAST</div>
                <div class="headline" style="color:{color};">{result['prediction']}</div>
                <div style="font-family:'IBM Plex Mono',monospace; color:#8A97A8; font-size:13px;">
                    NORMAL {probs['normal']*100:.1f}%  ·  BIKE {probs['bike_shortage']*100:.1f}%  ·  DOCK {probs['dock_shortage']*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        with st.spinner("Comparing all models..."):
            compare_result, err = api_get("/predict/compare", {"station_id": station_id, "horizon": horizon})

        if err:
            st.error(err)
        elif "results" in compare_result:
            st.markdown('<div class="section-label">MODEL COMPARISON</div>', unsafe_allow_html=True)
            comparison_data = [{
                "Model": MODEL_LABELS.get(m, m),
                "Prediction": r["prediction"],
                "Confidence": f"{r['confidence']*100:.1f}%"
            } for m, r in compare_result["results"].items()]
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

            st.markdown(f"""
            <div class="status-card">
                <div class="label">🏆 Best Overall (Test-Set Evaluation)</div>
                <div style="color:#EDEEF0; font-family:'Oswald',sans-serif; font-size:18px; margin-top:4px;">{MODEL_LABELS.get(compare_result['best_model_overall'])}</div>
                <div style="color:#8A97A8; font-size:12px; margin-top:4px;">{compare_result['best_model_overall_reason']}</div>
            </div>
            <div class="status-card">
                <div class="label">⚡ Most Confident on This Prediction</div>
                <div style="color:#EDEEF0; font-family:'Oswald',sans-serif; font-size:18px; margin-top:4px;">{MODEL_LABELS.get(compare_result['most_confident_this_prediction'])} — {compare_result['most_confident_score']*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        model_choice = "xgboost"

    st.markdown('<div class="section-label">CASCADE RISK — NEIGHBOURING STATIONS</div>', unsafe_allow_html=True)
    with st.spinner("Checking neighbouring stations..."):
        cascade_result, cascade_err = api_get("/cascade", {"station_id": station_id, "horizon": horizon, "model": model_choice})

    if cascade_err:
        st.error(cascade_err)
    elif "error" not in cascade_result:
        risk = cascade_result['cascade_risk']
        color = RISK_COLORS.get(risk, "#4CAF7D")
        st.markdown(f'<div class="status-card"><div class="label">Cascade Risk</div><div class="value" style="color:{color};">{risk}</div></div>', unsafe_allow_html=True)

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

    st.markdown('<div class="section-label">ALTERNATIVE STATIONS</div>', unsafe_allow_html=True)
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