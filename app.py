import streamlit as st
import requests
import pandas as pd

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Dublin Bikes Shortage Predictor",
    layout="centered"
)

# Hide Streamlit toolbar, Deploy button, menu, and footer
st.markdown("""
<style>
[data-testid="stToolbar"] {
    display: none;
}

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

st.title("🚲 Dublin Bikes Shortage Predictor")
st.caption(
    "Predicting bike and dock shortages 30–60 minutes in advance, with cascade detection across neighbouring stations"
)

# ==========================================================
# Models
# ==========================================================

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "lstm": "LSTM"
}

# ==========================================================
# Load Station Reference Data
# ==========================================================

@st.cache_data
def load_stations():
    ids_df = pd.read_csv(
        "../data/test_features_v2.csv",
        usecols=["station_id"]
    )

    names_df = pd.read_csv("../data/station_names.csv")

    merged = ids_df.drop_duplicates().merge(
        names_df,
        on="station_id",
        how="left"
    )

    merged["station_name"] = merged["station_name"].fillna("Unknown")

    merged["display"] = (
        merged["station_id"].astype(str)
        + " — "
        + merged["station_name"]
    )

    return merged.sort_values("station_id")


station_lookup_df = load_stations()


def get_station_name(sid):
    match = station_lookup_df[
        station_lookup_df["station_id"] == sid
    ]

    if not match.empty:
        return match.iloc[0]["station_name"]

    return "Unknown"

# ==========================================================
# Sidebar Controls
# ==========================================================

station_display = st.selectbox(
    "Select a station",
    station_lookup_df["display"]
)

station_id = int(station_display.split(" — ")[0])

horizon = st.radio(
    "Prediction Horizon",
    ["30min", "60min"],
    horizontal=True
)

view_mode = st.radio(
    "View Mode",
    [
        "Single Model",
        "Compare All Models"
    ],
    horizontal=True
)

model_choice = None

if view_mode == "Single Model":

    model_choice = st.selectbox(
        "Select Prediction Model",
        options=list(MODEL_LABELS.keys()),
        format_func=lambda x: MODEL_LABELS[x]
    )

predict_clicked = st.button(
    "Predict",
    type="primary"
)

# ==========================================================
# Prediction
# ==========================================================

if predict_clicked:

    # ------------------------------------------------------
    # Live Snapshot
    # ------------------------------------------------------

    live_response = requests.get(
        f"{API_BASE}/live-snapshot",
        params={
            "station_id": station_id
        }
    )

    live_result = live_response.json()

    col_status, col_weather = st.columns(2)

    with col_status:

        st.subheader("📍 Current Station")

        if "live_data" in live_result:

            ld = live_result["live_data"]

            st.metric(
                "Bikes Available",
                ld["available_bikes"]
            )

            st.metric(
                "Docks Available",
                ld["available_bike_stands"]
            )

            st.caption(
                f"🟢 Live — Updated {ld['last_update']}"
            )

        else:

            st.warning(
                "Live data unavailable. Using historical values."
            )

    with col_weather:

        st.subheader("🌦 Current Weather")

        root_info = requests.get(
            f"{API_BASE}/"
        ).json()

        if root_info.get("live_weather_available"):

            st.caption("🟢 Live weather active")

        else:

            st.caption(
                "⚠ Using historical weather"
            )

    st.divider()

    # ------------------------------------------------------
    # Single Model Prediction
    # ------------------------------------------------------

    if view_mode == "Single Model":

        with st.spinner("Running prediction..."):

            response = requests.get(
                f"{API_BASE}/predict",
                params={
                    "station_id": station_id,
                    "horizon": horizon,
                    "model": model_choice
                }
            )

            result = response.json()

        if "error" in result:

            st.error(result["error"])

        else:

            st.subheader(
                f"{MODEL_LABELS[model_choice]} Prediction"
            )

            st.success(result["prediction"])

            st.subheader(
                f"Risk Level: {result['risk_emoji']} {result['risk_level']}"
            )

            probs = result["probabilities"]

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Normal",
                    f"{probs['normal']*100:.2f}%"
                )

            with col2:

                st.metric(
                    "Bike Shortage",
                    f"{probs['bike_shortage']*100:.2f}%"
                )

            with col3:

                st.metric(
                    "Dock Shortage",
                    f"{probs['dock_shortage']*100:.2f}%"
                )

    # ------------------------------------------------------
    # Compare All Models
    # ------------------------------------------------------

    else:

        with st.spinner("Comparing all models..."):

            compare_response = requests.get(
                f"{API_BASE}/predict/compare",
                params={
                    "station_id": station_id,
                    "horizon": horizon
                }
            )

            compare_result = compare_response.json()

        if "results" in compare_result:

            comparison_data = []

            for model_name, model_result in compare_result["results"].items():

                comparison_data.append({

                    "Model": MODEL_LABELS.get(
                        model_name,
                        model_name
                    ),

                    "Prediction": model_result["prediction"],

                    "Confidence":
                        f"{model_result['confidence']*100:.2f}%"

                })

            st.subheader("Model Comparison")

            st.dataframe(
                pd.DataFrame(comparison_data),
                use_container_width=True,
                hide_index=True
            )

            st.success(
                f"🏆 Best Model: **{MODEL_LABELS.get(compare_result['best_model'], compare_result['best_model'])}**\n\n"
                f"{compare_result['best_model_reason']}"
            )

    st.divider()
    
        # ==========================================================
    # Cascade Risk Analysis
    # ==========================================================

    st.subheader("🔗 Cascade Risk Analysis")

    cascade_model = model_choice if model_choice else "xgboost"

    cascade_response = requests.get(
        f"{API_BASE}/cascade",
        params={
            "station_id": station_id,
            "horizon": horizon,
            "model": cascade_model
        }
    )

    cascade_result = cascade_response.json()

    if "error" not in cascade_result:

        risk_icons = {
            "High": "🔴",
            "Medium": "🟡",
            "Low": "🟢"
        }

        st.write(
            f"**Cascade Risk:** {risk_icons.get(cascade_result['cascade_risk'], '')} "
            f"{cascade_result['cascade_risk']}"
        )

        neighbour_rows = []

        for neighbour in cascade_result["neighbours"]:

            neighbour_rows.append({

                "Station ID": neighbour["station_id"],

                "Station Name": get_station_name(
                    neighbour["station_id"]
                ),

                "Predicted Status": neighbour["prediction"]

            })

        if neighbour_rows:

            st.dataframe(
                pd.DataFrame(neighbour_rows),
                use_container_width=True,
                hide_index=True
            )

        # ------------------------------------------------------
        # Map
        # ------------------------------------------------------

        try:

            coords = pd.read_csv(
                "../data/test_features_v2.csv",
                usecols=[
                    "station_id",
                    "latitude",
                    "longitude"
                ]
            )

            coords = (
                coords
                .drop_duplicates("station_id")
                .set_index("station_id")
            )

            map_points = []

            if station_id in coords.index:

                current = coords.loc[station_id]

                map_points.append({
                    "lat": current["latitude"],
                    "lon": current["longitude"]
                })

            for neighbour in cascade_result["neighbours"]:

                sid = neighbour["station_id"]

                if sid in coords.index:

                    row = coords.loc[sid]

                    map_points.append({

                        "lat": row["latitude"],
                        "lon": row["longitude"]

                    })

            if map_points:

                st.map(
                    pd.DataFrame(map_points)
                )

        except Exception:

            pass

    else:

        st.warning(
            "Cascade analysis unavailable."
        )

    st.divider()

    # ==========================================================
    # Alternative Station Recommendations
    # ==========================================================

    st.subheader(
        "🚲 Alternative Station Recommendations"
    )

    recommendation_response = requests.get(

        f"{API_BASE}/recommend",

        params={
            "station_id": station_id,
            "horizon": horizon,
            "model": cascade_model
        }

    )

    recommendation_result = recommendation_response.json()

    if (
        "recommendations" in recommendation_result
        and recommendation_result["recommendations"]
    ):

        recommendation_rows = []

        for station in recommendation_result["recommendations"]:

            recommendation_rows.append({

                "Station ID":
                    station["station_id"],

                "Station Name":
                    get_station_name(
                        station["station_id"]
                    ),

                "Distance (km)":
                    station["distance_km"],

                "Available Bikes":
                    station["bikes_available"],

                "Available Docks":
                    station["docks_available"],

                "Predicted Status":
                    station["predicted_status"]

            })

        st.dataframe(
            pd.DataFrame(recommendation_rows),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No nearby recommendations available."
        )

    st.divider()

    # ==========================================================
    # Footer
    # ==========================================================

    st.caption(
        "Dublin Bikes Shortage Prediction System\n"
        "Models: XGBoost | Logistic Regression | Random Forest | LSTM"
    )