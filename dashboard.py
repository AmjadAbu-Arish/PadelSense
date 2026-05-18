import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="PadelSense Dashboard", layout="wide")

st.title("🎾 PadelSense Analytics Dashboard")

# Paths
CSV_PATH = "outputs/match_summary.csv"
VIDEO_PATH = "outputs/output_video.mp4"
HEATMAP_PATH = "outputs/heatmap.png"

# Load data
@st.cache_data
def load_data():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        return df
    return None

df = load_data()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Match Video")
    if os.path.exists(VIDEO_PATH):
        # Streamlit supports playing standard mp4/h264 natively
        st.video(VIDEO_PATH)
    else:
        st.info("No output video found. Please run the main script first.")

    st.subheader("Ball Speed Analysis")
    if df is not None and "Ball_Speed_kmh" in df.columns:
        # Convert to numeric, ignoring errors to keep valid speeds
        df["Ball_Speed_kmh"] = pd.to_numeric(df["Ball_Speed_kmh"], errors="coerce")
        # Filter out rows with speed = 0.0 or NaNs for a cleaner plot (optional)
        df_speed = df[(df["Ball_Speed_kmh"] > 0) & (df["Ball_Speed_kmh"].notna())]
        st.line_chart(df_speed, x="Frame_Index", y="Ball_Speed_kmh", use_container_width=True)

        if not df_speed.empty:
            st.metric(label="Max Speed", value=f"{df_speed['Ball_Speed_kmh'].max():.1f} km/h")
            st.metric(label="Average Speed", value=f"{df_speed['Ball_Speed_kmh'].mean():.1f} km/h")
    else:
        st.info("Speed data not available in CSV.")

with col2:
    st.subheader("Ball Position Heatmap")
    if os.path.exists(HEATMAP_PATH):
        st.image(HEATMAP_PATH, caption="2D Density Map of Ball Positions", use_column_width=True)
    else:
        st.info("Heatmap image not found. Please run the main script first.")

    st.subheader("Event Log")
    if df is not None and "Event_Type" in df.columns and "Decision" in df.columns:
        # Filter frames where an event happened
        df_events = df[(df["Event_Type"] != "none") & (df["Event_Type"].notna())].copy()

        # We can format a nice timestamp from Frame_Index assuming 30 fps
        fps = 30.0 # Approximation for display if actual fps is unknown
        df_events["Timestamp"] = pd.to_datetime((df_events["Frame_Index"] / fps).astype(int), unit="s").dt.strftime("%M:%S")

        # Display an event log e.g. "14:02 - Valid Serve", "14:05 - Ball OUT - Hit Glass First"
        # Since we just have 'Decision' and 'Event_Type' we can combine them
        df_events["Log"] = df_events["Timestamp"] + " - Event: " + df_events["Event_Type"] + " - Decision: " + df_events["Decision"].fillna("")

        if not df_events.empty:
            st.dataframe(df_events[["Frame_Index", "Log"]], use_container_width=True, hide_index=True)
        else:
            st.write("No specific referee events (Out, Net, etc.) recorded.")
    else:
        st.info("Event data not available in CSV.")

st.markdown("---")
st.markdown("*PadelSense Professional - AI Referee & Analytics Platform*")
