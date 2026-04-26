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
        # Convert to numeric if it's not
        df["Ball_Speed_kmh"] = pd.to_numeric(df["Ball_Speed_kmh"], errors="coerce")
        # Filter out rows with speed = 0.0 or NaN for a cleaner plot (optional)
        df_speed = df[df["Ball_Speed_kmh"] > 0]
        st.line_chart(df_speed, x="Frame_Index", y="Ball_Speed_kmh", use_container_width=True)

        st.metric(label="Max Speed", value=f"{df_speed['Ball_Speed_kmh'].max():.1f} km/h" if not df_speed.empty else "N/A")
        st.metric(label="Average Speed", value=f"{df_speed['Ball_Speed_kmh'].mean():.1f} km/h" if not df_speed.empty else "N/A")
    else:
        st.info("Speed data not available in CSV.")

with col2:
    st.subheader("Ball Position Heatmap")
    if os.path.exists(HEATMAP_PATH):
        st.image(HEATMAP_PATH, caption="2D Density Map of Ball Positions", use_column_width=True)
    else:
        st.info("Heatmap image not found. Please run the main script first.")

    st.subheader("Detected Events")
    if df is not None and "Event_Type" in df.columns:
        # Filter frames where an event happened
        df_events = df[df["Event_Type"] != "none"][["Frame_Index", "Event_Type", "Decision"]]
        if not df_events.empty:
            st.dataframe(df_events, use_container_width=True)
        else:
            st.write("No specific referee events (Out, Net, etc.) recorded.")
    else:
        st.info("Event data not available in CSV.")

st.markdown("---")
st.markdown("*PadelSense Professional - AI Referee & Analytics Platform*")
