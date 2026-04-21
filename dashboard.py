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
    st.subheader("Match Video with Overlays (YOLO/TrackNet & Mini-Court)")
    if os.path.exists(VIDEO_PATH):
        # Streamlit supports playing standard mp4/h264 natively
        st.video(VIDEO_PATH)
    else:
        st.info("No output video found. Please run the main script first.")

    st.subheader("Ball Speed Analysis")
    if df is not None and "Ball_Speed_kmh" in df.columns:
        df["Ball_Speed_kmh"] = pd.to_numeric(df["Ball_Speed_kmh"], errors="coerce")
        df_speed = df[df["Ball_Speed_kmh"] > 0]
        if not df_speed.empty:
            st.line_chart(df_speed, x="Frame_Index", y="Ball_Speed_kmh", use_container_width=True)
            st.metric(label="Max Speed", value=f"{df_speed['Ball_Speed_kmh'].max():.1f} km/h")
            st.metric(label="Average Speed", value=f"{df_speed['Ball_Speed_kmh'].mean():.1f} km/h")
        else:
            st.info("No valid speed data found.")
    else:
        st.info("Speed data not available in CSV.")

with col2:
    st.subheader("Ball Position Heatmap")
    if os.path.exists(HEATMAP_PATH):
        st.image(HEATMAP_PATH, caption="2D Density Map of Ball Positions", use_column_width=True)
    else:
        st.info("Heatmap image not found. Please run the main script first.")

    st.subheader("Referee Event Log")
    if df is not None and "Event_Type" in df.columns and "Decision" in df.columns:
        # Filter frames where a referee decision was made
        # Also simulate a timestamp or use frame index
        df_events = df.dropna(subset=['Decision'])
        df_events = df_events[df_events["Decision"] != ""]

        if not df_events.empty:
            log_data = []
            FPS = 30 # Approximation if we don't have actual fps in CSV
            for _, row in df_events.iterrows():
                frame_idx = row["Frame_Index"]
                seconds = int(frame_idx) // FPS
                mins, secs = divmod(seconds, 60)
                timestamp = f"{mins:02d}:{secs:02d}"

                decision = row["Decision"]
                event_type = row["Event_Type"]

                # Format logic for event log
                if decision == "IN":
                    desc = "Valid Serve / Ball IN"
                elif decision == "OUT":
                    desc = "Ball OUT"
                elif decision == "NET":
                    desc = "Ball hit NET"
                else:
                    desc = f"Decision: {decision}"

                log_data.append({"Time": timestamp, "Frame": frame_idx, "Event": event_type, "Log": f"{timestamp} - {desc}"})

            df_log = pd.DataFrame(log_data)
            for log_str in df_log["Log"].tolist():
                st.markdown(f"**{log_str}**")
        else:
            st.write("No specific referee events (Out, Net, etc.) recorded.")
    else:
        st.info("Event data not available in CSV.")

st.markdown("---")
st.markdown("*PadelSense Professional - AI Referee & Analytics Platform*")
