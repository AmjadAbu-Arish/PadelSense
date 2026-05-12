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
        # Filter out rows with speed NaN
        df_speed = df[df["Ball_Speed_kmh"].notna()].copy()
        # Ensure it's float type
        df_speed["Ball_Speed_kmh"] = pd.to_numeric(df_speed["Ball_Speed_kmh"], errors='coerce')
        df_speed = df_speed[df_speed["Ball_Speed_kmh"] > 0]

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
    if df is not None and "Decision" in df.columns and "Event_Type" in df.columns:
        # Generate formatted event log
        fps = 30 # Assumption if not provided
        log_entries = []

        for _, row in df.iterrows():
            decision = str(row.get("Decision", ""))
            event = str(row.get("Event_Type", ""))
            if decision in ["IN", "OUT", "NET"] or event in ["player_hit"]:
                frame_idx = row["Frame_Index"]
                seconds = int(frame_idx) // fps
                mm, ss = divmod(seconds, 60)
                timestamp = f"{mm:02d}:{ss:02d}"

                if decision == "IN" and event == "bounce":
                    log_entries.append(f"{timestamp} - Ball IN (Bounce)")
                elif decision == "OUT":
                    # Use event to provide context
                    if event == "glass_hit":
                        log_entries.append(f"{timestamp} - Ball OUT (Hit Glass First)")
                    else:
                        log_entries.append(f"{timestamp} - Ball OUT")
                elif decision == "NET":
                    log_entries.append(f"{timestamp} - Ball NET")
                elif event == "player_hit" and decision == "":
                     log_entries.append(f"{timestamp} - Player Hit")

        if log_entries:
            for entry in log_entries:
                st.text(entry)
        else:
             st.write("No specific referee events recorded.")
    else:
        st.info("Event data not available in CSV.")

st.markdown("---")
st.markdown("*PadelSense Professional - AI Referee & Analytics Platform*")
