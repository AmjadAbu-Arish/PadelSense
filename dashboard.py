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

    st.subheader("Ball Speed Analysis (km/h)")
    if df is not None and "Ball_Speed_kmh" in df.columns:
        # Filter out rows with speed = 0.0 for a cleaner plot (optional)
        df['Ball_Speed_kmh'] = pd.to_numeric(df['Ball_Speed_kmh'], errors='coerce')
        df_speed = df[df['Ball_Speed_kmh'] > 0]
        st.line_chart(df_speed, x="Frame", y="Ball_Speed_kmh", use_container_width=True)

        st.metric(label="Max Speed", value=f"{df['Speed_kmh'].max():.1f} km/h")
        st.metric(label="Average Speed", value=f"{df_speed['Speed_kmh'].mean():.1f} km/h")
    else:
        st.info("Speed data not available in CSV.")

with col2:
    st.subheader("Ball Position Heatmap")
    if os.path.exists(HEATMAP_PATH):
        st.image(HEATMAP_PATH, caption="2D Density Map of Ball Positions", use_column_width=True)
    else:
        st.info("Heatmap image not found. Please run the main script first.")

    st.subheader("Match Event Log")
    if df is not None and "Event_Type" in df.columns:
        # Filter relevant events
        df_events = df[(df["Event_Type"] != "none") & (df["Event_Type"].notna()) & (df["Event_Type"] != "")]
        if not df_events.empty:
            for idx, row in df_events.iterrows():
                frame_idx = row['Frame_Index']
                event_type = row['Event_Type'].upper()
                decision = row['Decision'] if 'Decision' in row and pd.notna(row['Decision']) and row['Decision'] != "" else ""

                # Assume 30 fps for timestamp
                fps = 30.0
                total_seconds = frame_idx / fps
                mins = int(total_seconds // 60)
                secs = int(total_seconds % 60)
                timestamp = f"{mins:02d}:{secs:02d}"

                if decision:
                    log_text = f"**{timestamp}** - {decision} ({event_type})"
                else:
                    log_text = f"**{timestamp}** - {event_type}"

                st.markdown(f"- {log_text}")
        else:
            st.write("No specific referee events (Out, Net, etc.) recorded.")
    else:
        st.info("Event data not available in CSV.")

st.markdown("---")
st.markdown("*PadelSense Professional - AI Referee & Analytics Platform*")
