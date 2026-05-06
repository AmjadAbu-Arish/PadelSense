import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="PadelSense Dashboard", layout="wide")

st.title("🎾 PadelSense Analytics Dashboard")

# Paths
MATCH_SUMMARY_PATH = "outputs/match_summary.csv"
VIDEO_PATH = "outputs/output_video.mp4"
HEATMAP_PATH = "outputs/heatmap.png"

# Load data
@st.cache_data
def load_data():
    if os.path.exists(MATCH_SUMMARY_PATH):
        df = pd.read_csv(MATCH_SUMMARY_PATH)
        return df
    return None

df = load_data()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Main Match Video Feed")
    if os.path.exists(VIDEO_PATH):
        st.video(VIDEO_PATH)
    else:
        st.info("No output video found. Please run the main script first.")

    st.subheader("Ball Speed Analysis")
    if df is not None and "Ball_Speed_kmh" in df.columns:
        df_speed = df[pd.to_numeric(df["Ball_Speed_kmh"], errors='coerce') > 0].copy()
        if not df_speed.empty:
            df_speed["Ball_Speed_kmh"] = pd.to_numeric(df_speed["Ball_Speed_kmh"])
            st.line_chart(df_speed, x="Frame_Index", y="Ball_Speed_kmh", use_container_width=True)
            st.metric(label="Max Speed", value=f"{df_speed['Ball_Speed_kmh'].max():.1f} km/h")
        else:
            st.info("No ball speed > 0 found.")
    else:
        st.info("Speed data not available in CSV.")

with col2:
    st.subheader("Mini-Court Tracking (Heatmap)")
    if os.path.exists(HEATMAP_PATH):
        st.image(HEATMAP_PATH, caption="2D Density Map of Ball Positions", use_column_width=True)
    else:
        st.info("Heatmap image not found. Please run the main script first.")

    st.subheader("Event Log (Referee Decisions)")
    if df is not None and "Event_Type" in df.columns:
        # Filter for rows that have a significant event or decision
        df_events = df[(df["Event_Type"] != "none") | (df["Decision"].notna() & (df["Decision"] != ""))].copy()
        if not df_events.empty:
            df_events["Timestamp"] = (df_events["Frame_Index"] / 30.0).apply(lambda x: f"{int(x//60):02d}:{int(x%60):02d}")
            # Format Event Log text
            def format_log(row):
                msg = row['Event_Type'].capitalize()
                if pd.notna(row['Decision']) and row['Decision'] != "":
                    msg += f" - Decision: {row['Decision']}"
                return msg
            df_events["Log_Entry"] = df_events.apply(format_log, axis=1)
            event_log_display = df_events[["Timestamp", "Log_Entry"]].reset_index(drop=True)
            st.dataframe(event_log_display, use_container_width=True)
        else:
            st.write("No specific referee events recorded.")
    else:
        st.info("Event data not available in CSV.")

st.markdown("---")
st.markdown("*PadelSense Professional - AI Referee & Analytics Platform*")
