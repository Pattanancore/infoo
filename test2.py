import os
import math
import cv2
import numpy as np
import pandas as pd
import scipy.io
import streamlit as st
import altair as alt
from scipy.spatial import ConvexHull
import alphashape
from shapely.geometry import MultiPoint

# GitHub video links
video_files = {
    "APPAL_2a": "APPAL_2a_hull_area.mp4",
    "FOODI_2a": "FOODI_2a_hull_area.mp4",
    "MARCH_12a": "MARCH_12a_hull_area.mp4",
    "NANN_3a": "NANN_3a_hull_area.mp4",
    "SHREK_3a": "SHREK_3a_hull_area.mp4",
    "SIMPS_19a": "SIMPS_19a_hull_area.mp4"
}
base_url = "https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/processed%20hull%20area%20overlay/"

# ----------------------------
# UI Layout
# ----------------------------
st.title("🎬 Gaze Hull Visualization (GitHub Video + Convex/Concave Graph)")

selected_video = st.selectbox("🎥 เลือกวิดีโอ", list(video_files.keys()))
video_url = base_url + video_files[selected_video]
st.video(video_url)

# ----------------------------
# Mock gaze-based convex/concave data
# ----------------------------
np.random.seed(list(video_files.keys()).index(selected_video))
frames = 100
fps = 25

# Simulate convex & concave area
df = pd.DataFrame({
    "Frame": np.arange(frames),
    "Convex Area": 60000 + np.random.normal(0, 1500, frames).cumsum(),
    "Concave Area": 50000 + np.random.normal(0, 1000, frames).cumsum()
})

# Rolling average and F-C score
df['Convex Area (Rolling Avg)'] = df['Convex Area'].rolling(10, min_periods=1).mean()
df['Concave Area (Rolling Avg)'] = df['Concave Area'].rolling(10, min_periods=1).mean()
df['F-C score'] = 1 - (df['Convex Area (Rolling Avg)'] - df['Concave Area (Rolling Avg)']) / df['Convex Area (Rolling Avg)']
df['F-C score'] = df['F-C score'].fillna(0)

# ----------------------------
# Frame slider
# ----------------------------
st.markdown("---")
st.markdown("### 📊 กราฟ Hull Area และค่า Score")

current_frame = st.slider("🕓 เลือกเฟรม", min_value=0, max_value=frames-1, value=0)

# Altair chart
df_melt = df.reset_index().melt(id_vars='Frame', value_vars=[
    'Convex Area (Rolling Avg)', 'Concave Area (Rolling Avg)'
], var_name='Metric', value_name='Area')

chart = alt.Chart(df_melt).mark_line().encode(
    x='Frame',
    y='Area',
    color=alt.Color(
        'Metric:N',
        scale=alt.Scale(
            domain=['Convex Area (Rolling Avg)', 'Concave Area (Rolling Avg)'],
            range=['green', 'blue']
        )
    )
).properties(
    width=600,
    height=300
)

rule = alt.Chart(pd.DataFrame({'Frame': [current_frame]})).mark_rule(color='red').encode(x='Frame')

col_chart, col_score = st.columns([2, 1])

with col_chart:
    st.altair_chart(chart + rule, use_container_width=True)

with col_score:
    st.metric("Focus-Concentration Score", f"{df.loc[current_frame, 'F-C score']:.3f}")
