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
st.title("🎯 Understanding Viewer Focus Through Gaze Visualization")

# Explanation Sections
with st.expander("📌 Goal of This Visualization", expanded=True):
    st.markdown("""
    Is the viewer's attention firmly focused on key moments, or does it float, drifting between different scenes in search of something new?
    
    The goal of this visualization is to understand how viewers engage with a video by examining **where and how they focus their attention**. By comparing the areas where viewers look (represented by **convex and concave hulls**), the visualization highlights whether their attention stays focused on a specific part of the video or shifts around.
    
    Ultimately, this helps us uncover **patterns of focus and exploration**, providing insights into how viewers interact with different elements of the video.
    """)
    
with st.expander("📐 Explain Convex and Concave Concept"):
    st.markdown("""
    To analyze visual attention, we enclose gaze points with geometric boundaries:
    - **Convex Hull** wraps around all gaze points to show the overall extent of where viewers looked.
    - **Concave Hull** creates a tighter boundary that closely follows the actual shape of the gaze pattern, adapting to gaps and contours in the data.
    
    👉 **The difference in area between them reveals how dispersed or concentrated the viewers' gaze is.**
    """)
    
with st.expander("📊 Focus-Concentration (F-C) Score"):
    st.markdown("""
    The **Focus Concentration Score (FCS)** quantifies how focused or scattered a viewer's attention is during the video:
    - A score **close to 1** → gaze is tightly grouped → **high concentration**.
    - A score **much lower than 1** → gaze is more spread out → **lower concentration / more exploration**.
    
    It helps to measure whether attention is **locked onto a specific spot** or **wandering across the frame**.
    """)
    
with st.expander("🎥 Example: High vs Low F-C Score"):
    st.markdown("""
    - **High F-C Score**: The viewer's gaze remains focused in one tight area, suggesting strong interest or attention.
    - **Low F-C Score**: The gaze is scattered, moving across many regions of the screen, indicating exploration or distraction.
    
    You can observe this difference visually in the graph and video overlays as you explore different frames.
    """)

# Video selection
st.markdown("### 🎬 Select a Video")
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
