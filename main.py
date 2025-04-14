import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ----------------------------
# CONFIG
# ----------------------------
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
# SECTION: VIDEO DISPLAY
# ----------------------------
st.title("🎬 Gaze Hull Visualization")

selected_video = st.selectbox("🎥 เลือกวิดีโอ", list(video_files.keys()))
video_url = base_url + video_files[selected_video]
st.video(video_url)

# ----------------------------
# SECTION: MOCK GRAPH
# ----------------------------
st.markdown("## 📈 Convex & Concave Area Graph")

# สร้าง mock data สำหรับแต่ละวิดีโอ
np.random.seed(list(video_files.keys()).index(selected_video))
frames = 100
df = pd.DataFrame({
    "frame": np.arange(frames),
    "convex": 60000 + np.random.normal(0, 1500, frames).cumsum(),
    "concave": 50000 + np.random.normal(0, 1000, frames).cumsum()
})
df["convex_smooth"] = df["convex"].rolling(10, min_periods=1).mean()
df["concave_smooth"] = df["concave"].rolling(10, min_periods=1).mean()

# สร้าง chart
df_long = df.melt(id_vars="frame", value_vars=["convex_smooth", "concave_smooth"],
                  var_name="type", value_name="value")

current_frame = st.slider("🕓 เลือกเฟรม", min_value=0, max_value=frames-1, value=0)

chart = alt.Chart(df_long).mark_line().encode(
    x="frame",
    y="value",
    color=alt.Color("type", scale=alt.Scale(
        domain=["convex_smooth", "concave_smooth"],
        range=["green", "blue"]
    ))
).properties(width=700, height=300)

rule = alt.Chart(pd.DataFrame({"frame": [current_frame]})).mark_rule(color="red").encode(x="frame")

st.altair_chart(chart + rule, use_container_width=True)

# ----------------------------
# SECTION: F-C Score
# ----------------------------
df["fc_score"] = 1 - (df["convex_smooth"] - df["concave_smooth"]) / df["convex_smooth"]
st.metric("🔢 F-C Score", f"{df.loc[current_frame, 'fc_score']:.3f}")
