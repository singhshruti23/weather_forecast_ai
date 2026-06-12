import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Weather Forecast AI", page_icon=":cloud:", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(160deg, #FBF4FF 0%, #F3E0F7 45%, #E6D0F0 100%);
    }
    h1 {
        text-align: center;
        color: #6B4E8A;
        font-family: 'Georgia', serif;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 0px;
    }
    .subtitle {
        text-align: center;
        color: #A584B8;
        font-size: 16px;
        font-style: italic;
        margin-bottom: 30px;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(6px);
        padding: 28px;
        border-radius: 22px;
        margin-top: 20px;
        border: 1px solid rgba(255,255,255,0.6);
        box-shadow: 0px 8px 24px rgba(150,100,180,0.12);
    }
    .gauge-wrap {
        display: flex;
        justify-content: center;
        margin-bottom: 10px;
    }
    .gauge {
        width: 160px;
        height: 160px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: conic-gradient(#C77DD2 var(--pct), rgba(255,255,255,0.4) 0);
        box-shadow: inset 0px 0px 14px rgba(150,100,180,0.18);
    }
    .gauge-inner {
        width: 122px;
        height: 122px;
        border-radius: 50%;
        background: #FDF7FF;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        box-shadow: inset 0px 2px 6px rgba(150,100,180,0.12);
    }
    .gauge-value {
        font-size: 34px;
        font-weight: 700;
        color: #9B4F96;
        font-family: 'Georgia', serif;
    }
    .gauge-label {
        font-size: 13px;
        color: #A584B8;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #C77DD2 0%, #F4A6C9 100%);
        color: #ffffff;
        border: none;
        border-radius: 14px;
        padding: 12px 24px;
        font-size: 17px;
        font-weight: 600;
        width: 100%;
        letter-spacing: 1px;
        transition: opacity 0.2s ease;
    }
    div.stButton > button:hover {
        opacity: 0.85;
    }
    .result-card {
        text-align: center;
        padding: 10px 0px 4px 0px;
    }
    .result-label {
        font-size: 13px;
        color: #A584B8;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 6px;
    }
    .result-value {
        font-size: 30px;
        font-weight: 700;
        font-family: 'Georgia', serif;
    }
    [data-testid="stSlider"] label {
        color: #8A5FA0;
        font-weight: 600;
        font-size: 14px;
    }
    .section-title {
        color: #6B4E8A;
        font-family: 'Georgia', serif;
        font-weight: 600;
        border-bottom: 1px solid rgba(150,100,180,0.2);
        padding-bottom: 8px;
        margin-bottom: 14px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>Weather Forecast AI</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A soft, dreamy view of your sensor readings</div>", unsafe_allow_html=True)

model = joblib.load("models/weather_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

gauge_html = """
<div class="gauge-wrap">
    <div class="gauge" style="--pct: {pct}%;">
        <div class="gauge-inner">
            <div class="gauge-value">{temp:.1f}</div>
            <div class="gauge-label">deg C</div>
        </div>
    </div>
</div>
"""

gauge_placeholder = st.empty()
gauge_placeholder.markdown(gauge_html.format(pct=50, temp=25.0), unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    temperature = st.slider("Temperature (C)", 0.0, 50.0, 25.0)
with col2:
    humidity = st.slider("Humidity (%)", 0, 100, 65)
with col3:
    pressure = st.slider("Pressure (hPa)", 950, 1050, 1012)

pct = max(0, min(100, (temperature / 50.0) * 100))
gauge_placeholder.markdown(gauge_html.format(pct=pct, temp=temperature), unsafe_allow_html=True)

predict_clicked = st.button("Predict Weather")
st.markdown("</div>", unsafe_allow_html=True)

if predict_clicked:
    data = np.array([[temperature, humidity, pressure]])
    prediction = model.predict(data)
    weather_label = label_encoder.inverse_transform(prediction)[0]

    colors = {"Cool": "#7FA6D6", "Normal": "#9FC2A8", "Warm": "#F4A6C9", "Hot": "#D2618C"}
    color = colors.get(weather_label, "#A584B8")

    st.markdown(
        f"""
        <div class="glass-card result-card">
            <div class="result-label">Predicted Condition</div>
            <div class="result-value" style="color:{color}">{weather_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Recent Temperature and Humidity Trend</div>", unsafe_allow_html=True)
try:
    df = pd.read_csv("data/weather_data.csv")
    st.line_chart(df[["Temperature_C", "Humidity"]].tail(30))
except Exception:
    st.warning("Data file not found.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Model Performance</div>", unsafe_allow_html=True)
st.image("graphs/confusion_matrix.png")
st.markdown("</div>", unsafe_allow_html=True)