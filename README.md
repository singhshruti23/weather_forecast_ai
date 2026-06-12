# Weather Forecast AI

A complete end-to-end project that combines an ESP32-based temperature sensor (LM35) with a machine learning model to predict weather conditions (Cool, Normal, Warm, Hot) based on temperature, humidity, and pressure readings.

## Project Overview

This project has two parts:

1. **ESP32 C6 + LM35 Sensor (Hardware)** — Continuously monitors temperature, stores readings with timestamps, calculates min/max/avg temperature, and allows users to download the recorded data as a CSV file via a web interface.

2. **Weather Forecast AI (Software)** — A Python/Streamlit application that loads the CSV sensor data, trains a machine learning model, and provides an interactive dashboard to predict weather conditions based on live sensor readings.

## Project Structure
weather_forecast_ai/

├── data/

│   └── weather_data.csv

├── models/

│   ├── weather_model.pkl

│   └── label_encoder.pkl

├── graphs/

│   └── confusion_matrix.png

├── venv/

├── app.py

├── forecast.py

├── requirements.txt

└── README.md

## CSV Data Format

| Column | Description |
|---|---|
| Timestamp | Date and time of the sensor reading |
| Temperature_C | Current temperature in Celsius |
| Humidity | Humidity percentage |
| Pressure | Atmospheric pressure (hPa) |
| Min_Temp_C | Rolling minimum temperature |
| Max_Temp_C | Rolling maximum temperature |
| Avg_Temp_C | Rolling average temperature |
| Weather_Label | Weather category (Cool, Normal, Warm, Hot) |

## Setup Instructions

### 1. Create and activate a virtual environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies
```powershell
pip install -r requirements.txt
```

Dependencies: pandas, numpy, matplotlib, scikit-learn, streamlit, joblib, seaborn

## How to Use

### Step 1 - Train the Model
```powershell
python forecast.py
```

### Step 2 - Run the Dashboard
```powershell
streamlit run app.py
```
Opens at `http://localhost:8501`

### Step 3 - Access from a Phone
Use the Network URL shown in the terminal (e.g. `http://192.168.1.5:8501`) on a device connected to the same WiFi.

## Model Details

- Algorithm: Random Forest Classifier (scikit-learn)
- Features used: Temperature_C, Humidity, Pressure
- Target: Weather_Label (Cool, Normal, Warm, Hot)
- Dataset: 200 sample readings, balanced across all categories

## Progress So Far

- [x] Project folder structure created
- [x] Virtual environment set up
- [x] Required libraries installed
- [x] Sample weather dataset prepared
- [x] Model trained and saved
- [x] Interactive Streamlit dashboard built
- [x] Dashboard tested locally and on phone
- [ ] ESP32 + LM35 sensor firmware
- [ ] Connect real sensor data to the model

## Next Steps

- Write ESP32 firmware to read temperature from the LM35 sensor
- Set up a web server on the ESP32 to log readings and allow CSV download
- Replace sample dataset with real sensor data
- Retrain the model on real-world data