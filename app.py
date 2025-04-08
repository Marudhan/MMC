import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Material Property Predictor", layout="centered")

st.title("🔍 AI-Based Material Property Predictor")
st.markdown("Predict **Fatigue Strength** and **Wear Strength** from experimental values and processing parameters.")

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()
st.write("### Preview of Dataset", df.head())

# Drop rows with missing target values
df = df.dropna(subset=["Fatigue Strength", "Wear Strength"])

# Experimental properties used for training
features = [
    "AL", "B4C", "BioChar",
    "Stirrer Speed", "Melt Temp", "Furnace Temp", "Reinforcement Preheat Temp",
    "Tensile Strength (MPa)", "Hardness (BHN)", "Flexural Strength (MPa)",
    "Bending Stiffness (N-mm2)", "Displacement (mm)", "Impact Strength (J)"
]

X = df[features]
y = df[["Fatigue Strength", "Wear Strength"]]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# User input
st.write("### 🔧 Enter Composition and Parameters for Prediction")
user_input = {
    "AL": st.slider("Aluminium %", 80, 100, 90),
    "B4C": st.slider("B4C %", 0, 15, 5),
    "BioChar": st.slider("BioChar %", 0, 15, 5),
    "Stirrer Speed": st.slider("Stirrer Speed (rpm)", 100, 1000, 500),
    "Melt Temp": st.slider("Melt Temperature (°C)", 600, 800, 700),
    "Furnace Temp": st.slider("Furnace Temperature (°C)", 600, 800, 700),
    "Reinforcement Preheat Temp": st.slider("Reinforcement Preheat Temp (°C)", 200, 500, 300),
    "Tensile Strength (MPa)": st.number_input("Tensile Strength (MPa)", 0.0, 1000.0, 130.0),
    "Hardness (BHN)": st.number_input("Hardness (BHN)", 0.0, 200.0, 100.0),
    "Flexural Strength (MPa)": st.number_input("Flexural Strength (MPa)", 0.0, 1500.0, 700.0),
    "Bending Stiffness (N-mm2)": st.number_input("Bending Stiffness (N-mm2)", 0.0, 2e7, 1e7),
    "Displacement (mm)": st.number_input("Displacement (mm)", 0.0, 50.0, 15.0),
    "Impact Strength (J)": st.number_input("Impact Strength (J)", 0.0, 50.0, 12.0)
}

input_df = pd.DataFrame([user_input])
prediction = model.predict(input_df)

st.write("## 🎯 Predicted Properties")
st.write(f"**Fatigue Strength (MPa)**: `{prediction[0][0]:.2f}`")
st.write(f"**Wear Strength (mm³)**: `{prediction[0][1]:.4f}`")

st.write("---")
st.markdown("Built with ❤️ using Streamlit")