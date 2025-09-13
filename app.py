import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ------------------ APP TITLE ------------------
st.set_page_config(page_title="Hostel Energy Wastage Dashboard", layout="wide")
st.title("🏫 Hostel Energy Wastage Detection Dashboard")

# ------------------ LOAD DATA ------------------
DATA_FILE = "hostel_energy_realistic_final2.csv"

try:
    df = pd.read_csv(DATA_FILE)

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    # ------------------ CHECK REQUIRED COLUMNS ------------------
    required_columns = ["Room", "Lights_W", "Fan_W", "Laptop_W", "Heater_W"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        st.error(f"❌ Missing required columns in dataset: {', '.join(missing)}")
    else:
        # ------------------ FEATURE SELECTION ------------------
        features = df[required_columns]

        # Standardize data
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # ------------------ KMEANS CLUSTERING ------------------
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df["Cluster"] = kmeans.fit_predict(scaled_features)

        # ------------------ MAP CLUSTERS TO LABELS ------------------
        cluster_means = df.groupby("Cluster")[["Lights_W", "Fan_W", "Laptop_W", "Heater_W"]].mean()
        order = cluster_means.mean(axis=1).sort_values().index.tolist()
        mapping = {order[0]: "Low", order[1]: "Medium", order[2]: "High"}
        df["Wastage_Level"] = df["Cluster"].map(mapping)

        # ------------------ DASHBOARD ------------------
        st.subheader("📊 Wastage Level Summary")
        summary = df["Wastage_Level"].value_counts().reset_index()
        summary.columns = ["Wastage_Level", "Count"]
        st.bar_chart(summary.set_index("Wastage_Level"))

        # ------------------ ROOM STATUS ------------------
        st.subheader("🏠 Room-wise Wastage Levels")
        st.dataframe(df[["Room", "Day", "Time", "Lights_W", "Fan_W", "Laptop_W", "Heater_W", "Total_Energy_W", "Wastage_Level"]])

        # ------------------ ALERTS ------------------
        st.subheader("🚨 Real-Time Alerts")
        high_rooms = df[df["Wastage_Level"] == "High"]["Room"].unique().tolist()
        if high_rooms:
            st.warning(f"⚠️ High wastage detected in rooms: {', '.join(map(str, high_rooms))}")
        else:
            st.success("✅ No rooms currently in High Wastage.")

        # ------------------ TIME-SERIES VISUALIZATION ------------------
        if "Time" in df.columns:
            st.subheader("⏱ Energy Usage Over Time (Sample Room)")
            sample_room = df["Room"].iloc[0]
            room_data = df[df["Room"] == sample_room]

            fig, ax = plt.subplots()
            ax.plot(room_data["Time"], room_data["Lights_W"], label="Lights")
            ax.plot(room_data["Time"], room_data["Fan_W"], label="Fan")
            ax.plot(room_data["Time"], room_data["Laptop_W"], label="Laptop")
            ax.plot(room_data["Time"], room_data["Heater_W"], label="Heater")
            ax.set_title(f"Energy Usage Over Time - Room {sample_room}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Power (W)")
            ax.legend()
            st.pyplot(fig)

except FileNotFoundError:
    st.error(f"❌ Could not find `{DATA_FILE}`. Please place it in the same folder as app.py.")
