import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Smart Grid Monitoring System",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Smart Grid Load & Cyber-Anomaly Prediction")
st.write("AI-based monitoring system for electricity load and cyber anomalies")

df = pd.read_csv("data/dashboard_predictions.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

st.success("Dashboard data loaded successfully!")

st.subheader("📊 Smart Grid Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Current Load",
    f"{df.iloc[0]['current_load_mw']:.2f} MW"
)

col2.metric(
    "Predicted Next-Hour Load",
    f"{df.iloc[0]['predicted_next_hour_load_mw']:.2f} MW"
)

col3.metric(
    "Cyber Alerts",
    int(df["cyber_alert_actual"].sum())
)

st.subheader("⚡ Electricity Load")

load_chart = df[
    ["timestamp", "current_load_mw", "predicted_next_hour_load_mw"]
].copy()

load_chart = load_chart.sort_values("timestamp")

load_chart = load_chart.set_index("timestamp")

st.line_chart(load_chart)

st.subheader("🛡️ Cyber Security")

if df["cyber_alert_actual"].sum() > 0:
    st.warning(
        f"{int(df['cyber_alert_actual'].sum())} cyber alerts are present in the dataset."
    )
else:
    st.success("No cyber alerts detected.")

st.subheader("🚨 Attack Types")

attack_counts = df["attack_actual"].value_counts()

attack_table = attack_counts.reset_index()
attack_table.columns = ["Attack Type", "Count"]

st.dataframe(
    attack_table,
    use_container_width=True,
    hide_index=True
)

st.bar_chart(
    attack_table.set_index("Attack Type")
)

st.subheader("🔍 Latest Prediction")

latest = df.iloc[0]

result_col1, result_col2 = st.columns(2)

result_col1.metric(
    "Predicted Load",
    f"{latest['predicted_next_hour_load_mw']:.2f} MW"
)

if latest["cyber_alert_predicted"] == 1:
    result_col2.error("🚨 Cyber Alert Detected")
else:
    result_col2.success("✅ System Normal")

st.write(
    "Predicted Attack Type:",
    latest["attack_predicted"]
)

st.subheader("📈 Model Performance")

st.write("Load Prediction Model")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", "8.28 MW")
col2.metric("RMSE", "10.36 MW")
col3.metric("R² Score", "91.75%")

st.write("Cyber Anomaly Detection")

col1, col2, col3 = st.columns(3)

col1.metric("Precision", "1.00")
col2.metric("Recall", "0.45")
col3.metric("F1 Score", "0.62")

st.write("Attack Classification")

st.metric("Macro F1 Score", "0.62")