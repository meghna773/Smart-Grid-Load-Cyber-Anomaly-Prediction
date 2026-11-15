import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file_path = "data/bqeb_data_v1.csv"

df = pd.read_csv(file_path)

features = [
    "load_mw",
    "residential_load_mw",
    "commercial_load_mw",
    "industrial_load_mw",
    "solar_mw",
    "wind_mw",
    "renewable_total_mw",
    "battery_soc_pct",
    "battery_charge_mw",
    "battery_discharge_mw",
    "ev_sessions_active",
    "ev_load_mw",
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms",
    "solar_irradiance_wm2",
    "rainfall_mm",
    "energy_price_usd_mwh",
    "peak_tariff_flag",
    "demand_response_event",
    "frequency_hz",
    "voltage_pu",
    "line_loss_pct"
]

target = "next_hour_load_mw"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("Load Prediction Model")
print("---------------------")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nModel Performance")
print("-----------------")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)