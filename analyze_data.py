import pandas as pd
import matplotlib.pyplot as plt

file_path = "data/bqeb_data_v1.csv"

df = pd.read_csv(file_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])

print("Total records:", len(df))
print("Start time:", df["timestamp"].min())
print("End time:", df["timestamp"].max())

print("\nLoad Statistics:")
print(df["load_mw"].describe())

print("\nCyber Alert Counts:")
print(df["cyber_alert_flag"].value_counts())

print("\nAttack Classes:")
print(df["attack_class"].value_counts())

plt.figure(figsize=(12, 5))
plt.plot(df["timestamp"], df["load_mw"])
plt.xlabel("Time")
plt.ylabel("Load (MW)")
plt.title("Smart Grid Electricity Load")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()