import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

file_path = Path(__file__).with_name("traffic_experiments.csv")

data = pd.read_csv(file_path)

print("Our dataset:")
print(data.head())

normal = data[data["traffic_condition"] == "Normal"]
congested = data[data["traffic_condition"] == "Congested"]

normal_average = normal["route_time_minutes"].mean()
congested_average = congested["route_time_minutes"].mean()

print("Normal average:", normal_average)
print("Congested average:", congested_average)
difference = congested_average - normal_average

print("Extra travel time:", difference, "minutes")

conditions = ["Normal", "Congested"]
average_times = [normal_average, congested_average]

plt.bar(conditions, average_times)

plt.title("Impact of Traffic Congestion")
plt.xlabel("Traffic Condition")
plt.ylabel("Average Route Time (minutes)")

plt.ylim(0, max(average_times) + 3)

plt.tight_layout()

chart_path = Path(__file__).with_name("traffic_comparison.png")
plt.savefig(chart_path)

plt.show()