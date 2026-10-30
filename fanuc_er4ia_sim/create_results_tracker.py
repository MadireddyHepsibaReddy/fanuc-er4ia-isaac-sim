
import csv
from pathlib import Path

source = Path("test_scenarios.csv")
output = Path("experiment_results.csv")

fields = [
    "scenario_id",
    "seed",
    "cube_id",
    "color",
    "actual_x_m",
    "actual_y_m",
    "actual_yaw_deg",
    "predicted_x_m",
    "predicted_y_m",
    "predicted_yaw_deg",
    "detected",
    "color_correct",
    "position_error_mm",
    "orientation_error_deg",
    "grasp_success",
    "placement_success",
    "notes"
]

with source.open(newline="") as file:
    scenarios = list(csv.DictReader(file))

with output.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for cube in scenarios:
        writer.writerow({
            "scenario_id": cube["scenario_id"],
            "seed": cube["seed"],
            "cube_id": cube["cube_id"],
            "color": cube["color"],
            "actual_x_m": cube["x_m"],
            "actual_y_m": cube["y_m"],
            "actual_yaw_deg": cube["yaw_deg"]
        })

print(f"Created {output} with {len(scenarios)} records.")
print("Prediction and execution fields are blank until tested.")
