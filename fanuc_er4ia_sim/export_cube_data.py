
import csv
import scene_config as cfg

# Read the configured cube positions and orientations
rows = []

for i, (x, y, color) in enumerate(cfg.SORT_CUBES):
    rows.append({
        "cube_id": f"cube_{i + 1:02d}",
        "color": color,
        "x_m": x,
        "y_m": y,
        "yaw_deg": cfg.SORT_CUBE_YAWS_DEG[i]
    })

# Save the dataset
with open("cube_layout.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Success! Exported {len(rows)} cubes.")
print("Dataset saved as cube_layout.csv")

