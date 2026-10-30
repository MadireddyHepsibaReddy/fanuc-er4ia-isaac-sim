
import csv
import scene_config as cfg

rows = []

# Generate 10 reproducible arrangements
for scenario_id in range(1, 11):
    seed = 20261009 + scenario_id
    cubes = cfg.hackathon_cubes(seed)

    for i, (x, y, color) in enumerate(cubes):
        rows.append({
            "scenario_id": scenario_id,
            "seed": seed,
            "cube_id": f"cube_{i+1:02d}",
            "color": color,
            "x_m": round(x, 6),
            "y_m": round(y, 6),
            "yaw_deg": cfg.SORT_CUBE_YAWS_DEG[i]
        })

with open("test_scenarios.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {scenario_id} test scenarios")
print(f"Total cube records: {len(rows)}")
print("Saved to test_scenarios.csv")
