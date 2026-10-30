
import csv
import random
from pathlib import Path

SOURCE = Path("test_scenarios.csv")
OUTPUT = Path("orientation_scenarios.csv")

# Reproducible random number generator
rng = random.Random(20261015)

# Orientation test range in degrees
MIN_YAW = -45
MAX_YAW = 45

with SOURCE.open(newline="") as file:
    scenarios = list(csv.DictReader(file))

updated_rows = []

for row in scenarios:
    new_row = row.copy()

    # Generate a new cube orientation
    new_row["yaw_deg"] = round(
        rng.uniform(MIN_YAW, MAX_YAW), 2
    )

    updated_rows.append(new_row)

with OUTPUT.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=updated_rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(updated_rows)

print(f"Created {OUTPUT}")
print(f"Total cube records: {len(updated_rows)}")
print("Orientations randomized between -45 and 45 degrees")
