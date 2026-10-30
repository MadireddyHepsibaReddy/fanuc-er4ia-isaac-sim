
import csv
from collections import Counter

with open("cube_layout.csv", newline="") as file:
    cubes = list(csv.DictReader(file))

# Check the number of cubes
assert len(cubes) == 6, "Expected exactly 6 cubes"

# Check unique cube identifiers
ids = [cube["cube_id"] for cube in cubes]
assert len(ids) == len(set(ids)), "Duplicate cube IDs found"

# Check that each color appears twice
colors = Counter(cube["color"] for cube in cubes)
assert colors == {
    "yellow": 2,
    "red": 2,
    "blue": 2
}, f"Unexpected color distribution: {colors}"

# Check all coordinates and rotations are numeric
for cube in cubes:
    for column in ("x_m", "y_m", "yaw_deg"):
        float(cube[column])

print("All data quality checks passed!")
print(f"Total cubes: {len(cubes)}")
print(f"Color distribution: {dict(colors)}")
