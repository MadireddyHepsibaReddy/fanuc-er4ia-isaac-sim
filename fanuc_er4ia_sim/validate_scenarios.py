
import csv
import math
from collections import Counter, defaultdict

import scene_config as cfg

scenarios = defaultdict(list)

with open("test_scenarios.csv", newline="") as f:
    for row in csv.DictReader(f):
        scenarios[int(row["scenario_id"])].append(row)

assert len(scenarios) == 10, "Expected 10 scenarios"

for scenario_id, cubes in sorted(scenarios.items()):
    assert len(cubes) == 6, f"Scenario {scenario_id}: wrong cube count"

    colors = Counter(c["color"] for c in cubes)
    assert colors == {"yellow": 2, "red": 2, "blue": 2}

    positions = [
        (float(c["x_m"]), float(c["y_m"]))
        for c in cubes
    ]

    # Check that cube centers are separated.
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distance = math.dist(positions[i], positions[j])
            assert distance >= 0.065 - 0.000002, (
                f"Scenario {scenario_id}: cubes too close "
                f"({distance:.4f} m)"
            )

    # Check basic configured sampling-area bounds.
    for x, y in positions:
        assert 0.24 <= x <= 0.46
        assert -0.15 <= y <= 0.28

    print(f"Scenario {scenario_id}: PASS")

print("\nAll 10 scenario checks passed!")

