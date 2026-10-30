
import json
from collections import Counter
from pathlib import Path


def load_simulation_configuration(json_path):
    path = Path(json_path)

    with path.open(encoding="utf-8") as file:
        config = json.load(file)

    cubes = config["cubes"]

    if len(cubes) != 6:
        raise ValueError("Expected exactly six cubes")

    colors = Counter(cube["color"] for cube in cubes)

    if colors != {"yellow": 2, "blue": 2, "red": 2}:
        raise ValueError("Unexpected cube colors")

    positions = []
    orientations = []

    for cube in cubes:
        x, y = cube["position"]

        positions.append((
            float(x),
            float(y),
            cube["color"]
        ))

        orientations.append(float(cube["yaw_deg"]))

    return positions, orientations


if __name__ == "__main__":
    positions, orientations = load_simulation_configuration(
        "scenario_configs/scenario_01_orientation.json"
    )

    print("Simulation cube positions:")
    for cube in positions:
        print(cube)

    print("\nSimulation cube orientations:")
    print(orientations)
