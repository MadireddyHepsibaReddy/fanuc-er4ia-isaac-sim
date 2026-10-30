
import csv
import sys
from collections import Counter
from pathlib import Path

DATA_FILES = {
    "position": "test_scenarios.csv",
    "orientation": "orientation_scenarios.csv",
}


def load_scenario(scenario_id, variant="position"):
    if variant not in DATA_FILES:
        raise ValueError(f"Unknown variant: {variant}")

    data_file = Path(__file__).with_name(DATA_FILES[variant])

    with data_file.open(newline="") as file:
        rows = [
            row for row in csv.DictReader(file)
            if int(row["scenario_id"]) == scenario_id
        ]

    if len(rows) != 6:
        raise ValueError(
            f"Expected 6 cubes for scenario {scenario_id}; "
            f"found {len(rows)}"
        )

    colors = Counter(row["color"] for row in rows)

    if colors != {"yellow": 2, "red": 2, "blue": 2}:
        raise ValueError("Invalid color distribution")

    return [
        {
            "cube_id": row["cube_id"],
            "color": row["color"],
            "x_m": float(row["x_m"]),
            "y_m": float(row["y_m"]),
            "yaw_deg": float(row["yaw_deg"]),
        }
        for row in rows
    ]


if __name__ == "__main__":
    selected = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    variant = sys.argv[2] if len(sys.argv) > 2 else "position"

    cubes = load_scenario(selected, variant)
    print(f"Dataset variant: {variant}")

    print(f"Loaded scenario {selected}")

    for cube in cubes:
        print(
            f"{cube['cube_id']} | "
            f"{cube['color']:6} | "
            f"X={cube['x_m']:.4f} m | "
            f"Y={cube['y_m']:.4f} m | "
            f"Yaw={cube['yaw_deg']:.1f}°"
        )
