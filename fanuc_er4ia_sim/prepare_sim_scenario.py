
import argparse
import json
from pathlib import Path

from load_scenario import load_scenario


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, required=True)
    parser.add_argument(
        "--variant",
        choices=["position", "orientation"],
        default="position"
    )
    args = parser.parse_args()

    cubes = load_scenario(args.scenario, args.variant)

    output_dir = Path("scenario_configs")
    output_dir.mkdir(exist_ok=True)

    configuration = {
        "scenario_id": args.scenario,
        "variant": args.variant,
        "cubes": [
            {
                "id": cube["cube_id"],
                "color": cube["color"],
                "position": [
                    cube["x_m"],
                    cube["y_m"]
                ],
                "yaw_deg": cube["yaw_deg"]
            }
            for cube in cubes
        ]
    }

    output = (
        output_dir /
        f"scenario_{args.scenario:02d}_{args.variant}.json"
    )

    with output.open("w") as file:
        json.dump(configuration, file, indent=2)

    print(f"Created {output}")


if __name__ == "__main__":
    main()
