
import json
from pathlib import Path

from load_scenario import load_scenario

for variant in ("position", "orientation"):
    scenario_id = 1
    path = Path("scenario_configs") / (
        f"scenario_{scenario_id:02d}_{variant}.json"
    )

    with path.open() as file:
        config = json.load(file)

    expected = load_scenario(scenario_id, variant)
    actual = config["cubes"]

    assert config["scenario_id"] == scenario_id
    assert config["variant"] == variant
    assert len(actual) == 6

    for saved, original in zip(actual, expected):
        assert saved["id"] == original["cube_id"]
        assert saved["color"] == original["color"]
        assert abs(saved["position"][0] - original["x_m"]) < 1e-9
        assert abs(saved["position"][1] - original["y_m"]) < 1e-9
        assert abs(saved["yaw_deg"] - original["yaw_deg"]) < 1e-9

    print(f"Scenario 1 ({variant}): PASS")

print("Both JSON configurations validated!")
