"""Use the existing perception code and its loaded model for a grounded explanation."""
import argparse
import json
from pathlib import Path
import runpy
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True, type=Path)
parser.add_argument("--run", required=True, type=Path)
args = parser.parse_args()
project, run = args.project.resolve(), args.run.resolve()
sys.path.append(str(project))
sys.argv = [str(project / "perceive.py"), "--run", str(run), "--stack-cubes"]
values = runpy.run_path(str(project / "perceive.py"), run_name="__main__")
evidence = json.loads((run / "change_evidence.json").read_text())
summary = {key: evidence[key] for key in ("colour", "expected_xy_m", "measured_xy_m", "displacement_mm", "status")}
prompt = """Inspect the current cube crop and explain ONLY the supported pre-grasp problem.
The supplied calibrated image measurements compare the original vision plan
with this fresh image. No grasp has been attempted. The physical cause of any
position change is unobserved: do not blame the robot, a human, friction or a
failed grasp. This is a hybrid image-geometry and semantic perception system.
Return one JSON object with ALL these fields:
observed_change: target_position_changed, or unknown if evidence is insufficient;
cause: unknown;
grasp_attempted: false;
recommended_action: replan or request_human;
explanation: one short sentence about the measured change and stale grasp risk.
Use request_human if the image and measurements are inconsistent or ambiguous.
Image-derived evidence follows:\n""" + json.dumps(summary)
(run / "explanation_prompt.txt").write_text(prompt)
raw = values["ask"](values["model"], values["processor"], run / "changed_target_crop.png", prompt, 500)
(run / "ai_explanation_raw.txt").write_text(raw)
answer = values["parse_json"](raw)
if not isinstance(answer, dict):
    raise ValueError("AI explanation did not contain a JSON object")
(run / "ai_explanation.json").write_text(json.dumps(answer, indent=2) + "\n")
