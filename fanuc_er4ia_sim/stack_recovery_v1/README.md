# Stacking recovery pilot v1

This is a separate source extension for the existing fanuc_er4ia_sim project.
It does not replace root project Python files or modify completed run folders.
It preserves the six-cube tower task: yellow, yellow, blue, blue, red, red.
The source reviewed was the uploaded stacking_recovery_source_review.tar.gz.

## Implemented scope

The saved baseline scene is restored in a NEW Isaac session. Before objects are
released into physics, the first yellow cube is repositioned +10 mm in world X.
There is also a zero-shift control. This is an authored kinematic scene edit,
not a physical push, natural slip or a robot-induced disturbance.

At home, before the first pickup, a fresh 4K table-camera image is captured.
Image geometry checks the first vision target against its original planned XY.
If recovery is enabled and an unambiguous movement is detected, the same Isaac
world remains loaded while a fresh Cosmos perception pass and one structured
AI explanation run. Physics stepping pauses during that reasoning interval.
Fresh pixels and calibration enter the existing perceive.py --stack-cubes path;
scene_state.json, ground_truth.json and disturbance_truth.json do not enter it.

The existing planner is called through a wrapper. The wrapper compares the
measured and refined cube hypotheses against the same pixels and retains the
measured hypothesis if its fit is at least as good and it lies within the
original search bounds. It does not override a rejected boundary fit. This is
the image-fit comparison validated in the separate colour-sorting revision v3.

The candidate must preserve all six colours/stack levels and original tower
positions/heights. Original confidence and stack validation remain. Additional
recovery gates require fit >=0.85, unique changed target, current-image XY
agreement within 2 mm, yaw agreement within 3 degrees modulo 90, only the first
cube changing, and movement within 5 to 25 mm. The movement check retains a
2 mm consistency tolerance, 5 mm moved threshold and uncertainty band between.
No threshold is lowered to make this pilot pass.

All 1,656 trajectory samples are checked by a host IK preflight before the new
plan is adopted. This does not verify collisions or physical stability.
The executor then resumes the SAME simulator session using the revised plan.
Its original pickup, placement, yaw/tilt and adjacent-layer support checks are
retained. A failure stops the tower sequence according to the existing logic.
There is no automatic second grasp, later-pick replanning or post-grasp vision
recovery in this version. Later motion video is recorded but no additional
pre-pick snapshots are taken by this bounded pilot.

Uncertain images or rejected candidates stop before the first grasp and write
human_review_request.json plus an explicit pre_grasp_review_stop execution
report. This is a LOCAL review record, not a notification or approve/resume UI.
Physical execution success is evaluated with simulator state. Ground truth is
used for experiment setup/evaluation only, never to correct grasp coordinates.

## Preservation and dependencies

run_stack_pilot.py checks package and installed project source hashes before
launching. It creates a runtime copy of sort_scene.py only within the new run.
source_revision/ preserves the reviewed original project Python files for that
attempt; session_status.json records source/package/input hashes, mode, shift
and outcomes. initial_sort_plan.json preserves the old plan and
active_sort_plan.json records the plan used for motion.

This first pilot accepts only the default cube layout and a saved successful
six-placement baseline, initially runs/baseline_03_promptfix. Scenario environment
variables are cleared explicitly. Do not use the colour-sort project as --project.
The baseline's existing camera frame was taken with a parked arm; live recovery
uses a fresh frame at executor home. The new frame must pass all checks.

Isaac and Cosmos require the existing separate interpreters and GPU installation.
Their concurrent use must be tested for this stacking extension. The model
interpreter path is preserved without resolving its virtual-environment symlink.
Worker library paths are restored to avoid inheriting Kit's Python libraries.
The launcher refuses to start if another detected simulator/runner is active.
Coordinate with teammates; this is not a global machine reservation mechanism.
No shared-memory files are removed or reset by this tool.

The existing 90-second startup watchdog and 30-minute stage limit remain.
Startup timeouts are infrastructure failures, not grasp/recovery results.
Never reuse a populated run directory or hide incomplete attempts. Do not retry
repeatedly merely to obtain a pass. Native tracing is retained as the previously
observed startup workaround, not a proven root-cause fix.

## Transfer and host tests

Download stack_recovery_v1_source.zip into Windows Downloads. In LOCAL PowerShell:

```powershell
scp "$env:USERPROFILE\Downloads\stack_recovery_v1_source.zip" yt-twinworksb921@172.30.102.90:/home/yt-twinworksb921/workspace/fanuc_er4ia_sim/
```

On SPARK, check that stack_recovery_v1 does not already contain an earlier copy;
then extract the ZIP and run tests with the HOST interpreter:

```bash
cd /workspace/fanuc_er4ia_sim
python3 -m zipfile -e stack_recovery_v1_source.zip .
.venv/bin/python stack_recovery_v1/run_stack_pilot.py \
  --project /workspace/fanuc_er4ia_sim --self-test
```

Expected: 34 extension tests (8 image-position, 7 fit/review and 19 stacking).
The uploaded root project also passed its existing 33 tests locally.
Tests do not start Isaac, the model or physical hardware.

## Recommended trial order

1. Zero-shift enabled control: source inputs and original tower must still work.
2. +10 mm X with recovery enabled.
3. +10 mm X with recovery disabled, using the same source and conditions.
4. Repeat matched trials and retain every outcome where time allows.

Start with this read-only compatibility check:

```bash
.venv/bin/python stack_recovery_v1/run_stack_pilot.py \
  --project /workspace/fanuc_er4ia_sim \
  --source runs/baseline_03_promptfix \
  --out runs/stack_recovery_v1/control_enabled_attempt_001 \
  --shift-mm 0 --mode enabled --check
```

After checks pass and no other Isaac job is active, launch the control:

```bash
pgrep -af 'IsaacSim|run_pilot|run_stack_pilot|run_scenario_batch|run_colour_sort_pipeline'
export ISAAC=/workspace/IsaacSim/_build/linux-aarch64/release
export COSMOS=/workspace/cosmos-venv/bin/python
export COSMOS3_NANO_PATH=/workspace/models/Cosmos3-Nano
.venv/bin/python -u stack_recovery_v1/run_stack_pilot.py \
  --project /workspace/fanuc_er4ia_sim \
  --source runs/baseline_03_promptfix \
  --out runs/stack_recovery_v1/control_enabled_attempt_001 \
  --shift-mm 0 --mode enabled --trace-isaac
```

Control success requires six pickups/placements, but does not require a recovery
if the image is consistent. Then use --shift-mm 10 with NEW outputs
runs/stack_recovery_v1/x10_enabled_attempt_001 and
runs/stack_recovery_v1/x10_disabled_attempt_001 respectively. A shifted enabled
PASS additionally requires actual recovery success. A disabled run can complete
or fail; record the measured result. A pre-grasp review exits with code 2 and an
explicit outcome. Startup and other runtime exceptions remain failed attempts.

Inspect execution_report.json, execution_events.json, recovery_status.json,
recovery/position_check.json and recovery/fresh_perception/refinement_audit.json.
Raw model replies and the grounded explanation are retained alongside the fresh
image, calibration and proposed plan. Videos omit model wall-time pauses.

## Videos and submission evidence

After execution, use the installed encoder for both views (including incomplete
runs if actual motion frames exist):

```bash
.venv/bin/python encode_video.py \
  --frames runs/stack_recovery_v1/x10_enabled_attempt_001/frames \
  --out runs/stack_recovery_v1/x10_enabled_attempt_001/stack_table.mp4 --fps 30
.venv/bin/python encode_video.py \
  --frames runs/stack_recovery_v1/x10_enabled_attempt_001/side_frames \
  --out runs/stack_recovery_v1/x10_enabled_attempt_001/stack_side.mp4 --fps 30
```

Compare task completion, pickups, placements, stack support, review requests,
model/decision/session wall times and attempted versus applied recovery. Treat
simulator startup failures separately. One pair is a demonstration, not a
reliability estimate or evidence that an LLM is necessary for improvement.
The comparison tests observation plus hybrid replanning versus a stale plan.

Keep the earlier 20 selected successful stacking configurations and new recovery
trials identified separately. For CSI, this SDK simulator executor is NOT a
physical FANUC driver: controller interface, frames/tool conversion, gripper
commands, camera calibration and supervised motion validation remain required.
The target is a measured physical demonstration where feasible, without claiming
hardware performance from simulation results. A faithful account of limitations
is part of the submission evidence.

## Local validation completed

The existing 33 project tests and 34 extension tests passed. The saved unshifted
image was classified position_consistent (0.0280 mm displacement); guarded
replanning of saved detections accepted six picks with no skips. Baseline IK
preflight covered 1,656 samples with maximum position residual 0.09984 mm.
Five stub-simulator launcher cases passed: control, enabled success, disabled
incomplete execution, pre-grasp review, and missing-report error. These stub
checks do not validate physics, model inference or GPU execution.

This stacking extension has NOT yet been validated live in Isaac or on hardware.
The source package contains no captures, videos, weights or environments.
