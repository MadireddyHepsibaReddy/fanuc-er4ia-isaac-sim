"""One bounded pre-first-pick recovery inside the active Isaac session."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import scene_config as C
from cube_pose import square_yaw
from backproject import CameraModel
from check_pick_position import check_target


class HumanReviewRequired(RuntimeError):
    pass


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def request_human(run, reason):
    write(Path(run) / "human_review_request.json", dict(reason=str(reason),
          action="Stop; inspect the saved images, model replies and validation logs",
          automatically_sent=False, grasp_attempted=False))
    raise HumanReviewRequired(str(reason))


def validate_explanation(answer):
    if (answer.get("observed_change") != "target_position_changed"
            or answer.get("cause") != "unknown" or answer.get("grasp_attempted") is not False
            or answer.get("recommended_action") != "replan"
            or not isinstance(answer.get("explanation"), str) or not answer["explanation"].strip()):
        raise ValueError("AI explanation is missing, unsupported, or requests human review")


def validate_update(original, candidate, observation):
    from sort_scene import validate_plan
    if any(p.get("task") != "cube_stack" or p.get("stack_ready") is not True for p in (original, candidate)):
        raise ValueError("Expected two accepted stack plans")
    validate_plan(original)
    validate_plan(candidate)
    if observation.get("status") != "target_moved" or observation.get("candidate_count") != 1:
        raise ValueError("No unambiguous moved-target image evidence")
    measured = np.asarray(observation.get("measured_xy_m"), float)
    measured_yaw = float(observation.get("image_fit_yaw_deg", float("nan")))
    if measured.shape != (2,) or not np.isfinite(measured).all() or not np.isfinite(measured_yaw):
        raise ValueError("Non-finite current image pose")
    key = lambda pick: pick["stack_level"]
    old = {key(p): p for p in original["picks"]}
    new = {key(p): p for p in candidate["picks"]}
    if set(old) != set(new) or len(new) != 6 or any(old[k]["colour"] != new[k]["colour"] for k in old):
        raise ValueError("Replanning changed stack colour order or lost the six-cube inventory")
    changed = key(original["picks"][0])
    for slot, before in old.items():
        after = new[slot]
        fit = float(after.get("pose_silhouette_iou", 0))
        if not np.isfinite(fit) or fit < .85:
            raise ValueError("Fresh cube geometry does not pass the original perception fit floor")
        if slot != changed and np.linalg.norm(np.array(before["xy"]) - after["xy"]) > .002:
            raise ValueError("Another cube changed; outside this one-target recovery pilot")
    target = new[changed]
    if np.linalg.norm(np.array(target["xy"]) - observation["measured_xy_m"]) > .002:
        raise ValueError("Fresh AI-assisted plan disagrees with moved-target position evidence")
    if abs(square_yaw(target["yaw_deg"] - observation["image_fit_yaw_deg"])) > 3:
        raise ValueError("Fresh target yaw disagrees with the current image geometry")
    delta = float(np.linalg.norm(np.array(target["xy"]) - old[changed]["xy"]))
    if not .005 <= delta <= .025:
        raise ValueError("Revised target is outside the bounded movement range")
    return dict(target_colour=target["colour"], target_stack_level=target["stack_level"],
                old_xy_m=old[changed]["xy"], new_xy_m=target["xy"],
                revised_yaw_deg=target["yaw_deg"], change_mm=delta * 1000)


def controlled_shift(run, state, plan, wrappers, world, command, q):
    """Experiment setup only. No truth values are returned to the recovery controller."""
    from stack_shift import validate_request
    settings = json.loads((Path(run) / "live_settings.json").read_text())
    request = validate_request(state, plan, settings["target_name"], settings["delta_world_m"])
    target = wrappers[request["target_name"]]
    position, quat = target.get_world_pose()
    target.set_world_pose(np.array(position) + request["delta_world_m"], quat)
    # Objects are still kinematic here. Settle the authored edit before release
    # so it does not become a simulated velocity impulse.
    for _ in range(12):
        command(q, C.JAW_OPEN)
        world.step(render=True)
    after, _ = target.get_world_pose()
    write(Path(run) / "disturbance_truth.json", dict(request,
          actual_delta_world_m=(np.array(after) - position).tolist(),
          setup_only=True, detector_received_truth=False))
    if not np.allclose(np.array(after) - position, request["delta_world_m"], atol=1e-6, rtol=0):
        raise ValueError("Controlled experiment shift was not applied as specified")
    print("[recovery] controlled cube shift applied; releasing objects for physical execution", flush=True)


def child(command, cwd, log, timeout=900, library_environment=None):
    """Inherit the simulator process group so parent cancellation stops the worker."""
    started = time.monotonic()
    environment = dict(os.environ)
    for key, value in (library_environment or {}).items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    with Path(log).open("x") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise RuntimeError(f"Worker returned {code}; inspect {log}")
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
    return time.monotonic() - started


def before_first_pick(run, plan, cam, calibration, world, command, q):
    """Called once before any grasp, while the same simulator world stays loaded."""
    run = Path(run)
    settings = json.loads((run / "live_settings.json").read_text())
    root = run / "recovery"
    root.mkdir(exist_ok=False)
    started = time.monotonic()
    record = dict(mode=settings["mode"], scope="one controlled shift before first pick",
                  recovery_attempted=False, recovery_applied=False,
                  same_simulator_session=True, automatic_replanning_limit=1,
                  geometry_position_check=True, new_cosmos_inference=False)
    write(run / "recovery_status.json", record)
    try:
        from stack_observation import save_observation
        for _ in range(6):
            command(q, C.JAW_OPEN)
            world.step(render=True)
        evidence = save_observation(root, 0, cam.get_rgba(), calibration,
                                    dict(detection_id=plan["picks"][0]["detection_id"],
                                         phase="live_before_first_pick", no_grasp_yet=True))
        image = cv2.imread(evidence["image"])
        observation = check_target(image, calibration, plan["picks"][0], {})
        write(root / "position_check.json", observation)
        record["observation_status"] = observation["status"]
        print("[recovery] current image:", observation["status"], flush=True)
        if settings["mode"] == "disabled":
            record["decision"] = "Comparison baseline: retain original plan; existing physical abort gates remain"
            return plan
        if observation["status"] == "position_consistent":
            record["decision"] = "No position change detected; retain original plan"
            return plan
        if observation["status"] != "target_moved":
            request_human(run, "Current target image is uncertain")
        record["recovery_attempted"] = True
        record["replanning_attempts"] = 1
        write(run / "recovery_status.json", record)
        working = root / "fresh_perception"
        working.mkdir()
        # No scene state, object poses or ground-truth file is supplied here.
        (working / "world_camera.png").write_bytes(Path(evidence["image"]).read_bytes())
        write(working / "camera.json", calibration)
        write(working / "change_evidence.json", observation)
        camera = CameraModel(calibration)
        uv = camera.project([*observation["measured_xy_m"], camera.table_z + C.CUBE_EDGE / 2])
        radius = round(35 * camera.W / 1920)
        lo = np.maximum(0, np.array(uv, int) - radius)
        hi = np.minimum([camera.W, camera.H], np.array(uv, int) + radius + 1)
        if not cv2.imwrite(str(working / "changed_target_crop.png"), image[lo[1]:hi[1], lo[0]:hi[0]]):
            raise ValueError("Could not save current target crop")
        runtime = Path(__file__).resolve().parent
        project = settings["project"]
        print("[recovery] pausing simulation steps for fresh Cosmos perception and explanation", flush=True)
        record["cosmos_worker_started"] = True
        write(run / "recovery_status.json", record)
        seconds = child([settings["cosmos"], "-u", str(runtime / "cosmos_worker.py"),
                         "--project", project, "--run", str(working)], project, working / "cosmos.log",
                        library_environment=settings["worker_library_environment"])
        record.update(cosmos_wall_seconds=seconds, new_cosmos_inference=True)
        explanation = json.loads((working / "ai_explanation.json").read_text())
        validate_explanation(explanation)
        child([settings["host"], "-u", str(runtime / "stack_planner.py"),
               "--project", project, "--run", str(working)], project, working / "planning.log", timeout=120,
              library_environment=settings["worker_library_environment"])
        candidate = json.loads((working / "sort_plan.json").read_text())
        update = validate_update(plan, candidate, observation)
        child([settings["host"], "-u", str(runtime / "stack_preflight.py"),
               "--project", project, "--run", str(working)], project, working / "trajectory_preflight.log", timeout=180,
              library_environment=settings["worker_library_environment"])
        write(run / "active_sort_plan.json", candidate)
        record.update(recovery_applied=True, update=update, decision="Use validated fresh vision plan")
        print("[recovery] candidate validated; resuming the SAME simulation with updated grasps", flush=True)
        return candidate
    except HumanReviewRequired:
        record["decision"] = "human_review_requested"
        raise
    except Exception as error:
        record.update(decision="human_review_requested", reason=str(error))
        request_human(run, error)
    finally:
        record["decision_wall_seconds"] = time.monotonic() - started
        write(run / "recovery_status.json", record)


def augment_report(run, report):
    run = Path(run)
    status = json.loads((run / "recovery_status.json").read_text())
    report.update(live_recovery_scope="one pre-first-pick update; not general failure recovery",
                  recovery_attempted=status["recovery_attempted"],
                  recovery_applied=status["recovery_applied"],
                  recovery_success=bool(status["recovery_applied"] and report["task_complete"]),
                  same_simulator_session=True, physical_success_verification="simulator_state_evaluation",
                  recovery_enabled=status["mode"] == "enabled", controller_extra_physics_steps=18,
                  active_plan_file="active_sort_plan.json", recovery_status=status,
                  active_detections_file="recovery/fresh_perception/detections.json" if status["recovery_applied"] else "detections.json",
                  observation_resolution_scale=2, video_resolution_scale=1,
                  human_requests_are_local_records=True)
    if status["mode"] == "enabled" and not report["task_complete"]:
        write(run / "human_review_request.json", dict(reason=report.get("abort_reason") or "Physical execution incomplete",
              action="Inspect execution evidence; no automatic second retry", automatically_sent=False,
              grasp_attempted=True))
    return report
