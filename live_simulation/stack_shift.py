"""Controlled scene-edit specification and scoring; not a perception input."""
import copy
import re
import numpy as np


def validate_request(state, plan, target_name, delta):
    """Simulator setup may use authored object identity; the image check must not."""
    if not re.fullmatch(r"SortCube\d+", target_name):
        raise ValueError("Disturbance target must be an explicit cube prim name")
    if plan.get("task") != "cube_stack" or len(plan.get("picks", [])) != 6:
        raise ValueError("Expected an accepted six-pick stacking plan")
    offset = np.asarray(delta, float)
    if offset.shape != (3,) or not np.isfinite(offset).all() or not any(np.allclose(offset, allowed, atol=1e-12, rtol=0) for allowed in allowed_offsets()):
        raise ValueError("Use a zero-shift control or +10 mm world-X shift")
    saved = state["objects"][target_name]
    position, orientation = np.asarray(saved["position"], float), np.asarray(saved["orientation"], float)
    if (position.shape != (3,) or orientation.shape != (4,)
            or not np.isfinite(position).all() or not np.isfinite(orientation).all()
            or not np.isclose(np.linalg.norm(orientation), 1, atol=1e-5)):
        raise ValueError("Invalid captured disturbance target pose")
    expected = np.asarray(plan["picks"][0]["xy"], float)
    if expected.shape != (2,) or not np.isfinite(expected).all() or np.linalg.norm(position[:2] - expected) > .002:
        raise ValueError("Explicit disturbance object is not aligned with the first planned target")
    destination = position + offset
    for name, other in state["objects"].items():
        if name != target_name and name.startswith("SortCube"):
            if np.linalg.norm(destination[:2] - np.asarray(other["position"], float)[:2]) < .025:
                raise ValueError("Disturbed target would be too close to another cube")
    return dict(target_name=target_name, delta_world_m=offset.tolist(),
                original_position_m=position.tolist(), requested_position_m=destination.tolist(),
                original_orientation_wxyz=orientation.tolist(),
                disturbance_type="controlled_kinematic_reposition", physical_push_simulated=False)



def allowed_offsets():
    return [[0,0,0],[.010,0,0]]


def adjusted_labels(truth, target_name, delta):
    """Evaluation labels only; NEVER consumed by perception or planning."""
    result = copy.deepcopy(truth)
    labels = [obj for obj in result["objects"] if obj["name"] == target_name]
    if len(labels) != 1:
        raise ValueError("Evaluation cube label missing or repeated")
    for index in range(3):
        labels[0]["xyz"][index] += delta[index]
    return result
