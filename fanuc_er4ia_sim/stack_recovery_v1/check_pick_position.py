"""Offline image-derived target-position diagnostic. Never updates a grasp.

Uses the original Cosmos-confirmed plan, current camera pixels and calibration.
No object ground truth or execution-report object positions enter this check.
Not a new Cosmos call, identity tracker, yaw gate or live recovery controller.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import scene_config as C
from backproject import CameraModel
from cube_pose import refine_pose

SEARCH_RADIUS_M = .025
POSITION_CONSISTENT_M = .002
MOVED_M = .005
MIN_SILHOUETTE_IOU = .85  # Same image-geometry acceptance floor as initial perception.


def validate_camera(record):
    width, height = record["resolution"]
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError("Invalid camera resolution")
    matrix = np.asarray(record["camera_to_world"], float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-5)
            or not np.allclose(matrix[:3, 3], 0) or not np.isclose(matrix[3, 3], 1)):
        raise ValueError("Invalid camera transform")
    values = [record[k] for k in ("focal_length_mm", "horizontal_aperture_mm", "vertical_aperture_mm")]
    if not np.isfinite([*values, record["table_top_z"]]).all() or min(values) <= 0:
        raise ValueError("Invalid camera intrinsics")
    if record.get("camera") != "table":
        raise ValueError("This diagnostic requires the fixed table camera")


def in_fixture(xy, bins):
    return any(np.all(np.abs(np.asarray(xy) - cfg["xy"]) <=
                         np.asarray(cfg["outer"]) / 2 + .003) for cfg in bins.values())


def check_target(image, camera, pick, bins):
    """Return position evidence or uncertainty; never treat missing pixels as unchanged."""
    validate_camera(camera)
    cam = CameraModel(camera)
    if image is None or image.shape != (cam.H, cam.W, 3) or image.dtype != np.uint8:
        raise ValueError("Image does not match camera resolution/type")
    xy = np.asarray(pick["xy"], float)
    colour = pick["colour"]
    confidence = float(pick.get("pose_confidence", 0))
    if (xy.shape != (2,) or not np.isfinite(xy).all() or colour not in ("yellow", "blue", "red")
            or not np.isfinite(confidence) or not .8 <= confidence <= 1
            or pick.get("eligible") is not True):
        raise ValueError("Invalid or unaccepted planned target")
    result = dict(detection_id=pick["detection_id"], colour=colour, expected_xy_m=xy.tolist(),
                  status="uncertain", reason="No unambiguous cube-shaped colour component in search area",
                  method="image_colour_components_and_calibrated_cube_silhouette",
                  semantics_source="original_Cosmos_confirmed_plan",
                  new_cosmos_inference=False, plan_updated=False, yaw_check_implemented=False,
                  thresholds=dict(search_radius_m=SEARCH_RADIUS_M, consistent_m=POSITION_CONSISTENT_M,
                                  moved_m=MOVED_M, silhouette_iou=MIN_SILHOUETTE_IOU))
    uv = cam.project([*xy, cam.table_z + C.CUBE_EDGE / 2])
    if uv is None or not (0 <= uv[0] < cam.W and 0 <= uv[1] < cam.H):
        result["reason"] = "Planned target is outside camera view"
        return result
    corners = [cam.project([*(xy + [a, b]), cam.table_z + C.CUBE_EDGE / 2])
               for a in (-SEARCH_RADIUS_M, SEARCH_RADIUS_M) for b in (-SEARCH_RADIUS_M, SEARCH_RADIUS_M)]
    if any(p is None for p in corners):
        result["reason"] = "Search region projects behind camera"
        return result
    lo = np.maximum(0, np.floor(np.min(corners, axis=0) - 16).astype(int))
    hi = np.minimum([cam.W, cam.H], np.ceil(np.max(corners, axis=0) + 16).astype(int))
    crop = image[lo[1]:hi[1], lo[0]:hi[0]]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    gate = ((h >= 12) & (h <= 42) if colour == "yellow" else
            (h >= 95) & (h <= 135) if colour == "blue" else (h <= 12) | (h >= 170))
    mask = (gate & (s >= 45) & (v >= 35)).astype(np.uint8)
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for component in range(1, n):
        if stats[component, 4] < 40:
            continue
        point = cam.to_plane(*(centers[component] + lo), z=cam.table_z + C.CUBE_EDGE / 2)
        if point is None or np.linalg.norm(point[:2] - xy) > SEARCH_RADIUS_M or in_fixture(point[:2], bins):
            continue
        # Isolate this component so another same-colour object cannot silently
        # become the fit target inside refine_pose's crop.
        isolated = np.zeros_like(image)
        isolated_crop = isolated[lo[1]:hi[1], lo[0]:hi[0]]
        isolated_crop[labels == component] = crop[labels == component]
        measured, yaw, quality = refine_pose(isolated, cam, {"colour": colour}, point[:2], 0,
                                             search_half_width=45)
        if quality < MIN_SILHOUETTE_IOU or in_fixture(measured, bins):
            continue
        delta = float(np.linalg.norm(measured - xy))
        if delta <= SEARCH_RADIUS_M:
            candidates.append(dict(measured_xy_m=np.asarray(measured).tolist(), displacement_mm=delta * 1000,
                                   image_fit_yaw_deg=float(yaw), silhouette_iou=float(quality)))
    result["candidate_count"] = len(candidates)
    if len(candidates) != 1:
        if len(candidates) > 1:
            result["reason"] = "Multiple plausible same-colour cubes; identity is ambiguous"
        result["suggested_next_step"] = "Obtain a clearer observation or request human review"
        return result
    result.update(candidates[0])
    delta = result["displacement_mm"] / 1000
    if delta <= POSITION_CONSISTENT_M:
        result.update(status="position_consistent", reason="Image-derived centre is within diagnostic tolerance",
                      suggested_next_step="Position evidence only; existing execution gates remain required")
    elif delta >= MOVED_M:
        result.update(status="target_moved", reason="Image-derived centre differs substantially from planned centre",
                      suggested_next_step="Stop for semantic confirmation and validated replanning; do not reuse stale grasp")
    else:
        result.update(reason="Displacement falls in the diagnostic uncertainty band",
                      suggested_next_step="Obtain a clearer observation or request human review")
    return result

