"""Compare fresh measured and refined poses against ONE observed pixel component."""
import cv2
import numpy as np
import scene_config as C
from cube_pose import refine_pose, silhouette, square_yaw


def prefer_measured_pose(measured, refined, bounds_centre, reference_yaw):
    """Choose by image overlap, never by requested shift or simulator truth."""
    xy, yaw, quality = measured
    inside = (np.asarray(xy).shape == (2,) and np.isfinite([*xy, yaw, quality]).all()
              and np.all(np.abs(np.asarray(xy) - bounds_centre) <= .0038)
              and abs(square_yaw(yaw - reference_yaw)) <= 18)
    return bool(inside and quality >= .85 and quality >= refined[2])


def guarded_refine(img, cam, obj, xy, yaw, audit=None):
    refined_xy, refined_yaw, refined_quality = refine_pose(img, cam, obj, xy, yaw)
    measurement = obj.get('image_pose_measurement', {})
    try:
        measured_xy = np.asarray(measurement['position_world_m'], float)[:2]
        measured_yaw = float(measurement['yaw_deg'])
        if (measured_xy.shape != (2,) or not np.isfinite([*measured_xy, measured_yaw]).all()
                or measurement.get('source') != 'calibrated_image_silhouette_fit'):
            raise ValueError('Missing finite fresh image measurement')
    except (KeyError, TypeError, ValueError):
        if audit is not None:
            audit.append(dict(label=obj.get('label'), selection='original_refinement', reason='no valid image measurement'))
        return refined_xy, refined_yaw, refined_quality
    # Reproduce the original planner's region and component selection. Both
    # hypotheses are scored on identical pixels/calibration, not their old scores.
    H, W = img.shape[:2]
    u, v = cam.project([*xy, cam.table_z + C.CUBE_EDGE / 2])
    radius = round(40 * W / 1920)
    x0, y0 = max(0, int(u)-radius), max(0, int(v)-radius)
    crop = img[y0:min(H, int(v)+radius+1), x0:min(W, int(u)+radius+1)]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, value = cv2.split(hsv)
    colour = obj['colour']
    gate = ((h>=12)&(h<=42) if colour=='yellow' else
            (h>=95)&(h<=135) if colour=='blue' else (h<=12)|(h>=170))
    mask = (gate & (s>=45) & (value>=35)).astype(np.uint8)
    n, labels, stats, centres = cv2.connectedComponentsWithStats(mask,8)
    components = [i for i in range(1,n) if stats[i,4]>=40]
    if not components:
        return refined_xy, refined_yaw, refined_quality
    component = min(components, key=lambda i:np.linalg.norm(centres[i]-[u-x0,v-y0]))
    observed = labels==component
    def score(position, angle):
        predicted = silhouette(cam, position, angle, mask.shape, (x0,y0))>0
        return float(np.logical_and(predicted,observed).sum()/max(1,np.logical_or(predicted,observed).sum()))
    measured_quality = score(measured_xy, measured_yaw)
    # A rejected boundary fit stays rejected; do not turn it into a rescue.
    candidate_quality = score(refined_xy, refined_yaw) if refined_quality > 0 else 0.0
    select = bool(refined_quality > 0 and prefer_measured_pose(
        (measured_xy, measured_yaw, measured_quality),
        (refined_xy, refined_yaw, candidate_quality), np.asarray(xy), yaw))
    if audit is not None:
        audit.append(dict(label=obj.get('label'), colour=colour,
                          measured_xy_m=measured_xy.tolist(), measured_yaw_deg=measured_yaw,
                          refined_xy_m=np.asarray(refined_xy).tolist(), refined_yaw_deg=float(refined_yaw),
                          measured_iou=measured_quality, refined_iou=candidate_quality,
                          original_refinement_valid=bool(refined_quality>0),
                          selection='fresh_image_measurement' if select else 'original_refinement'))
    return ((measured_xy.copy(), square_yaw(measured_yaw), measured_quality) if select
            else (refined_xy, refined_yaw, refined_quality))
