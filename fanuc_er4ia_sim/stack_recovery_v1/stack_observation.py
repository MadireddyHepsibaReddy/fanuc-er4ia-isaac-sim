"""Read live fixed-camera calibration and save pre-grasp image evidence."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import scene_config as C


def live_calibration(stage, path, resolution):
    from pxr import UsdGeom
    prim=stage.GetPrimAtPath(path)
    camera=UsdGeom.Camera(prim)
    xf=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    return dict(camera='table', resolution=list(resolution),
                focal_length_mm=float(camera.GetFocalLengthAttr().Get()),
                horizontal_aperture_mm=float(camera.GetHorizontalApertureAttr().Get()),
                vertical_aperture_mm=float(camera.GetVerticalApertureAttr().Get()),
                camera_to_world=[[float(xf[r][c]) for c in range(4)] for r in range(4)],
                table_top_z=float(C.TABLE_TOP_Z+C.SURFACE_DZ),robot_base_xy=[0.,0.])


def save_observation(folder,index,rgba,calibration,metadata):
    from check_pick_position import validate_camera
    validate_camera(calibration)
    W,H=calibration['resolution']
    a=np.asarray(rgba)
    if a.shape!=(H,W,4) or a.dtype!=np.uint8:
        raise ValueError('Live RGBA camera frame does not match calibration')
    root=Path(folder);path=root/f'before_pick_{index:02d}.png'
    if path.exists():
        raise ValueError('Observation already exists; preserve the run')
    if not cv2.imwrite(str(path),cv2.cvtColor(a[:,:,:3],cv2.COLOR_RGB2BGR)):
        raise ValueError('Could not write live camera evidence')
    record=dict(metadata,camera=calibration,image=path.name,
                image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                observation_only=True,plan_updated=False)
    path.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    return dict(image=str(path),metadata=str(path.with_suffix('.json')))
