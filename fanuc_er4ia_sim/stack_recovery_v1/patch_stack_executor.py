"""Build a guarded runtime copy; never edit the baseline executor."""
import hashlib


def patch_executor(raw,expected):
    if hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError('Stacking executor differs from the tested revision')
    text=raw.decode().replace("\r\n", "\n")
    edits=[
      ('import numpy as np\n','import sys\nsys.path.append(os.environ["TWINWORKS_PROJECT"])\n\nimport numpy as np\n'),
      ('    a = ap.parse_args()\n','    a = ap.parse_args()\n    if a.physics_only or a.camera != "table":\n        ap.error("Stack recovery requires the rendered table camera")\n'),
      ('        cam = (\n','        live_resolution = tuple(2 * value for value in C.TABLE_CAM["resolution"])\n        cam = (\n'),
      ('resolution=configs[a.camera]["resolution"],','resolution=live_resolution,'),
      ('        for name in wrappers:\n','        from live_recovery import controlled_shift\n        controlled_shift(run, state, plan, wrappers, world, command, q)\n        for name in wrappers:\n'),
      ('        before = poses()\n','        from stack_observation import live_calibration\n        observation_calibration = live_calibration(stage, cam_paths["table"], live_resolution)\n        from live_recovery import before_first_pick\n        plan = before_first_pick(run, plan, cam, observation_calibration, world, command, q)\n        before = poses()\n'),
      ('rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR',
       'cv2.resize(rgba[:, :, :3].astype(np.uint8), tuple(C.TABLE_CAM["resolution"]), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR'),
      ('        (run / "execution_report.json").write_text(json.dumps(report, indent=2) + "\\n")',
       '        from live_recovery import augment_report\n        report = augment_report(run, report)\n        (run / "execution_report.json").write_text(json.dumps(report, indent=2) + "\\n")'),
    ]
    for old,new in edits:
        if text.count(old)!=1:
            raise ValueError('Expected stack integration point missing or repeated: '+old[:65])
        text=text.replace(old,new)
    compile(text,'sort_scene_stack_live.py','exec')
    return text
