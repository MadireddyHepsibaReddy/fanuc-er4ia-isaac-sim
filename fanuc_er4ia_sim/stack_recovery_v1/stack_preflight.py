"""Replay executor IK samples without Isaac, physics or ground-truth inputs."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np


def validate_trajectory(plan):
    import scene_config as C
    import fk,ik
    from sort_scene import validate_plan,segments,ease,turn
    validate_plan(plan)
    limits=np.deg2rad([C.JOINT_LIMITS_DEG[f'J{i}'] for i in range(1,7)])
    ik.set_tool_axis(C.TOOL_AXIS);ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
    tool=tuple(np.array(C.TOOL_AXIS)*C.TOOL_LENGTH)
    home=np.array([.30,0,.24]);down=[0,0,-1]
    q=np.deg2rad([C.HOME_POSE_DEG[f'J{i}'] for i in range(1,7)])
    q,ok=ik.solve(q,home,down,limits,tool_offset=tool)
    if not ok:raise ValueError('Preflight home IK failed')
    maximum=0.;count=0
    for original in plan['picks']:
        pick=dict(original)
        r,_=fk.fk(q);current=r@np.array(C.FINGER_SPREAD_AXIS)
        spread=np.array(pick['spread_dir'])
        if np.dot(current,spread)<0:spread=-spread
        pick['spread_dir']=spread.tolist()
        destination=np.array(pick['place_spread_dir'])
        if np.dot(spread,destination)<0:destination=-destination
        pick['place_spread_dir']=destination.tolist()
        for name,p0,p1,duration,j0,j1,direction in segments(pick,home):
            start=None
            if direction is not None and name in ('approach','carry'):
                r,_=fk.fk(q);start=r@np.array(C.FINGER_SPREAD_AXIS)
                start[2]=0;start/=np.linalg.norm(start)
            n=round(duration*30)
            for i in range(n):
                t=ease((i+1)/n);target=p0+(p1-p0)*t
                d=turn(start,direction,t) if start is not None else direction
                q,ok=ik.solve(q,target,down,limits,tool_offset=tool,spread_dir=d)
                if not ok:
                    q,ok=ik.solve(q,target,down,limits,tool_offset=tool,spread_dir=d,iters=500,damping=.02)
                r,actual=fk.tool_pose(q,tool)
                error=float(np.linalg.norm(actual-target))
                aligned=d is None or abs(np.dot(r@np.array(C.FINGER_SPREAD_AXIS),d))>.999
                if not ok or error>.001 or not aligned:
                    raise ValueError('Preflight IK or alignment failed')
                maximum=max(maximum,error);count+=1
    return dict(samples=count,max_ik_position_error_mm=maximum*1000,
                physics_validated=False,collision_free_verified=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',required=True,type=Path);ap.add_argument('--run',required=True,type=Path)
    a=ap.parse_args();sys.path.append(str(a.project.resolve()))
    plan=json.loads((a.run/'sort_plan.json').read_text())
    print(json.dumps(validate_trajectory(plan),indent=2))
if __name__=='__main__':main()
