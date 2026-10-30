"""Run the installed vision planner with an isolated image-fit comparison."""
import argparse
import json
from pathlib import Path
import sys


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',required=True,type=Path)
    ap.add_argument('--run',required=True,type=Path)
    a=ap.parse_args()
    sys.path.append(str(a.project.resolve()))
    import plan_sort
    from guarded_refinement import guarded_refine
    audit=[]
    plan_sort.refine_pose=lambda img,cam,obj,xy,yaw: guarded_refine(img,cam,obj,xy,yaw,audit)
    sys.argv=['plan_sort.py','--run',str(a.run.resolve()),'--task','cube_stack']
    try:
        plan_sort.main()
    finally:
        (a.run/'refinement_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    main()
