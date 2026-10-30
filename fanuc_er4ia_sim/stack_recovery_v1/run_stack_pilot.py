"""One-session pre-first-pick stacking recovery in a new run; baseline untouched."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import shutil,subprocess,sys,unittest


def stamp():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',required=True,type=Path)
    ap.add_argument('--source',type=Path)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--mode',choices=['enabled','disabled'],default='enabled')
    ap.add_argument('--shift-mm',type=int,choices=[0,10],default=10)
    ap.add_argument('--isaac',default=os.environ.get('ISAAC'))
    ap.add_argument('--cosmos',default=os.environ.get('COSMOS'))
    ap.add_argument('--model',default=os.environ.get('COSMOS3_NANO_PATH'))
    ap.add_argument('--trace-isaac',action='store_true')
    ap.add_argument('--check',action='store_true');ap.add_argument('--self-test',action='store_true')
    a=ap.parse_args();project=a.project.expanduser().resolve();package=Path(__file__).resolve().parent
    manifest=json.loads((package/'package_manifest.json').read_text())
    for name,digest in manifest.items():
        if sha(package/name)!=digest:raise ValueError('Extension source changed: '+name)
    if not(project/'sort_scene.py').is_file():ap.error('Use the existing fanuc_er4ia_sim stacking project')
    os.environ.pop('TWINWORKS_TASK',None);os.environ.pop('TWINWORKS_SCENARIO',None)
    sys.path.append(str(project))
    import scene_config as C
    if C.SORT_TASK!='cube_stack' or C.SHOW_BINS:ap.error('This pilot requires the default stacking profile without bins')
    if a.self_test:
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in ('test_pick_position','test_revision_v3','test_stack_recovery'))
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if not a.source or not a.out:ap.error('Set --source and --out')
    source=(project/a.source).resolve();out=(project/a.out).resolve()
    if not source.is_relative_to(project/'runs') or not source.is_dir():ap.error('Use a saved run inside the stacking runs folder')
    if not out.is_relative_to(project/'runs') or out.exists() or out.is_relative_to(source):ap.error('Use a NEW separate output run')
    from sort_scene import validate_plan
    from stack_shift import validate_request,adjusted_labels
    from patch_stack_executor import patch_executor
    import numpy as np
    expected=json.loads((package/'project_source_sha256.json').read_text())
    for name,digest in expected.items():
        if sha(project/name)!=digest:raise ValueError('Project source differs from reviewed revision: '+name)
    names=['sort_plan.json','scene_state.json','ground_truth.json','detections.json','world_camera.png','camera.json','execution_report.json']
    inputs={name:(source/name).read_bytes() for name in names}
    plan=json.loads(inputs['sort_plan.json']);state=json.loads(inputs['scene_state.json']);baseline=json.loads(inputs['execution_report.json'])
    validate_plan(plan)
    if plan.get('task')!='cube_stack' or not plan.get('stack_ready') or len(plan['picks'])!=6:ap.error('Source needs an accepted six-cube stack plan')
    if not (baseline.get('task_complete') is True and baseline.get('verified_pickups')==6 and baseline.get('verified_placements')==6):ap.error('Source baseline is not verified complete')
    for i,(x,y,colour) in enumerate(C.SORT_CUBES):
        if np.linalg.norm(np.asarray(state['objects'][f'SortCube{i}']['position'])[:2]-[x,y])>.002:
            ap.error('Source does not match the default cube layout; scenario profiles are not supported by this first pilot')
    delta=[a.shift_mm/1000,0,0];spec=validate_request(state,plan,'SortCube0',delta)
    x,y,z=spec['requested_position_m']
    if not(C.BACKDROP_X[0]<=x<=C.BACKDROP_X[1] and abs(y)<=C.BACKDROP_WIDTH/2):ap.error('Shift leaves the work surface')
    if np.linalg.norm(np.array([x,y])-C.CUBE_STACK_XY)<.025:ap.error('Shift overlaps stack site')
    patched=patch_executor((project/'sort_scene.py').read_bytes(),expected['sort_scene.py'])
    if a.check:
        print(f'Stack pilot compatibility passed: {a.shift_mm} mm world-X; mode={a.mode}.')
        print('No files changed; no simulator or model started.');return 0
    if not a.isaac:ap.error('Set ISAAC')
    isaac=Path(a.isaac).expanduser().resolve()/'python.sh'
    if not os.access(isaac,os.X_OK):ap.error('Isaac launcher missing')
    cosmos=str(Path(a.cosmos).expanduser().absolute()) if a.cosmos else None
    if a.mode=='enabled' and(not cosmos or not os.access(cosmos,os.X_OK) or not a.model or not(Path(a.model)/'config.json').is_file()):ap.error('Set working COSMOS and COSMOS3_NANO_PATH')
    if a.trace_isaac and not shutil.which('strace'):ap.error('strace missing')
    active=subprocess.run(['pgrep','-af','/IsaacSim/|kit/python|kit/kit|run_pilot|run_stack_pilot|run_scenario_batch|run_colour_sort_pipeline'],capture_output=True,text=True)
    if active.returncode not in (0,1):raise RuntimeError('Could not check active processes')
    others=[line for line in active.stdout.splitlines() if int(line.split()[0])!=os.getpid()]
    if others:raise RuntimeError('Another simulation/runner is active; coordinate first:\n'+'\n'.join(others))
    import fcntl
    from session_process import save,launch
    from review_outcome import record_pregrasp_review,PreGraspReview
    import session_process
    with(project/'.stack_recovery_pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out.mkdir(parents=True,exist_ok=False);runtime=out/'runtime';runtime.mkdir()
        revision=out/'source_revision';revision.mkdir()
        for name in expected:shutil.copy2(project/name,revision/name)
        for name,raw in inputs.items():
            destination='baseline_execution_report.json' if name=='execution_report.json' else name
            (out/destination).write_bytes(raw)
        save(out/'ground_truth.json',adjusted_labels(json.loads(inputs['ground_truth.json']),'SortCube0',delta))
        for name in ['initial_sort_plan.json','active_sort_plan.json']:(out/name).write_bytes(inputs['sort_plan.json'])
        runtime_names=['live_recovery.py','cosmos_worker.py','stack_shift.py','check_pick_position.py','session_process.py','guarded_refinement.py','stack_planner.py','stack_preflight.py','stack_observation.py']
        for name in runtime_names:shutil.copy2(package/name,runtime/name)
        (runtime/'sort_scene_stack_live.py').write_text(patched)
        save(out/'live_settings.json',dict(project=str(project),host=str(Path(sys.executable).absolute()),cosmos=cosmos,mode=a.mode,
             target_name='SortCube0',delta_world_m=delta,
             worker_library_environment={key:os.environ.get(key) for key in('PYTHONHOME','PYTHONPATH','LD_LIBRARY_PATH','LD_PRELOAD')}))
        record=dict(status='running',task='cube_stack',revision='stack_recovery_v1',mode=a.mode,shift_mm=a.shift_mm,
             started=stamp(),source_run=str(source),source_sha256=expected,package_sha256=manifest,
             input_sha256={k:hashlib.sha256(v).hexdigest() for k,v in inputs.items()},trace_enabled=a.trace_isaac,
             physical_hardware_validated=False)
        save(out/'session_status.json',record)
        environment=dict(os.environ,TWINWORKS_PROJECT=str(project),PYTHONUNBUFFERED='1')
        environment.pop('COSMOS3_ROOT',None)
        if a.model:environment['COSMOS3_NANO_PATH']=str(Path(a.model).expanduser().resolve())
        session_process.HERE=project
        command=[str(isaac),'-u',str(runtime/'sort_scene_stack_live.py'),'--run',str(out)]
        print(f'START one-session STACK recovery; mode={a.mode}; shift={a.shift_mm} mm world-X',flush=True)
        try:
            result=launch(command,environment,out/'execution.log',isaac=True,trace=a.trace_isaac)
            record['command_result']=result
            if not(out/'execution_report.json').exists():
                review=record_pregrasp_review(out)
                if review is not None:raise PreGraspReview(review['abort_reason'])
                raise RuntimeError('Execution report missing without a pre-grasp review; inspect execution.log')
            report=json.loads((out/'execution_report.json').read_text())
            complete=(report.get('task_complete') is True and report.get('verified_pickups')==6 and report.get('verified_placements')==6)
            if a.mode=='enabled':
                if not complete or(a.shift_mm and report.get('recovery_success') is not True):raise RuntimeError('Stack pilot incomplete; inspect report')
                record['status']='success';print('PASS one-session STACK recovery',flush=True)
            else:
                record['status']='comparison_baseline_complete';print('Comparison baseline completed; task_complete:',report.get('task_complete'),flush=True)
            record['ended']=stamp();save(out/'session_status.json',record)
        except PreGraspReview as error:
            record.update(status='human_review_requested',reason=str(error),ended=stamp());save(out/'session_status.json',record)
            print('REVIEW REQUESTED before grasp:',error,flush=True);print('Results:',out);return 2
        except BaseException as error:
            record.update(status='failed',reason=str(error),ended=stamp());save(out/'session_status.json',record)
            print('STOP; evidence preserved:',out,flush=True);raise
        print('Results:',out,flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
