#!/usr/bin/env python3
"""Run the 19 remaining TwinWorks scenarios with preserved per-attempt evidence.

Run with the project's HOST .venv Python on Spark, not with Isaac Python.
This runner launches existing capture, perception, planning, and execution code.
It does not implement recovery during motion or change acceptance thresholds.
"""
import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_files(project):
    found = []
    for item in sorted(project.iterdir()):
        if item.is_file() and item.suffix in ('.py', '.sh', '.txt', '.md'):
            found.append(item)
        elif item.name in ('assets', 'docs', 'textures') and item.is_dir():
            found.extend(p for p in sorted(item.rglob('*')) if p.is_file() and not p.is_symlink())
    return found


def active_isaac():
    processes = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().split(b'\0')
            # Inspect executable and script arguments, not matching shell source text.
            decoded = [a.decode(errors='replace') for a in args if a]
            if any(('/IsaacSim/' in a or a.endswith('/kit/kit')) and '\n' not in a for a in decoded):
                processes.append({'pid': int(entry.name), 'args': decoded})
        except (OSError, PermissionError):
            pass
    return processes


def prepare_isaac(batch, allow_reset, events):
    active = active_isaac()
    require(not active, 'Another Isaac process is present; no semaphore changed: ' + str(active))
    semaphore = Path('/dev/shm/sem.carbonite-sharedmemory')
    if allow_reset and semaphore.exists():
        destination = batch / 'startup_lock_backups'
        destination.mkdir(exist_ok=True)
        target = destination / ('sem.carbonite-sharedmemory.' + str(time.time_ns()))
        shutil.move(str(semaphore), str(target))
        events.append({'time': stamp(), 'action': 'moved_idle_semaphore', 'backup': str(target)})


def stop_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def command(args, cwd, env, log, timeout, record):
    record.update(command=[str(a) for a in args], started=stamp(), log=str(log))
    with log.open('wb') as output:
        process = subprocess.Popen([str(a) for a in args], cwd=cwd, env=env,
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            stop_group(process)
            raise
    record.update(ended=stamp(), returncode=code)
    require(code == 0, 'Command returned ' + str(code) + '; inspect ' + str(log))


def summarize(batch):
    records = []
    for folder in sorted(batch.glob('scenario_*')):
        attempts = sorted(folder.glob('attempt_*/status.json'))
        if attempts:
            records.append(read(attempts[-1]))
    columns = ['scenario', 'attempt', 'status', 'failed_stage', 'reason',
               'task_complete', 'verified_pickups', 'verified_placements', 'run_directory']
    with (batch / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    save(batch / 'summary.json', records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default='/workspace/fanuc_er4ia_sim')
    parser.add_argument('--batch', required=True, help='New or existing campaign folder')
    parser.add_argument('--isaac', default='/workspace/IsaacSim/_build/linux-aarch64/release')
    parser.add_argument('--cosmos', default='/workspace/cosmos-venv/bin/python')
    parser.add_argument('--model', default='/workspace/models/Cosmos3-Nano')
    parser.add_argument('--limit', type=int, help='Run only this many unfinished scenarios')
    parser.add_argument('--list', action='store_true', help='Validate selection without launching anything')
    parser.add_argument('--retry-failed', action='store_true', help='New attempts for previous failed scenarios')
    parser.add_argument('--reset-idle-semaphore', action='store_true',
                        help='Use the observed startup workaround only after checking for Isaac processes')
    parser.add_argument('--start-at', help='First scenario filename to include')
    args = parser.parse_args()
    project, batch = Path(args.project).resolve(), Path(args.batch).resolve()
    require(args.limit is None or args.limit > 0, '--limit must be positive')
    selected = [project / 'scenario_configs' / f'scenario_{i:02d}_{variant}.json'
                for i in range(1, 11) for variant in ('position', 'orientation')
                if (i, variant) != (1, 'orientation')]
    if args.start_at:
        names = [p.name for p in selected]
        require(args.start_at in names, 'Unknown starting scenario: ' + args.start_at)
        selected = selected[names.index(args.start_at):]
    for path in selected:
        require(path.is_file(), 'Missing configuration: ' + str(path))
        config = read(path)
        cubes = config.get('cubes', [])
        require(len(cubes) == 6, 'Expected six cubes: ' + str(path))
        require(sorted(c.get('color', '') for c in cubes) == ['blue','blue','red','red','yellow','yellow'],
                'Invalid cube colors: ' + str(path))
        for cube in cubes:
            require(len(cube['position']) == 2 and all(math.isfinite(float(v)) for v in
                    [*cube['position'], cube['yaw_deg']]), 'Invalid coordinates: ' + str(path))
    if args.list:
        print(f'Selected {len(selected)} scenarios.')
        for path in selected:
            print(path.name)
        return 0

    isaac, cosmos = Path(args.isaac).resolve(), Path(args.cosmos).expanduser().absolute()
    host = project / '.venv/bin/python'
    for path in (host, cosmos, isaac / 'python.sh'):
        require(path.is_file() and os.access(path, os.X_OK), 'Missing executable: ' + str(path))
    require((Path(args.model) / 'config.json').is_file(), 'Model config missing')
    batch.mkdir(parents=True, exist_ok=True)
    lock = (project / '.twinworks_batch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    files = source_files(project)
    hashes = {str(p.relative_to(project)): digest(p) for p in files}
    manifest = {'source': hashes, 'configurations': {p.name: digest(p) for p in selected}}
    manifest_path = batch / 'source_manifest.json'
    if manifest_path.exists():
        require(read(manifest_path) == manifest, 'Source/configuration changed; use a new batch folder')
    else:
        require(not any(batch.glob('scenario_*')), 'Existing results without a manifest; use a new folder')
        for path in files:
            target = batch / 'shared_source' / path.relative_to(project)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        save(manifest_path, manifest)
        save(batch / 'campaign.json', {'created': stamp(), 'project': str(project), 'host': str(host),
             'isaac': str(isaac), 'cosmos': str(cosmos), 'model': str(Path(args.model).resolve()),
             'model_config_sha256': digest(Path(args.model) / 'config.json'),
             'selected': [p.name for p in selected],
             'excluded_completed_run': str(project / 'runs/scenario_01_orientation_02'),
             'limits_seconds': {'capture': 600, 'perception': 1800, 'planning': 120,
                                'execution': 1200, 'encoding': 600},
             'reset_idle_semaphore': args.reset_idle_semaphore})
        for interpreter, name in ((host, 'host'), (cosmos, 'cosmos')):
            result = subprocess.run([str(interpreter), '-m', 'pip', 'freeze'],
                                    capture_output=True, text=True, timeout=60, check=True)
            (batch / (name + '_packages.txt')).write_text(result.stdout)

    env = os.environ.copy()
    env.update(OMNI_KIT_ALLOW_ROOT='1', PYTHONUNBUFFERED='1',
               LD_PRELOAD='/lib/aarch64-linux-gnu/libgomp.so.1', COSMOS3_NANO_PATH=str(Path(args.model).resolve()))
    env.pop('COSMOS3_ROOT', None)
    executed = 0
    for config in selected:
        folder = batch / config.stem
        attempts = sorted(folder.glob('attempt_*/status.json'))
        successful = [
            path for path in attempts
            if read(path).get('status') == 'success'
        ]
        if successful:
            print('SKIP', config.stem, 'already successful',
                  successful[-1].parent.name, flush=True)
            continue
        if attempts:
            previous = read(attempts[-1])
            if previous['status'] == 'success' or (previous['status'] == 'failed' and not args.retry_failed):
                print('SKIP', config.stem, previous['status'], flush=True)
                continue
        if args.limit is not None and executed >= args.limit:
            break
        require(shutil.disk_usage(batch).free >= 8 * 1024**3, 'Less than 8 GiB free; stopping campaign')
        number = len(list(folder.glob('attempt_*'))) + 1
        run = folder / f'attempt_{number:03d}'
        run.mkdir(parents=True, exist_ok=False)
        shutil.copy2(config, run / 'scenario_config.json')
        env['TWINWORKS_SCENARIO'] = str(config)
        state = {'scenario': config.stem, 'attempt': number, 'status': 'running',
                 'started': stamp(), 'run_directory': str(run), 'stages': {}, 'startup_events': []}
        save(run / 'status.json', state)
        print('START', config.stem, run, flush=True)
        stage = 'capture'
        try:
            stages = [('capture', [isaac/'python.sh', '-u', project/'capture_scene.py', '--out', run], 600),
                      ('perception', [cosmos, '-u', project/'perceive.py', '--run', run, '--stack-cubes'], 1800),
                      ('planning', [host, project/'plan_sort.py', '--run', run], 120),
                      ('execution', [isaac/'python.sh', '-u', project/'sort_scene.py', '--run', run], 1200),
                      ('table_video', [host, project/'encode_video.py', '--frames', run/'frames',
                                       '--out', run/'sort_table.mp4', '--fps', '30'], 600),
                      ('side_video', [host, project/'encode_video.py', '--frames', run/'side_frames',
                                      '--out', run/'sort_side.mp4', '--fps', '30'], 600)]
            for stage, cmd, timeout in stages:
                print(' ', config.stem, stage, flush=True)
                if stage in ('capture', 'execution'):
                    prepare_isaac(batch, args.reset_idle_semaphore, state['startup_events'])
                record = state['stages'].setdefault(stage, {})
                save(run / 'status.json', state)
                command(cmd, project, env, run / (stage + '.log'), timeout, record)
                if stage == 'capture':
                    for name in ('world_camera.png','camera.json','ground_truth.json','scene_state.json'):
                        require((run/name).is_file() and (run/name).stat().st_size > 0, 'Missing capture output: '+name)
                    require((run/'world_camera.png').read_bytes()[:8] == b'\x89PNG\r\n\x1a\n', 'Invalid image header')
                    for name in ('camera.json','ground_truth.json','scene_state.json'):
                        read(run/name)
                elif stage == 'perception':
                    require(isinstance(read(run/'detections.json').get('objects'), list), 'Invalid detections')
                elif stage == 'planning':
                    require(len(read(run/'sort_plan.json').get('picks', [])) == 6, 'Planner did not accept exactly six picks')
                elif stage == 'execution':
                    report = read(run/'execution_report.json')
                    for key in ('task_complete','verified_pickups','verified_placements','abort_reason'):
                        state[key] = report.get(key)
                    require(report.get('task_complete') is True and report.get('verified_pickups') == 6
                            and report.get('verified_placements') == 6, 'Execution report did not verify all six layers')
                elif stage.endswith('_video'):
                    name = 'sort_table.mp4' if stage == 'table_video' else 'sort_side.mp4'
                    require((run/name).is_file() and (run/name).stat().st_size > 0, 'Missing video: '+name)
                save(run/'status.json', state)
            state['status'] = 'success'
        except KeyboardInterrupt:
            state.update(status='interrupted', failed_stage=stage, reason='Interrupted by operator', ended=stamp())
            save(run/'status.json', state)
            summarize(batch)
            print('Stopped; evidence preserved in', run, flush=True)
            return 130
        except Exception as exc:
            state.update(status='failed', failed_stage=stage, reason=str(exc) or type(exc).__name__)
        state['ended'] = stamp()
        save(run/'status.json', state)
        summarize(batch)
        print(state['status'].upper(), config.stem, state.get('reason',''), flush=True)
        executed += 1
    print('Results:', batch, flush=True)
    return 2 if any(r['status'] == 'failed' for r in read(batch/'summary.json')) else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('STOP:', str(exc) or type(exc).__name__, file=sys.stderr)
        sys.exit(1)
