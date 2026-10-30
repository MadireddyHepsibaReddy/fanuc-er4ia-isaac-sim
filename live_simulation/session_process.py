from pathlib import Path
import os, json, shutil, signal, subprocess, time
HERE = Path(__file__).resolve().parent

def save(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)

def stop(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()

def launch(command, env, log, *, isaac=False, trace=False):
    if log.exists():
        raise ValueError(f"Log already exists: {log}. Preserve this run and use a fresh run for retries.")
    if isaac and trace:
        executable = shutil.which("strace")
        if not executable:
            raise ValueError("strace is required for --trace-isaac")
        command = [executable, "-f", "-tt", "-T", "-s", "100", "-e",
                   "trace=futex,openat,mmap,flock,fcntl,connect",
                   "-o", str(log.with_suffix(".native_trace.log")), *command]
    child_env = dict(env)
    if isaac:
        child_env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        if os.geteuid() == 0:
            child_env["OMNI_KIT_ALLOW_ROOT"] = "1"
        if os.uname().machine == "aarch64":
            libraries = child_env.get("LD_PRELOAD", "").split()
            if not any("libgomp" in value for value in libraries):
                child_env["LD_PRELOAD"] = " ".join([*libraries, "libgomp.so.1"])
    started = time.monotonic()
    with log.open("x") as output:
        process = subprocess.Popen(command, cwd=HERE, env=child_env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            startup_seen = False
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if isaac and not startup_seen:
                    # A bounded startup watchdog avoids the previous 20-minute blind wait.
                    with log.open() as stream:
                        head = stream.read(256 * 1024)
                    startup_seen = "Simulation App Startup Complete" in head
                    if not startup_seen and elapsed > 90:
                        raise TimeoutError(f"Isaac startup exceeded 90 seconds; inspect {log}")
                if elapsed > (1800 if isaac else 1800):
                    raise TimeoutError(f"Stage exceeded time limit; inspect {log}")
                time.sleep(.5)
            if process.returncode:
                raise RuntimeError(f"Process returned {process.returncode}; inspect {log}")
        except BaseException:
            stop(process)
            raise
    return dict(seconds=round(time.monotonic() - started, 3), log=log.name,
                trace_enabled=bool(isaac and trace), interpreter_command=command)
