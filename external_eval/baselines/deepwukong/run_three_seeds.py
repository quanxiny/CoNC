#!/usr/bin/env python3
"""Launch the fixed three-seed CPU recipe; each seed owns its resumable outputs."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import adapter as a


def main():
    a.OUTPUT.mkdir(parents=True, exist_ok=True)
    with (a.OUTPUT / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.frozen.signed_read(a.OUTPUT / "protocol.json", "result_sha256")
        a.frozen.signed_read(a.OUTPUT / "data/preparation.json", "result_sha256")
        env = {**os.environ, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "4",
               "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
        jobs = []
        for seed in (11, 29, 47):
            log_path = a.OUTPUT / f"s{seed}.log"
            stream = log_path.open("a")
            command = [sys.executable, "-u", str(a.BASE / "train_evaluate.py"), "train",
                       "--seed", str(seed), "--threads", "4"]
            process = subprocess.Popen(command, env=env, cwd=a.WORKSPACE,
                                       stdout=stream, stderr=subprocess.STDOUT)
            stream.close()
            jobs.append((seed, process, command, str(log_path)))
        a.signed(a.OUTPUT / "launch.json", {"controller_pid": os.getpid(), "started_unix": time.time(),
                 "protocol_sha256": a.sha(a.OUTPUT / "protocol.json"),
                 "controller_sha256": a.sha(__file__), "jobs": [
                     {"seed": seed, "pid": proc.pid, "command": cmd, "log": log}
                     for seed, proc, cmd, log in jobs]})
        print(json.dumps({"event": "launched", "controller_pid": os.getpid(),
                          "jobs": [{"seed": s, "pid": p.pid} for s, p, _, _ in jobs]}), flush=True)
        status = {}
        while len(status) < len(jobs):
            for seed, process, _, _ in jobs:
                if seed not in status and process.poll() is not None:
                    code = process.returncode
                    status[seed] = {"exit_code": code, "complete": (a.OUTPUT / "runs" / f"s{seed}" / "complete.json").exists()}
                    print(json.dumps({"event": "seed_exit", "seed": seed, **status[seed]}), flush=True)
            if len(status) < len(jobs):
                time.sleep(10)
        a.signed(a.OUTPUT / "controller_complete.json", {"status": "COMPLETE" if all(
                 r["exit_code"] == 0 and r["complete"] for r in status.values()) else "FAILED", "seeds": status})
        if any(r["exit_code"] != 0 or not r["complete"] for r in status.values()):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
