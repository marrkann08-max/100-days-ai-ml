"""Start a real Flower federation on this machine and run the app on it.

    .venv\\Scripts\\python deploy\\run_local.py

1. a SuperLink (the server side: Fleet API on :9092, Control API on :9093)
2. one SuperNode per district, each a separate process connecting over gRPC
   with its own partition-id
3. `flwr run` submits the app in deploy/ to the SuperLink and streams its logs

Everything is torn down at the end. Flower's state and per-process logs go to
.flwr/ and deploy/logs/ (both ignored by git). TLS is off (--insecure) because
every process is on localhost; across real machines you'd pass certificates.
"""
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(sys.executable).parent
LOGS = ROOT / "deploy" / "logs"
CLIENTS = 4


def wait_for_port(port, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(.5)
    raise TimeoutError(f"nothing listening on port {port}")


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    home = ROOT / ".flwr"
    home.mkdir(exist_ok=True)
    # A project-local Flower config, so nothing is written to the user's home folder.
    (home / "config.toml").write_text(
        '[superlink]\ndefault = "lab"\n\n[superlink.lab]\naddress = "127.0.0.1:9093"\ninsecure = true\n', encoding="utf-8")
    env = dict(os.environ, FLWR_HOME=str(home),
               PYTHONPATH=os.pathsep.join(filter(None, (str(ROOT), str(ROOT / "deploy"), os.environ.get("PYTHONPATH")))))

    processes = []

    def start(name, args):
        log = open(LOGS / f"{name}.log", "w", encoding="utf-8")
        processes.append((subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log))

    try:
        start("superlink", [str(SCRIPTS / "flower-superlink"), "--insecure", "--database", str(home / "state.db")])
        wait_for_port(9092)
        wait_for_port(9093)
        for k in range(CLIENTS):
            start(f"supernode-{k}", [str(SCRIPTS / "flower-supernode"), "--insecure", "--superlink", "127.0.0.1:9092",
                                     "--node-config", f"partition-id={k} num-partitions={CLIENTS}",
                                     "--port", str(9110 + k)])
        time.sleep(5)
        print(f"SuperLink and {CLIENTS} SuperNodes running; submitting the app...", flush=True)
        run = subprocess.run([str(SCRIPTS / "flwr"), "run", str(ROOT / "deploy"), "lab", "--stream"],
                             cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        (LOGS / "flwr-run.log").write_text(run.stdout + run.stderr, encoding="utf-8")
        print("\n".join(l for l in (run.stdout + run.stderr).splitlines()
                        if "aggregate" in l or "Received" in l or "ERROR" in l or "finished" in l)[-3000:])
        return run.returncode
    finally:
        for proc, log in reversed(processes):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()


if __name__ == "__main__":
    sys.exit(main())
