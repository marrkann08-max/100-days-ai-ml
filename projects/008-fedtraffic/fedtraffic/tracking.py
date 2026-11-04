"""Experiment logging: every run gets a folder that is enough to explain and repeat it.

runs/<time>_<name>/
  config.yaml     the exact settings, after overrides
  run.json        git commit, uncommitted-changes flag, versions, seed, timings, final results
  log.jsonl       one line per epoch or federated round
"""
from datetime import datetime
import json
import platform
import subprocess
import sys
import time

import yaml

from .config import ROOT

RUNS = ROOT / "runs"


def git_state():
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    dirty = git("status", "--porcelain", "--", ".")
    return {"commit": git("rev-parse", "HEAD") or "unknown",
            "dirty": bool(dirty), "changed_files": dirty.count("\n") + 1 if dirty else 0}


def versions():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "torch", "flwr", "ray"):
        module = sys.modules.get(name)
        if module is not None:
            out[name] = getattr(module, "__version__", "?")
    return out


class Run:
    def __init__(self, name, config, group="misc"):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = RUNS / f"{stamp}_{name}"
        suffix = 1
        while self.dir.exists():
            suffix += 1
            self.dir = RUNS / f"{stamp}_{name}_{suffix}"
        self.dir.mkdir(parents=True)
        self.name, self.group, self.config = name, group, config
        self.started = time.time()
        (self.dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        self.meta = {"name": name, "group": group, "started": datetime.now().isoformat(timespec="seconds"),
                     "git": git_state(), "versions": versions(), "seed": config.get("seed")}
        self.save_meta()

    def log(self, **row):
        row["elapsed_s"] = round(time.time() - self.started, 2)
        with open(self.dir / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def finish(self, **results):
        self.meta["duration_s"] = round(time.time() - self.started, 1)
        self.meta["results"] = results
        self.save_meta()
        return results

    def save_meta(self):
        (self.dir / "run.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")


def load_runs(group=None):
    runs = []
    for path in sorted(RUNS.glob("*/run.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if "results" not in meta or (group and meta["group"] != group):
            continue
        meta["dir"] = path.parent
        log = path.parent / "log.jsonl"
        meta["log"] = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
        runs.append(meta)
    return runs
