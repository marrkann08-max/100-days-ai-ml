import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "configs" / "default.yaml"


def load(path=DEFAULT, overrides=None):
    """Read a YAML config and apply dotted overrides, e.g. {"model.name": "gru"}."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for key, value in (overrides or {}).items():
        set_path(config, key, value)
    return config


def set_path(config, dotted, value):
    node = config
    *parents, last = dotted.split(".")
    for part in parents:
        node = node.setdefault(part, {})
    node[last] = value


def get_path(config, dotted):
    node = config
    for part in dotted.split("."):
        node = node[part]
    return node


def digest(section):
    """Stable short hash of a config section, used to key cached data."""
    text = json.dumps(section, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def resolve(config, key):
    """Absolute path for a path-valued data setting."""
    path = Path(config["data"][key])
    return path if path.is_absolute() else ROOT / path


def copy_of(config):
    return copy.deepcopy(config)
