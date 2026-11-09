"""Settings from <repo>/.config (KEY=VALUE lines, template: config.example); environment variables override."""
import os
import shlex

from .common import REPO


def load(path=REPO / ".config"):
    conf = {}
    if path.exists():
        for line in path.read_text().splitlines():
            parts = shlex.split(line, comments=True)  # handles quotes and trailing comments like bash
            if parts and "=" in parts[0]:
                k, _, v = parts[0].partition("=")
                conf[k] = v
    conf.update({k: v for k, v in os.environ.items() if k in conf or k.startswith(("WSP_", "R2_"))})
    return conf


def get(key, default=""):
    return load().get(key) or default
