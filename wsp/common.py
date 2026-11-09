"""Paths and small helpers shared by every module.

Paths come from the environment (set by `./gpu run` on the GPU box; other settings: wsp/config.py, .config):
  WSP_WORK  disposable work dir for big files (models, generations, activations, data cache); default <repo>/.work
Everything worth keeping lives in the repo under cases/.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CASES = REPO / "cases"
WORK = Path(os.environ.get("WSP_WORK", REPO / ".work"))
DATA, RUNS, MODELS = WORK / "data", WORK / "runs", WORK / "models"


def read_jsonl(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def stable_seed(*parts):
    """Seed that depends only on its parts (problem id, sample index, salt): base and edited runs of the same
    problems draw the same random stream, which makes paired comparisons much less noisy."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF
