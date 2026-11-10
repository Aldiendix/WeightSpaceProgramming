"""First edit spec for the loop method: base generations on practice problems -> loop pairs -> direction ->
top-64 writers of the two strongest non-final layers. Run from the repo root:
PYTHONPATH=. python examples/submission-loop/derive/derive.py <out_dir> [profile]"""
import sys
from pathlib import Path

import numpy as np

from wsp import data, profiles
from wsp.directions import extract, make_pairs
from wsp.edits import dir_ablate_spec
from wsp.gen import generate
from wsp.probe import Probe

N_PROBLEMS, N_PAIRS, PER_TRACE, TOPK, ALPHA, N_LAYERS = 64, 48, 6, 64, 6.0, 2

out = Path(sys.argv[1])
name = sys.argv[2] if len(sys.argv) > 2 else "qwen35-0.8b-think"
prof, base = profiles.get(name), profiles.base_dir(name)
rows = generate(base, data.balanced_sample(N_PROBLEMS, seed=12345), prof, gpu_mem=0.5)
probe = Probe(base)
d = extract(probe, make_pairs(rows, probe.tok, prof["thinking"], N_PAIRS, per_trace=PER_TRACE))
layers = sorted(np.argsort(-d["writer"].sum(1)[:-1])[:N_LAYERS].tolist())  # strongest writers, never the last layer
out.mkdir(parents=True, exist_ok=True)
np.savez(out / "directions.npz", **d)
print(dir_ablate_spec(out / "directions.npz", name, layers, TOPK, ALPHA, out, "edit"))
