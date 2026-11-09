"""Re-derive ../edit.json from the official base model and practice problems (3 forward passes).
Run from the repo root: PYTHONPATH=. python submissions/example/qwen35-0.8b-loop/derive/derive.py <out_dir>"""
import sys
from pathlib import Path

import numpy as np

from wsp import data, profiles
from wsp.directions import extract, make_pairs
from wsp.edits import dir_ablate_spec
from wsp.gen import generate
from wsp.probe import Probe

ENTRY = Path(__file__).resolve().parent.parent
PROFILE, N_PROBLEMS, N_PAIRS, PER_TRACE, TOPK, ALPHA, N_LAYERS = "qwen35-0.8b-think", 64, 48, 6, 64, 6.0, 2

out = Path(sys.argv[1] if len(sys.argv) > 1 else ENTRY / "rederived")
prof = profiles.get(PROFILE)
rows = generate(prof["hf"], data.balanced_sample(N_PROBLEMS, seed=12345), prof, gpu_mem=0.5)  # pass 1
probe = Probe(prof["hf"])
d = extract(probe, make_pairs(rows, probe.tok, prof["thinking"], N_PAIRS, per_trace=PER_TRACE))  # passes 2-3
totals = d["writer"].sum(1)
layers = sorted(np.argsort(-totals[:-1])[:N_LAYERS].tolist())  # largest writers, never the last layer
out.mkdir(parents=True, exist_ok=True)
np.savez(out / "directions.npz", **d)
print(dir_ablate_spec(out / "directions.npz", prof["hf"], layers, TOPK, ALPHA, out, "edit"))
