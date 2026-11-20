"""Write the dose-scan candidates from the fitted legs (all fits share the parent's singular basis).

  python derive/scan.py --case C --answer A1 A2 --answer-mc AM --hint J [--ka 1.0 1.5] [--kj 0 1.5 2.5]

Candidate S_<ka>_<kj> has z = ka * (mean(z_A1, z_A2) + z_AM / 3) + kj * z_J:
- the answer leg: the mean of the two instruction+multiple-choice seeds, with the multiple-choice-weighted seed added
  at one third of the same dose;
- the hint leg at its own dose kj (kj = 0: answer leg only).
Each candidate is written as cases/C/edits/S_<ka>_<kj>.json/.npz (dots in the name become 'p').
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from svf_rl import write_spec  # noqa: E402
from wsp import case as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True)
ap.add_argument("--answer", nargs="+", required=True)
ap.add_argument("--answer-mc", required=True)
ap.add_argument("--hint", required=True)
ap.add_argument("--ka", nargs="+", type=float, default=[1.0, 1.5])
ap.add_argument("--kj", nargs="+", type=float, default=[0.0, 1.5, 2.5])
ap.add_argument("--min-change", type=float, default=0.002)
a = ap.parse_args()
E = C.cdir(a.case) / "edits"
load = lambda n: np.load(E / f"{n}.raw.npz")
A = [load(n) for n in a.answer]
AM, J = load(a.answer_mc), load(a.hint)
names = [k[2:] for k in A[0].files if k.startswith("U|")]
for r in A[1:] + [AM, J]:
    for n in names:
        assert np.allclose(np.abs((r[f"U|{n}"] * A[0][f"U|{n}"]).sum(0)), 1, atol=1e-3), f"basis differs: {n}"
for ka in a.ka:
    for kj in a.kj:
        comps = {n: (A[0][f"U|{n}"], ka * (np.mean([r[f"z|{n}"] for r in A], 0) + AM[f"z|{n}"] / 3) + kj * J[f"z|{n}"])
                 for n in names}
        name = f"S_{ka:g}_{kj:g}".replace(".", "p")
        write_spec(C.conf(a.case), a.case, name, comps, a.min_change)
