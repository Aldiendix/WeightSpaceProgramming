"""A held-out split of the toolkit's generated reasoning puzzles (practice:bbhsynth) for screening: N puzzles, the same
number from every puzzle family (seeded), none in the case's other splits. Written as $WSP_CASES/CASE/sets/NAME.json, so
`wsp.case run` / prun.py can use it as a set and sample.py leaves it out of the training draw.
    python puzzle_split.py CASE --name puzzle_hold --per-family 20"""
import argparse
import collections
import glob
import json
import os
import random
from pathlib import Path

from wsp import case as C, data

ap = argparse.ArgumentParser()
ap.add_argument("case"); ap.add_argument("--name", default="puzzle_hold"); ap.add_argument("--per-family", type=int, default=20)
ap.add_argument("--seed", default="puzzle-hold:7")
a = ap.parse_args()
d = Path(os.environ.get("WSP_CASES", "cases")) / a.case / "sets"
taken = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
for f in glob.glob(str(d / "*.json")):
    if Path(f).stem != a.name:
        taken |= set(json.load(open(f)))
fam = collections.defaultdict(list)
for r in data._practice("bbhsynth"):
    if r["id"] not in taken:
        fam[r["subject"]].append(r["id"])
rng = random.Random(a.seed); ids = []
for f in sorted(fam):
    xs = sorted(fam[f]); rng.shuffle(xs); ids += xs[: a.per_family]
rng.shuffle(ids)
d.mkdir(parents=True, exist_ok=True)
json.dump(ids, open(d / f"{a.name}.json", "w"))
print(f"{a.name}: {len(ids)} puzzles, {len(fam)} families x {a.per_family}; {sum(map(len, fam.values())) - len(ids)} left for training")
