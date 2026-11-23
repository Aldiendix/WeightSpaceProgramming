"""The case's sets for this method, written as $WSP_CASES/CASE/sets/NAME.json (each disjoint from the case's other splits):
  puzzle_hold  20 generated puzzles of every family (practice:bbhsynth): the screen, never fitted
  recall_pool  every other puzzle of the rule families: the forced-recall questions
  keepP        96 gsm8k_train + 96 math_train + 128 ifsynth + 64 mmlupro problems: the parent's own answers there become
               keep rows (the edit stays with the parent on them)
  keepB        24 puzzles of every other family (not in puzzle_hold): keep rows too
    python recall_sets.py CASE [--seed 7]"""
import argparse
import collections
import json
import os
import random
from pathlib import Path

from wsp import case as C, data

RULE_FAMILIES = ("hyperbaton", "temporal_sequences", "object_counting")
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("case"); ap.add_argument("--seed", default="7")
a = ap.parse_args()
d = Path(os.environ.get("WSP_CASES", "cases")) / a.case / "sets"; d.mkdir(parents=True, exist_ok=True)
rng = random.Random(f"recall-sets:{a.seed}")
taken = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
fam = collections.defaultdict(list)
for r in data._practice("bbhsynth"):
    fam[r["subject"]].append(r["id"])
hold, pool, keepB = [], [], []
for f in sorted(fam):
    xs = sorted(fam[f]); rng.shuffle(xs)
    hold += xs[:20]
    (pool if f in RULE_FAMILIES else keepB).extend(xs[20:] if f in RULE_FAMILIES else xs[20:44])
keepP = []
for src, n in (("gsm8k_train", 96), ("math_train", 96), ("ifsynth", 128), ("mmlupro", 64)):
    xs = sorted(r["id"] for r in data._practice(src) if r["id"] not in taken); rng.shuffle(xs); keepP += xs[:n]
for name, ids in (("puzzle_hold", hold), ("recall_pool", pool), ("keepP", keepP), ("keepB", keepB)):
    rng.shuffle(ids); json.dump(ids, open(d / f"{name}.json", "w")); print(f"{name}: {len(ids)}")
