"""Problem sets for the run, all from shipped practice sets and disjoint from the case's dev and holdout splits.

  python derive/make_sets.py --case C --out /workspace/work/rl [--shards 4]
      -> <out>/train_ids_<i>.json (training shards, mixed sources) and cases/C/sets/ifsfresh.json (a fresh
         instruction-following screen set, never trained on: pass it to svf_rl.py --exclude-sets)
  python derive/make_sets.py --case C --out /workspace/work/rl --hard <out>/gen_*.jsonl [--hard-shards 2]
      -> <out>/hard_ids_<i>.json: ifsynth and mmlupro training problems that the parent did NOT solve in every sample
         (mixed or unsolved), for a second round of sampling that enlarges the mixed groups.
Only `practice:` sources are read; benchmark sets are never touched.
"""
import argparse
import collections
import json
import os
import random

from wsp import case as C
from wsp import data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-source", nargs="+",
                    default=["gsm8k_train=320", "math_train=400", "mmlupro=320", "ifsynth=320"])
    ap.add_argument("--fresh-if", type=int, default=256, help="size of the fresh ifsynth screen set")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--hard", nargs="*", default=None, help="first-round sample files -> hard_ids shards")
    ap.add_argument("--hard-shards", type=int, default=2)
    ap.add_argument("--seed", type=int, default=21)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.hard is not None:
        ok = collections.defaultdict(list)
        for f in a.hard:
            for line in open(f):
                r = json.loads(line)
                if r["id"].startswith(("ifsynth", "mmlupro")):
                    ok[r["id"]].append(bool(r["correct"]) and not r.get("cut"))
        ids = sorted(i for i, v in ok.items() if not all(v))
        random.Random(a.seed).shuffle(ids)
        for s in range(a.hard_shards):
            with open(os.path.join(a.out, f"hard_ids_{s}.json"), "w") as f:
                json.dump(ids[s::a.hard_shards], f)
        print(f"{len(ids)} not-always-solved ifsynth/mmlupro problems in {a.hard_shards} shards -> {a.out}")
        return
    used = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
    rng = random.Random(a.seed)
    ifpool = [r["id"] for r in data.load("practice:ifsynth") if r["id"] not in used]
    rng.shuffle(ifpool)
    fresh = ifpool[: a.fresh_if]
    sets = C.cdir(a.case) / "sets"
    sets.mkdir(parents=True, exist_ok=True)
    with open(sets / "ifsfresh.json", "w") as f:
        json.dump(fresh, f)
    used |= set(fresh)
    ids = []
    for spec in a.per_source:
        src, n = spec.split("=")
        pool = [r["id"] for r in data.load(f"practice:{src}") if r["id"] not in used]
        rng.shuffle(pool)
        ids += pool[: int(n)]
    rng.shuffle(ids)
    for s in range(a.shards):
        with open(os.path.join(a.out, f"train_ids_{s}.json"), "w") as f:
            json.dump(ids[s::a.shards], f)
    print(f"{len(ids)} training problems in {a.shards} shards -> {a.out}; fresh IF screen set {len(fresh)} -> {sets}")


if __name__ == "__main__":
    main()
