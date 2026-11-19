"""More of the parent's own answers: sample it on practice problems outside the case's dev and holdout splits
(material for pairs_more.py).

    python more_answers.py CASE base OUT_PREFIX --per-source 256 --n 4 --max-tokens 8192 --shard i/n [--seed 11]

Problems: `per-source` from each practice source of the case, none in its dev / holdout splits. Writes
OUT_PREFIX.<shard>.jsonl (rows like a run file).
"""
import argparse

from wsp import case as C, data, profiles
from wsp.gen import generate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--per-source", type=int, default=256)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--salt", default="x0")
    a = ap.parse_args()
    conf = C.conf(a.case)
    ex = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
    srcs = conf.get("practice", data.PRACTICE)
    items = data.balanced_sample(a.per_source * len(srcs), a.seed, exclude=ex, sources=srcs)
    i, n = map(int, a.shard.split("/"))
    items = [it for j, it in enumerate(sorted(items, key=lambda r: r["id"])) if j % n == i]
    print(f"shard {i}/{n}: {len(items)} problems x {a.n}", flush=True)
    generate(C.model_path(a.case, a.model), items, profiles.get(conf["profile"]), n=a.n, out=f"{a.out}.{i}.jsonl",
             salt=a.salt, max_tokens=a.max_tokens)
    print("GEN_DONE", i, flush=True)


if __name__ == "__main__":
    main()
