"""Group-relative rows from the model's own sampled answers (sample.py output): for every problem whose samples disagree
(some right, some not), each finished sample becomes a row weighted by its advantage r - mean(r of the other samples)
(cut samples count in the mean but get no row). --span answer: the prompt is the question plus the sample's own
thinking (no loss there), the target is its final answer; --span full: the target is the whole sample.
A few problems are held out (their right samples, weight 1) to pick the fit's best step.
    python group_rows.py CASE OUT.jsonl SAMPLES.*.jsonl [--span full] [--held-frac 0.05]"""
import argparse
import collections
import json
import random

from transformers import AutoTokenizer
from wsp import case as C, data, profiles

ap = argparse.ArgumentParser()
ap.add_argument("case"); ap.add_argument("out"); ap.add_argument("files", nargs="+")
ap.add_argument("--span", choices=["answer", "full"], default="full")
ap.add_argument("--held-frac", type=float, default=0.05)
a = ap.parse_args()
prof = profiles.get(C.conf(a.case)["profile"])
tok = AutoTokenizer.from_pretrained(str(profiles.base_dir(prof["name"])))
items = data.by_id()
by = collections.defaultdict(list)
for f in a.files:
    for line in open(f):
        r = json.loads(line); by[r["id"]].append(r)
rows, stat = [], collections.Counter()
for pid, rs in sorted(by.items()):
    rew = [1.0 if (r["correct"] and not r["cut"]) else 0.0 for r in rs]
    if len(rs) < 2 or min(rew) == max(rew):
        stat["uniform"] += 1; continue
    stat["mixed"] += 1
    prompt0 = tok.apply_chat_template([{"role": "user", "content": data.user_message(items[pid])}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=prof["thinking"])
    for i, r in enumerate(rs):
        t = r["text"]; j = t.find("</think>")
        if r["cut"] or j < 0:
            continue
        adv = rew[i] - (sum(rew) - rew[i]) / (len(rs) - 1)
        if abs(adv) < 1e-9:
            continue
        k = 0 if a.span == "full" else j + len("</think>")
        rows.append({"id": pid, "source": r["source"], "prompt": prompt0 + t[:k], "target": t[k:], "w": adv})
        stat["pos" if adv > 0 else "neg"] += 1
random.Random(0).shuffle(rows)
ids = sorted({r["id"] for r in rows}); random.Random(1).shuffle(ids)
held = set(ids[: max(8, int(a.held_frac * len(ids)))])
with open(a.out, "w") as f:
    for r in rows:
        if r["id"] not in held:
            f.write(json.dumps(r) + "\n")
with open(a.out.replace(".jsonl", ".held.jsonl"), "w") as f:
    for r in rows:
        if r["id"] in held and r["w"] > 0:
            f.write(json.dumps({**r, "w": 1.0}) + "\n")
print(dict(stat), "rows", len(rows), "held-out problems", len(held))
