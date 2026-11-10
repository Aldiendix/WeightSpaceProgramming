"""Behavior directions from matched contrast pairs, and the MLP neurons that write them.

1. pairs     kind "loop" (and other CONTRASTS): from the model's own correct traces, cut at a sentence boundary; the
             "good" branch continues with the real text, the "bad" branch with the failure behavior. Same prefix, so
             the contrast isolates the behavior, not the content.
             kind "correct": several samples of the same problem; the "good" text is the reasoning of a correct
             answer, the "bad" one that of a wrong finished answer (same problem, equal length windows).
2. extract   Per layer l: u_l = normalized mean(resid | bad) - mean(resid | good) over the continuation tokens, with
             split-half stability and separation; writer score of MLP neuron j:
             (mean_bad a_j - mean_good a_j) * (u_l . W_down[:, j]).
3. selectivity  Mean log-prob of good and bad continuations under a model: a good edit lowers "bad" and keeps "good".
"""
import random
import re

import numpy as np

from .common import stable_seed
from .data import by_id, user_message

SENT = re.compile(r"(?<=[.!?。])\s+|\n+")


def _sentences(text, min_chars=25):
    out, pos = [], 0
    for m in SENT.finditer(text):
        if m.start() > pos:
            out.append((pos, m.end()))
        pos = m.end()
    if pos < len(text):
        out.append((pos, len(text)))
    return [s for s in out if s[1] - s[0] > min_chars]


def _loop(think, sents, i, length, key):
    """Repeat a block of the last 1-4 sentences (with their own separators), as real loops cycle over blocks.
    Returns (bad continuation, chars to skip before measuring: the first repetition only re-states the text)."""
    k = min(1 + stable_seed(key, i) % 4, i)
    unit = think[sents[i - k][0]:sents[i][0]]
    return (unit * (length // max(len(unit), 1) + 1))[:length], len(unit)


CONTRASTS = {"loop": _loop}


def _correct_pairs(rows, prompt_of, n, rng, window=6000, min_chars=1000):
    by = {}
    for r in rows:
        if not r["cut"]:
            by.setdefault(r["id"], []).append(r)
    pairs = []
    for pid in rng.sample(sorted(by), len(by)):
        good = [r for r in by[pid] if r["correct"]]
        bad = [r for r in by[pid] if not r["correct"]]
        for g, b in zip(rng.sample(good, len(good)), rng.sample(bad, len(bad))):
            gt, bt = (x["text"].split("</think>")[0] for x in (g, b))
            m = min(len(gt), len(bt), window)
            if m < min_chars:
                continue
            base = prompt_of(pid)
            seg = [len(base), len(base) + m]
            pairs.append({"id": pid, "good": {"text": base + gt[:m], "seg": seg},
                          "bad": {"text": base + bt[:m], "seg": seg}})
    return pairs[:n]


def make_pairs(rows, tok, thinking, n, seed=0, per_trace=1, kind="loop", cont_chars=800, min_measured=120):
    """rows: generation rows. -> list of {id, good, bad} (see the module doc for the kinds)."""
    rng, items, pairs = random.Random(seed), by_id(), []
    if kind == "correct":
        def prompt_of(pid):
            return tok.apply_chat_template([{"role": "user", "content": user_message(items[pid])}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=thinking)
        return _correct_pairs([r for r in rows if r["id"] in items], prompt_of, n, rng)
    cand = [r for r in rows if r["correct"] and not r["cut"] and r["id"] in items]
    rng.shuffle(cand)
    for r in cand:
        think = r["text"].split("</think>")[0]
        sents = _sentences(think)
        if len(sents) < 12:
            continue
        prompt = tok.apply_chat_template([{"role": "user", "content": user_message(items[r["id"]])}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=thinking)
        lo, hi = len(sents) // 5 + 1, (len(sents) * 4) // 5
        for i in sorted(rng.sample(range(lo, hi), min(per_trace, hi - lo))):
            cut = sents[i][0]
            good = think[cut:cut + cont_chars]
            bad, skip = CONTRASTS[kind](think, sents, i, len(good), r["id"])
            if len(good) < cont_chars // 2 or skip > len(good) - min_measured:
                continue
            base = prompt + think[:cut]
            seg = [len(base) + skip, len(base) + len(good)]
            pairs.append({"id": r["id"], "cut": cut, "good": {"text": base + good, "seg": seg},
                          "bad": {"text": base + bad, "seg": seg}})
        if len(pairs) >= n:
            break
    return pairs[:n]


def extract(probe, pairs):
    g = probe.read([q["good"] for q in pairs])
    b = probe.read([q["bad"] for q in pairs])
    D = b["resid"] - g["resid"]  # [N, L, d]
    u = D.mean(0)
    U = u / np.linalg.norm(u, axis=-1, keepdims=True)
    h1, h2 = D[: len(D) // 2].mean(0), D[len(D) // 2:].mean(0)
    pg, pb = (g["resid"] * U).sum(-1), (b["resid"] * U).sum(-1)  # [N, L]
    da = b["mlp"].mean(0) - g["mlp"].mean(0)  # [L, inter]
    writer = np.stack([da[l] * (U[l] @ probe.down_proj(l)) for l in range(len(U))])
    return {"U": U, "writer": writer,
            "stability": (h1 * h2).sum(-1) / (np.linalg.norm(h1, axis=-1) * np.linalg.norm(h2, axis=-1)),
            "separation": (pb.mean(0) - pg.mean(0)) / np.sqrt(0.5 * (pb.var(0) + pg.var(0)) + 1e-9),
            "pair_agreement": (pb > pg).mean(0),
            "relative_norm": np.linalg.norm(u, axis=-1) / np.linalg.norm(g["resid"], axis=-1).mean(0),
            "logp_good": g["logp"], "logp_bad": b["logp"]}


def layer_table(d, topk=64):
    rows = []
    for l, w in enumerate(d["writer"]):
        pos = w[w > 0].sum()
        rows.append({"layer": l, "stability": float(d["stability"][l]), "separation": float(d["separation"][l]),
                     "writer_total": float(w.sum()),
                     f"top{topk}_share": float(np.sort(w)[::-1][:topk].sum() / pos) if pos > 0 else 0.0})
    return rows


def selectivity(probe, pairs):
    g = probe.read([q["good"] for q in pairs], want=("logp",))["logp"]
    b = probe.read([q["bad"] for q in pairs], want=("logp",))["logp"]
    return {"logp_good": float(g.mean()), "logp_bad": float(b.mean())}
