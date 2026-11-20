"""Fit gains on the model's own successful answers (rejection-sampled self-distillation), written as scale_cols ops.

  python -m wsp.fit run <case> <name> --runs <model>__<set> [...] [--layers 4-21] [--modules down_proj o_proj out_proj]
        [--per-problem 2] [--max-len 12000] [--lr 2e-3] [--epochs 1] [--frac 1.0] [--seed 0]
  python -m wsp.fit spec <case> <name> --fits F1 [F2 ...] [--dose 1.0] [--levels 256] [--shuffle]

Use it when the model can already solve many problems it fails (pass@k well above accuracy, many correct answers
among those that finish): training on its own finished, correct answers moves it toward them.
1. Data: the case's runs on practice problems (`wsp.case run <case> <model> <set>`, e.g. on a split made with
   `wsp.case split`; any model of the case, so a second round can use the fitted edit's answers; a run's prompt
   overrides, hint or think, are rebuilt). Kept: answers that finished and are correct, at most --per-problem per
   problem, at most --max-len tokens with the prompt. Problems of the case's other splits (dev, holdout, ...) are
   never used: they select and confirm.
2. Parameters: one gain per input column group of the chosen linear layers in --layers, starting at 1 on the case's
   starting model: for an MLP's output projection one gain per unit (column), for an attention output projection one
   per head (its block of columns), else one per column. A gain on input columns is exactly `scale_cols` on that
   matrix. The layers found are printed per name: check that the MLP is among them (--modules names it).
3. Loss: next-token loss on the answer tokens. AdamW, warm-up, linear decay, gradient checkpointing; one GPU.
4. Output: raw gains in $WSP_WORK/cases/<case>/fits/<name>.npz and a spec cases/<case>/edits/<name>.json at dose 1.
   `spec` averages several fits (different --seed / --frac: averaging cancels fit noise) in log space, scales the
   change by --dose (a dose series, as for any edit), and with --shuffle permutes each tensor's gains over its
   groups: the control with the same values on random units. Gains are quantised to --levels factors per tensor (one
   scale_cols op per level; the payload is only the index lists, whatever the number of levels), and the
   quantisation error is printed.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np

from . import case as C
from . import data, profiles
from .common import read_jsonl, write_json
from .edits import decoder_prefix, spec_size, tensor_index

DEFAULT_MODULES = ("down_proj", "w2", "output_linear", "o_proj", "out_proj")  # MLP outputs, then attention outputs


def _layers(s, n):
    if not s:
        return list(range(n))
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def _width(rel, mod, cfg):
    """Columns per gain: one head's block for an attention output projection, else 1 (an MLP unit, a column)."""
    if "attn" not in rel and "attention" not in rel:
        return 1
    w = (getattr(cfg, "linear_value_head_dim", None) if "linear" in rel else None) or getattr(cfg, "head_dim", None) \
        or cfg.hidden_size // cfg.num_attention_heads
    return w if w and mod.in_features % w == 0 else 1


def targets(model, layers, modules):
    """[(tensor name relative to the decoder prefix, module, columns per gain)] for every linear layer named in
    `modules` inside the given decoder layers."""
    import torch  # noqa: PLC0415
    from .probe import decoder  # noqa: PLC0415

    _, stack = decoder(model)
    cfg = model.config.get_text_config()
    out = []
    for l in layers:
        for rel, m in stack[l].named_modules():
            if isinstance(m, torch.nn.Linear) and rel.split(".")[-1] in modules:
                out.append((f"layers.{l}.{rel}.weight", m, _width(rel, m, cfg)))
    if not out:
        raise SystemExit(f"no linear layer named {', '.join(modules)} in layers {layers[0]}-{layers[-1]}")
    kinds = {}
    for t, _, w in out:
        kinds.setdefault((t.split(".", 2)[2].removesuffix(".weight"), w), []).append(t)
    for (rel, w), ts in sorted(kinds.items()):
        print(f"[fit] gains on {rel}: {len(ts)} layers, {w} column(s) per gain", flush=True)
    return out


class Gains:
    """Multiplicative gains on the inputs of the target layers (forward pre-hooks)."""

    def __init__(self, tgts, device):
        import torch  # noqa: PLC0415
        self.params, self.hooks = {}, []
        for name, m, w in tgts:
            g = torch.nn.Parameter(torch.ones(m.in_features // w, device=device))
            self.params[name] = (g, w)
            self.hooks.append(m.register_forward_pre_hook(
                lambda _, x, g=g, w=w: (x[0] * g.repeat_interleave(w).to(x[0].dtype),)))

    def values(self):
        return {n: g.detach().float().cpu().numpy() for n, (g, _) in self.params.items()}

    def remove(self):
        for h in self.hooks:
            h.remove()


def _turn_end(tok):
    """The text a generated answer ends on (the chat template's end of turn, else EOS)."""
    mark = "Zq7Xw"
    s = tok.apply_chat_template([{"role": "user", "content": "u"}, {"role": "assistant", "content": mark}], tokenize=False)
    after = tok(s[s.rfind(mark) + len(mark):], add_special_tokens=False).input_ids
    special = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}) or {})
    return tok.decode([after[0]]) if after and after[0] in special else (tok.eos_token or "")


def examples(case, runs, tok, prof, per_problem, max_len, frac, seed):
    """Tokenised (prompt, answer) of the kept answers -> ([{"id", "ids", "start", "source"}], counts)."""
    idx = data.by_id()
    fit_sets = {t.split("__", 1)[1] for t in runs}
    held = {i for p in (C.cdir(case) / "sets").glob("*.json") if p.stem not in fit_sets for i in C.set_ids(case, p.stem)}
    by, counts = {}, {"answers": 0, "held_out": 0, "not_practice": 0, "kept_problems": 0, "too_long": 0}
    prompts = {}
    for tag in runs:
        _, over = C.split_model(tag.split("__", 1)[0])
        hint, think = over.get("hint"), over.get("think")
        p = C.RUNS / case / f"{tag}.jsonl"
        if not p.exists():
            raise SystemExit(f"no run {tag}; runs of this case: {', '.join(sorted(q.stem for q in p.parent.glob('*.jsonl')))}")
        for r in read_jsonl(p):
            counts["answers"] += 1
            if r["id"] in held:
                counts["held_out"] += 1
            elif r["id"] not in idx:
                counts["not_practice"] += 1
            elif r["correct"] and not r["cut"]:
                by.setdefault(r["id"], []).append(r)
                prompts[(r["id"], r["k"], tag)] = (hint, prof["thinking"] if think is None else bool(think))
                r["_tag"] = tag
    rng = random.Random(seed)
    pids = sorted(by)
    pids = rng.sample(pids, max(1, round(frac * len(pids)))) if frac < 1 else pids
    end = _turn_end(tok)
    ex = []
    for pid in pids:
        for r in rng.sample(by[pid], min(per_problem, len(by[pid]))):
            hint, thinking = prompts[(pid, r["k"], r["_tag"])]
            prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(idx[pid], hint)}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
            p = tok(prompt).input_ids  # as vLLM tokenises a text prompt
            ids = p + tok(r["text"] + end, add_special_tokens=False).input_ids
            if len(ids) > max_len:
                counts["too_long"] += 1
                continue
            ex.append({"id": pid, "ids": ids, "start": len(p), "source": r.get("source")})
    counts["kept_problems"], counts["examples"] = len({e["id"] for e in ex}), len(ex)
    rng.shuffle(ex)
    return ex, counts


def _loss(model, e, chunk=1024):
    """Mean next-token loss on the answer tokens, logits computed in chunks."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415
    from torch.utils.checkpoint import checkpoint  # noqa: PLC0415
    from .probe import decoder  # noqa: PLC0415

    owner, _ = decoder(model)
    cfg = model.config.get_text_config()
    head, cap, scale = model.get_output_embeddings(), getattr(cfg, "final_logit_softcapping", None), \
        getattr(cfg, "logits_scaling", None)
    dev = head.weight.device
    ids = torch.tensor([e["ids"]], device=dev)
    pos = torch.arange(e["start"] - 1, len(e["ids"]) - 1, device=dev)  # positions predicting the answer tokens
    tgt = ids[0, pos + 1]

    def logp(h):
        z = head(h).float()
        if scale:  # Granite divides its logits
            z = z / scale
        if cap:
            z = cap * torch.tanh(z / cap)
        return F.log_softmax(z, -1)

    def nll(hc, tc):
        return -logp(hc).gather(1, tc[:, None]).sum()
    h = owner(input_ids=ids).last_hidden_state[0, pos]
    total = h.new_zeros((), dtype=torch.float32)
    for a in range(0, len(pos), chunk):
        total = total + checkpoint(nll, h[a:a + chunk], tgt[a:a + chunk], use_reentrant=False)
    return total / len(pos)


def train(model, ex, tgts, lr=2e-3, epochs=1.0, accum=8, device="cuda", log_every=10):
    """Fit the gains on the examples -> (gains {tensor: array}, loss log)."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False
    if hasattr(model, "activation_checkpointing_enable"):  # transformers >= 5.x
        model.activation_checkpointing_enable(activation_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()  # checkpointing is active only in training mode; the models' dropout is 0
    gains = Gains(tgts, device)
    try:
        return _train(model, ex, gains, lr, epochs, accum, log_every)
    finally:
        gains.remove()
        model.eval()


def _train(model, ex, gains, lr, epochs, accum, log_every):
    import torch  # noqa: PLC0415

    opt = torch.optim.AdamW([g for g, _ in gains.params.values()], lr=lr, weight_decay=0.0)
    order = [ex[i % len(ex)] for i in range(max(1, int(len(ex) * epochs)))]
    steps = math.ceil(len(order) / accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5) * max(0.1, 1 - s / steps))
    log, run, t0 = [], 0.0, time.time()
    for n, e in enumerate(order):
        lo = _loss(model, e)
        (lo / accum).backward()
        run += lo.item()
        if (n + 1) % accum == 0 or n + 1 == len(order):
            opt.step()
            sched.step()
            opt.zero_grad()
            k = n % accum + 1
            log.append({"step": len(log) + 1, "loss": run / k, "s": round(time.time() - t0)})
            run = 0.0
            if len(log) % log_every == 0 or n + 1 == len(order):
                print(json.dumps(log[-1]), flush=True)
    return gains.values(), log


def _levels(col, moved, levels):
    """Groups of the moved columns sharing one factor: each distinct value its own group when there are at most
    `levels` of them (exact), else a least-squares grouping into `levels` value ranges (1-D k-means)."""
    v = col[moved]
    vals = np.unique(v)
    if len(vals) <= levels:
        return [moved[v == x] for x in vals]
    half = levels // 2  # start from quantiles and from an even grid: the rare strongly moved units keep their value
    c = np.unique(np.concatenate([np.quantile(v, (np.arange(half) + 0.5) / half),
                                  np.linspace(v.min(), v.max(), levels - half)]))
    for _ in range(100):
        lab = np.searchsorted((c[1:] + c[:-1]) / 2, v)
        new = np.array([v[lab == j].mean() for j in range(len(c)) if (lab == j).any()])
        if len(new) == len(c) and np.allclose(new, c):
            break
        c = new
    lab = np.searchsorted((c[1:] + c[:-1]) / 2, v)
    return [moved[lab == j] for j in range(len(c)) if (lab == j).any()]


def write_spec(case, name, gains, widths, dose=1.0, levels=256, min_change=0.002, shuffle=False, seed=0):
    """Gains {tensor (relative to the decoder prefix): per-group array} -> cases/<case>/edits/<name>.json/.npz."""
    c = C.conf(case)
    prefix = decoder_prefix(tensor_index(profiles.base_dir(c["profile"])))
    d = C.cdir(case) / "edits"
    rng = np.random.default_rng(seed)
    arrays, ops, err, tot = {}, [], 0.0, 0.0
    for t, g in sorted(gains.items()):
        g = np.exp(dose * np.log(np.clip(np.asarray(g, np.float64), 1e-3, None)))
        if shuffle:
            g = rng.permutation(g)
        col = np.repeat(g, widths[t])  # one factor per input column
        moved = np.where(np.abs(col - 1) >= min_change)[0]
        if not len(moved):
            continue
        for q, cols in enumerate(_levels(col, moved, levels)):
            key = f"{t.replace('.weight', '').replace('.', '_')}_{q}"
            f = float(np.round(col[cols].mean(), 6))
            arrays[key] = cols.astype(np.int32)
            ops.append({"op": "scale_cols", "tensor": prefix + t, "cols": f"{name}.npz#{key}", "factor": f})
            err += float(((col[cols] - f) ** 2).sum())
        tot += float(((col - 1) ** 2).sum())
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{name}.npz", **arrays)
    spec = {"name": name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(C.cdir(case) / c["parent"], d)
    write_json(d / f"{name}.json", spec)
    tensors, payload = spec_size(d / f"{name}.json")
    print(f"{d / f'{name}.json'}: {len(ops)} ops on {tensors} tensors, payload {payload} bytes; quantisation error "
          f"{100 * (err / tot) ** 0.5 if tot else 0:.1f}% of the change (raise --levels if large)")
    return d / f"{name}.json"


def _fit_path(case, name):
    return C.scratch(case) / "fits" / f"{name}.npz"


def cmd_run(a):
    # kernels compiled for the backward pass: a cache per GPU, since fits started side by side race on a shared one
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser(
        f"~/.triton/cache-gpu{os.environ.get('CUDA_VISIBLE_DEVICES', 'all').replace(',', '-')}"))
    from .probe import load_model  # noqa: PLC0415

    prof = profiles.get(C.conf(a.case)["profile"])
    model, tok = load_model(C.model_path(a.case, C.BASE), a.device)
    n_layers = model.config.get_text_config().num_hidden_layers
    tgts = targets(model, _layers(a.layers, n_layers), tuple(a.modules))
    ex, counts = examples(a.case, a.runs, tok, prof, a.per_problem, a.max_len, a.frac, a.seed)
    print(json.dumps(counts), flush=True)
    if not ex:
        raise SystemExit("no answer to fit on: run the case's model on practice problems outside dev and holdout "
                         "(e.g. `wsp.case split <case> fit --n 512`, then `wsp.case run <case> base fit --n 4`)")
    import torch  # noqa: PLC0415
    torch.manual_seed(a.seed)
    gains, log = train(model, ex, tgts, a.lr, a.epochs, a.accum, a.device)
    widths = {t: w for t, _, w in tgts}
    p = _fit_path(a.case, a.name)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **gains)
    write_json(p.with_suffix(".json"), {"args": {k: v for k, v in vars(a).items() if k != "func"}, "data": counts,
                                        "widths": widths, "log": log,
                                        "gain_range": [float(min(v.min() for v in gains.values())),
                                                       float(max(v.max() for v in gains.values()))]})
    write_spec(a.case, a.name, gains, widths)


def cmd_spec(a):
    fits = [np.load(_fit_path(a.case, f)) for f in a.fits]
    widths = json.load(open(_fit_path(a.case, a.fits[0]).with_suffix(".json")))["widths"]
    keys = sorted(fits[0].files)
    if any(sorted(f.files) != keys for f in fits):
        raise SystemExit("the fits gain different tensors (other --layers / --modules): average like with like")
    avg = {k: np.exp(np.mean([np.log(np.clip(f[k], 1e-3, None)) for f in fits], 0)) for k in keys}
    write_spec(a.case, a.name, avg, widths, a.dose, a.levels, a.min_change, a.shuffle, a.seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="fit gains on the case's own correct answers -> raw gains + a spec at dose 1")
    r.add_argument("case")
    r.add_argument("name")
    r.add_argument("--runs", nargs="+", required=True, help="runs of the case, as <model>__<set> (file stems)")
    r.add_argument("--layers", help="decoder layers a-b (default: all)")
    r.add_argument("--modules", nargs="+", default=list(DEFAULT_MODULES),
                   help="names of the linear layers whose inputs get gains")
    r.add_argument("--per-problem", type=int, default=2)
    r.add_argument("--max-len", type=int, default=12000, help="max tokens of prompt + answer")
    r.add_argument("--lr", type=float, default=2e-3)
    r.add_argument("--epochs", type=float, default=1.0)
    r.add_argument("--accum", type=int, default=8, help="answers per optimiser step")
    r.add_argument("--frac", type=float, default=1.0, help="random fraction of the problems (for several fits)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--device", default="cuda")
    s = sub.add_parser("spec", help="average fits in log space, set a dose (or shuffle as a control) -> a spec")
    s.add_argument("case")
    s.add_argument("name")
    s.add_argument("--fits", nargs="+", required=True)
    s.add_argument("--dose", type=float, default=1.0, help="g = exp(dose * mean log g)")
    s.add_argument("--levels", type=int, default=256, help="factors per tensor (the payload does not grow with it)")
    s.add_argument("--min-change", type=float, default=0.002)
    s.add_argument("--shuffle", action="store_true", help="control: each tensor's gains on random groups")
    s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"run": cmd_run, "spec": cmd_spec}[a.cmd](a)


if __name__ == "__main__":
    main()
