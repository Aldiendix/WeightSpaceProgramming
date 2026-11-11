"""Case workflow CLI: python -m wsp.case <command> ...   (see .claude/skills/weight-edit/SKILL.md for the protocol)

A case = one base model (or inherited edit) + one behavior to remove. Records protocol: only what cannot be
regenerated cheaply and stays useful is kept in git; everything else is reproducible or scratch.
  kept in cases/<case>/:
    case.json                  {"profile", "contrast", "subsets": {name: "split:n"}, "frozen", "parent", "practice"}
    sets/dev.json, sets/holdout.json   the practice splits (subsets are derived from case.json rules)
    parent/                    (stacked cases) the inherited edit, verbatim
    edits/<edit>.json|.npz     edit specs (the products; models rebuild bit for bit from them)
    results/<model>__<set>.json    {"summary", "paired"}; <model>__selectivity.json; directions.json (per layer)
    NOTES.md                   decisions, by hand
  generated, not kept: RESULTS.md (`report`; the evidence table in playbook.md is the one generated view in git)
  scratch in $WSP_WORK: runs/<case>/<model>__<set>.jsonl, cases/<case>/{pairs*.json,directions.npz} (edits carry
    the directions and neurons they use), models/<case>__<model>/
<model> is "base" or an edit name, optionally with sampling overrides "<model>@key=value,..." (a decoding baseline,
paired with the plain base run); <set> is a case set or subset name, or `bench:<name>`.
"""
import argparse
import json
import os
import re

from . import data, profiles
from .common import CASES, MODELS, REPO, RUNS, WORK, read_json, read_jsonl, write_json
from .stats import paired, summarize

BASE = "base"


def cdir(case):
    return CASES / case


def conf(case):
    return read_json(cdir(case) / "case.json")


def scratch(case):
    return WORK / "cases" / case


def set_ids(case, s):
    """Ids of a split (sets/<s>.json) or of a subset rule "split:n" in case.json: the first n/k problems of each
    practice source, in split order."""
    path = cdir(case) / "sets" / f"{s}.json"
    if path.exists():
        return read_json(path)
    split, n = conf(case)["subsets"][s].split(":")
    ids, srcs = set_ids(case, split), [p.split("_")[0] for p in conf(case).get("practice", data.PRACTICE)]
    return [i for src in srcs for i in [i for i in ids if i.startswith(src)][: int(n) // len(srcs)]]


def set_items(case, s):
    if s.startswith("bench:"):
        return s.split(":", 1)[1], data.load(s)
    idx = data.by_id()
    return s, [idx[i] for i in set_ids(case, s)]


def run_path(case, model, tag):
    return RUNS / case / f"{model}__{tag}.jsonl"


def split_model(model):
    """'base@repetition_penalty=1.1,hint=concise' -> ('base', {'repetition_penalty': 1.1, 'hint': 'concise'})"""
    name, _, over = model.partition("@")
    def val(v):
        try:
            return float(v)
        except ValueError:
            return v
    return name, {k: val(v) for k, v in (kv.split("=") for kv in over.split(",") if kv)}


def model_path(case, model):
    """The case's starting model (the pinned base, or the parent edit built on it for a stacked case), or the built
    model of one of its edits (built on first use)."""
    parent = conf(case).get("parent")
    if model == BASE and not parent:
        return str(profiles.base_dir(conf(case)["profile"]))
    spec = cdir(case) / parent if model == BASE else cdir(case) / "edits" / f"{model}.json"
    path = MODELS / f"{case}__{model}"
    if not (path / "wsp_edit_manifest.json").exists():
        from .edits import build  # noqa: PLC0415
        build(spec, path)
    return str(path)


def _base_run_for(case, keys, tag):
    """The base run of this case covering all (problem, sample) keys: the one of the same set first, else another
    covering them (e.g. base__dev covers dev128). Variants of a benchmark share ids (bench:mmlupro500_sug), so the
    same set must win."""
    same = run_path(case, BASE, tag)
    others = sorted((RUNS / case).glob(f"{BASE}__*.jsonl"), key=lambda p: p.stat().st_size)
    for p in ([same] if same.exists() else []) + [p for p in others if p != same and tag not in data.BENCHMARKS]:
        rows = read_jsonl(p)
        if keys <= {(r["id"], r["k"]) for r in rows}:
            return rows
    return None


def _write_result(case, model, tag):
    rows = read_jsonl(run_path(case, model, tag))
    res = {"summary": summarize(rows), "paired": None}
    if model != BASE:
        base = _base_run_for(case, {(r["id"], r["k"]) for r in rows}, tag)
        res["paired"] = paired(base, rows) if base else None
    write_json(cdir(case) / "results" / f"{model}__{tag}.json", res)
    return res


def cmd_init(a):
    d = cdir(a.case)
    if (d / "case.json").exists():
        raise SystemExit(f"{d} exists")
    subsets = dict(re.fullmatch(r"(\w+)=(\w+:\d+)", x).groups() for x in a.subset)
    c = {"profile": a.profile, "contrast": a.contrast, "subsets": subsets, "frozen": None, "parent": None,
         "practice": list(a.practice)}
    if a.parent:  # stack on an existing edit: copy it (with its own parents) into the case
        from .edits import copy_parent, load_chain  # noqa: PLC0415
        if load_chain(a.parent)[-1][0]["base"] != a.profile:
            raise SystemExit("the parent edit is for another base")
        c["parent"] = copy_parent(a.parent, d / "parent").relative_to(d).as_posix()
    write_json(d / "case.json", c)
    include, src = (read_json(a.include) if a.include else []), tuple(a.practice)
    dev = include + [r["id"] for r in data.balanced_sample(a.dev - len(include), a.seed, exclude=include, sources=src)]
    sets = {"dev": dev, "holdout": [r["id"] for r in
                                    data.balanced_sample(a.holdout, a.seed + 1, exclude=dev, sources=src)]}
    for name, ids in sets.items():
        write_json(d / "sets" / f"{name}.json", ids)
    print({k: len(set_ids(a.case, k)) for k in (*sets, *subsets)})


def cmd_run(a):
    from .gen import generate  # noqa: PLC0415

    prof = profiles.get(conf(a.case)["profile"])
    tag, items = set_items(a.case, a.set)
    out = run_path(a.case, a.model, tag)
    name, sampling = split_model(a.model)
    if a.force or not out.exists():
        generate(model_path(a.case, name), items, prof, n=a.n, out=out, sampling=sampling)
    res = _write_result(a.case, a.model, tag)
    if a.model == BASE:  # pair every edit already run on this set
        for p in (RUNS / a.case).glob(f"*__{tag}.jsonl"):
            if not p.name.startswith(BASE + "__"):
                _write_result(a.case, p.name.split("__")[0], tag)
    print(json.dumps(res["paired"] or res["summary"], indent=1))


def cmd_pairs(a):
    from transformers import AutoTokenizer  # noqa: PLC0415
    from .directions import make_pairs  # noqa: PLC0415

    prof = profiles.get(conf(a.case)["profile"])
    exclude = {i for s in a.exclude for i in set_ids(a.case, s)}
    exclude |= {q["id"] for p in a.exclude_pairs for q in read_json(scratch(a.case) / f"{p}.json")}
    rows = [r for s in a.sets for r in read_jsonl(run_path(a.case, BASE, s)) if r["id"] not in exclude]
    pairs = make_pairs(rows, AutoTokenizer.from_pretrained(str(profiles.base_dir(prof["name"]))), prof["thinking"], a.n, a.seed, a.per_trace,
                       kind=conf(a.case)["contrast"])
    write_json(scratch(a.case) / f"{a.out}.json", pairs)
    print(f"{len(pairs)} pairs from {len({q['id'] for q in pairs})} traces -> {a.out}.json")


def cmd_extract(a):
    import numpy as np  # noqa: PLC0415
    from .directions import extract, layer_table  # noqa: PLC0415
    from .probe import Probe  # noqa: PLC0415

    d = extract(Probe(model_path(a.case, BASE), a.device), read_json(scratch(a.case) / f"{a.pairs}.json"))
    np.savez_compressed(scratch(a.case) / "directions.npz", **d)
    table = layer_table(d)
    write_json(cdir(a.case) / "results" / "directions.json", table)
    for r in table:
        print("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))


def cmd_selectivity(a):
    from .directions import selectivity  # noqa: PLC0415
    from .probe import Probe  # noqa: PLC0415

    res = selectivity(Probe(model_path(a.case, a.model), a.device), read_json(scratch(a.case) / f"{a.pairs}.json"))
    write_json(cdir(a.case) / "results" / f"{a.model}__selectivity.json", res)
    print(res)


def cmd_edit(a):
    from .edits import dir_ablate_spec, low_rank_spec, spec_size  # noqa: PLC0415

    c = conf(a.case)
    parent = os.path.relpath(cdir(a.case) / c["parent"], cdir(a.case) / "edits") if c.get("parent") else None
    layers = "-".join(map(str, a.layers))
    if a.low_rank:
        name = a.name or f"LR_{a.low_rank}_L{layers}_keep{a.keep:g}"
        spec = low_rank_spec(c["profile"], a.layers, a.low_rank, a.keep, cdir(a.case) / "edits", name, parent)
    else:
        tgt = "" if a.targets == ["down_proj"] else "_" + "+".join(t.split("_")[0] for t in a.targets)
        name = a.name or ("RAND_" if a.random_control else "") + f"L{layers}_k{a.topk}_a{a.alpha:g}{tgt}"
        spec = dir_ablate_spec(scratch(a.case) / "directions.npz", c["profile"], a.layers, a.topk, a.alpha,
                               cdir(a.case) / "edits", name, a.random_control, parent=parent, targets=a.targets)
    tensors, payload = spec_size(spec)
    print(f"{spec}: {tensors} tensors, payload {payload} bytes")


def cmd_freeze(a):
    c = conf(a.case)
    if not list((cdir(a.case) / "results").glob(f"{a.edit}__*.json")):
        raise SystemExit("test the edit first (no results for it)")
    c["frozen"] = a.edit
    write_json(cdir(a.case) / "case.json", c)


def cmd_screen(a):
    """Base runs of several profiles on one fixed id list: the screening case has no single profile."""
    from .gen import generate  # noqa: PLC0415

    d = cdir("screening")
    idx = data.by_id()
    items = [idx[i] for i in read_json(d / "sets" / f"{a.set}.json")]
    for p in a.profiles:
        out = run_path("screening", p, a.set)
        if not out.exists():
            generate(profiles.base_dir(p), items, profiles.get(p), out=out)
        write_json(d / "results" / f"{p}__{a.set}.json", {"summary": summarize(read_jsonl(out)), "paired": None})


def _pct(x):
    return "" if x is None else f"{100 * x:.1f}%"


def _results(case):
    out = {}
    for p in sorted((cdir(case) / "results").glob("*__*.json")):
        model, tag = p.stem.split("__")
        out[(model, tag)] = read_json(p)
    return out


def case_tables(case):
    res = _results(case)
    lines = [f"# Results: {case} (generated by `python -m wsp.case report`)", ""]
    if case != "screening":
        c = conf(case)
        lines += [f"Profile `{c['profile']}`, contrast `{c['contrast']}`, frozen edit: `{c.get('frozen')}`.", ""]
    lines += ["| model | set | n | acc | cut share | acc when finished |", "|---|---|---|---|---|---|"]
    for (m, t), r in res.items():
        if t != "selectivity" and (r["paired"] is None):
            s = r["summary"]
            lines.append(f"| {m} | {t} | {s['n']} | {_pct(s['acc'])} | {_pct(s['cut_share'])} | {_pct(s['acc_when_finished'])} |")
    sel = {m: r for (m, t), r in res.items() if t == "selectivity"}
    rows = [(m, t, r["paired"]) for (m, t), r in res.items() if t != "selectivity" and r["paired"]]
    if rows or sel:
        b = sel.get(BASE, {})
        lines += ["", f"Selectivity = mean log-prob of held-out good / bad continuations (base "
                      f"{b.get('logp_good', float('nan')):.3f} / {b.get('logp_bad', float('nan')):.3f}).", "",
                  "| edit | set | base -> edit | gain ± SE | rescued / lost | cut base -> edit | selectivity |",
                  "|---|---|---|---|---|---|---|"]
        for m, t, p in rows:
            s = sel.get(m)
            s = f"{s['logp_good']:.3f} / {s['logp_bad']:.3f}" if s else ""
            lines.append(f"| {m} | {t} | {_pct(p['base_acc'])} -> {_pct(p['edit_acc'])} | "
                         f"{100 * p['gain']:+.1f} ± {100 * p['se']:.1f} | {p['rescues']} / {p['losses']} | "
                         f"{_pct(p['base_cut'])} -> {_pct(p['edit_cut'])} | {s} |")
    return "\n".join(lines) + "\n"


def headline():
    """Frozen edit of every case on every confirmation set (sets named dev* are for selection and are left out)."""
    lines = ["| case | set | base | frozen edit | gain ± SE | rescued / lost |", "|---|---|---|---|---|---|"]
    for d in sorted(CASES.iterdir()):
        if not (d / "case.json").exists() or not conf(d.name).get("frozen"):
            continue
        for (m, t), r in _results(d.name).items():
            if m == conf(d.name)["frozen"] and t != "selectivity" and not t.startswith("dev") and r["paired"]:
                p = r["paired"]
                lines.append(f"| {d.name} | {t} | {_pct(p['base_acc'])} | {_pct(p['edit_acc'])} | "
                             f"{100 * p['gain']:+.1f} ± {100 * p['se']:.1f} | {p['rescues']} / {p['losses']} |")
    return "\n".join(lines)


def cmd_report(a):
    for d in sorted(CASES.iterdir()):
        if (d / "results").is_dir():
            (d / "RESULTS.md").write_text(case_tables(d.name))
    rep = REPO / "playbook.md"
    if not os.access(rep, os.W_OK):  # read-only checkout: only the case records are written
        print(headline())
        return
    txt = rep.read_text()
    block = f"<!-- headline:begin -->\n{headline()}\n<!-- headline:end -->"
    rep.write_text(re.sub(r"<!-- headline:begin -->.*<!-- headline:end -->", lambda _: block, txt, flags=re.S))
    print(headline())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def p(name, *args, **kw):
        s = sub.add_parser(name, **kw)
        for a in args:
            s.add_argument(*a[0], **a[1])
        return s

    case = (["case"], {})
    dev = (["--device"], {"default": "cuda"})
    p("init", case, (["--profile"], {"required": True}), (["--contrast"], {"default": "loop"}),
      (["--dev"], {"type": int, "default": 256}), (["--holdout"], {"type": int, "default": 512}),
      (["--include"], {"help": "json id list forced into dev (e.g. screening problems already looked at)"}),
      (["--subset"], {"action": "append", "default": [], "help": "name=split:n, e.g. dev128=dev:128"}),
      (["--seed"], {"type": int, "default": 7}),
      (["--practice"], {"nargs": "+", "default": list(data.PRACTICE), "help": "practice sources for the splits"}),
      (["--parent"], {"help": "edit to build on, e.g. a published submission's edit/edit.json (copied verbatim): "
                                "the case starts from that model"}),
      help="create a case and its practice splits")
    p("run", case, (["model"], {"help": "base or an edit name"}), (["set"], {"help": "case set or bench:<name>"}),
      (["--n"], {"type": int, "default": 1, "help": "samples per problem"}), (["--force"], {"action": "store_true"}),
      help="generate + grade, write results (paired with base when possible)")
    p("pairs", case, (["--sets"], {"nargs": "+", "required": True, "help": "base runs to take good traces from"}),
      (["--exclude"], {"nargs": "*", "default": [], "help": "sets whose problems are not used"}),
      (["--exclude-pairs"], {"nargs": "*", "default": [], "help": "pair files whose traces are not reused"}),
      (["--n"], {"type": int, "default": 48}), (["--per-trace"], {"type": int, "default": 3}),
      (["--seed"], {"type": int, "default": 0}), (["--out"], {"default": "pairs"}), help="build contrast pairs")
    p("extract", case, (["--pairs"], {"default": "pairs"}), dev, help="directions + writer scores -> directions.npz")
    p("selectivity", case, (["model"], {}), (["--pairs"], {"default": "pairs_heldout"}), dev,
      help="log-prob of held-out good / bad continuations")
    p("edit", case, (["--layers"], {"type": int, "nargs": "+", "required": True}),
      (["--topk"], {"type": int, "default": 64}), (["--alpha"], {"type": float, "default": 6.0}),
      (["--random-control"], {"action": "store_true"}), (["--name"], {}),
      (["--targets"], {"nargs": "+", "default": ["down_proj"], "choices": ["down_proj", "o_proj"],
                       "help": "matrices the direction is ablated from (o_proj: all columns)"}),
      (["--low-rank"], {"choices": ["gate_proj", "up_proj", "down_proj"],
                        "help": "instead: replace that MLP matrix by its rank-k approximation (LASER)"}),
      (["--keep"], {"type": float, "default": 0.1, "help": "fraction of the rank kept by --low-rank"}),
      help="write an edit spec (dir_ablate on the case's directions, or --low-rank)")
    p("freeze", case, (["edit"], {}), help="record the frozen edit before HOLDOUT / benchmarks")
    p("screen", (["profiles"], {"nargs": "+"}), (["--set"], {"default": "screen64"}),
      help="base runs of several profiles on cases/screening/sets/<set>.json")
    p("report", help="write cases/*/RESULTS.md and the evidence table in playbook.md")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
