"""Rules for edits and submissions (rules.json): checks and their CLI.

python -m wsp.rules check <spec.json | submissions/<team>/<entry>/> [--track lite|open] [--build]
    static checks (ops, editable tensors, payload, tensor count; for a submission dir also files, entry.json and a
    code scan); --build also builds the edit on the official base and checks the magnitude limits.
python -m wsp.rules compare <a.json> <b.json>   reproduction check of a re-derived spec against the submitted one
python -m wsp.rules doc     regenerate the limits section of README.md from rules.json
"""
import json
import re
from pathlib import Path

from .common import REPO, WORK, read_json

RULES = read_json(REPO / "rules.json")


def tensor_violation(name, prefix, ndim=None):
    """Why `name` may not be edited (None if it may). prefix: the main decoder's name prefix."""
    r = RULES["editable_tensors"]
    if r["require_decoder_layer"] and not re.match(re.escape(prefix) + r"layers\.\d+\.", name):
        return "not inside a main decoder layer"
    if not name.endswith(r["require_suffix"]) or re.search(r["forbid_pattern"], name):
        return "not an editable weight matrix (embedding / head / norm / bias)"
    if ndim is not None and ndim != r["require_ndim"]:
        return f"{ndim}-D parameter"
    return None


def _array_refs(spec):
    for op in spec["ops"]:
        for k in ("u", "cols", "rows"):
            if op.get(k) is not None:
                yield op[k]


def payload_bytes(spec, spec_dir):
    """Bytes of every distinct array the spec carries (its information budget)."""
    import numpy as np  # noqa: PLC0415

    total, seen = 0, set()
    for v in _array_refs(spec):
        key = json.dumps(v)
        if key in seen:
            continue
        seen.add(key)
        if isinstance(v, str):
            path, _, k = v.partition("#")
            total += np.load(Path(spec_dir) / path)[k].nbytes
        else:
            a = np.asarray(v)
            total += a.size * (4 if a.dtype.kind == "f" else 8)
    return total


def check_spec(spec_path, track):
    from .edits import decoder_prefix, snapshot, tensor_index  # noqa: PLC0415

    spec_path = Path(spec_path)
    spec, lim, out = read_json(spec_path), RULES["tracks"][track], []
    prefix = decoder_prefix(tensor_index(snapshot(spec["base"])))
    for op in spec["ops"]:
        if op["op"] not in RULES["ops"]["allowed"]:
            out.append(f"op {op['op']} not allowed")
        why = tensor_violation(op["tensor"], prefix)
        if why:
            out.append(f"{op['tensor']}: {why}")
    n = len({op["tensor"] for op in spec["ops"]})
    if lim["max_tensors"] is not None and n > lim["max_tensors"]:
        out.append(f"{n} tensors > {lim['max_tensors']}")
    pb = payload_bytes(spec, spec_path.parent)
    if pb > lim["max_payload_bytes"]:
        out.append(f"payload {pb} bytes > {lim['max_payload_bytes']}")
    return out, {"tensors": n, "payload_bytes": pb}


def decoder_matrix_norm(base):
    """Frobenius norm over all editable matrices of the base model (cached in $WSP_WORK/data)."""
    from safetensors import safe_open  # noqa: PLC0415
    from .edits import decoder_prefix, snapshot, tensor_index  # noqa: PLC0415

    cache = WORK / "data" / "decoder_norms.json"
    norms = read_json(cache) if cache.exists() else {}
    if base not in norms:
        d = snapshot(base)
        idx = tensor_index(d)
        prefix, sq = decoder_prefix(idx), 0.0
        for sh in sorted(set(idx.values())):
            with safe_open(str(d / sh), "pt") as f:
                for n in f.keys():
                    if tensor_violation(n, prefix) is None:
                        t = f.get_tensor(n)
                        if t.dim() == RULES["editable_tensors"]["require_ndim"]:
                            sq += float(t.float().pow(2).sum())
        norms[base] = sq ** 0.5
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(norms, indent=1))
    return norms[base]


def check_manifest(man, track):
    lim, out = RULES["tracks"][track], []
    for t in man["tensors"]:
        if t["relative_change"] > lim["max_relative_change_per_tensor"]:
            out.append(f"{t['name']}: relative change {t['relative_change']:.3f} > {lim['max_relative_change_per_tensor']}")
    g = (sum((t["relative_change"] * t["base_norm"]) ** 2 for t in man["tensors"]) ** 0.5
         / decoder_matrix_norm(man["spec"]["base"]))
    if g > lim["max_global_relative_change"]:
        out.append(f"global relative change {g:.4f} > {lim['max_global_relative_change']}")
    return out, {"global_relative_change": g, "changed_entries": man["changed_entries_total"]}


def check_submission(d):
    s, out = RULES["submission"], []
    d = Path(d)
    rel = d.resolve().relative_to(REPO).as_posix() + "/"
    if not re.match(s["path_pattern"], rel):
        out.append(f"{rel} does not match {s['path_pattern']}")
    files = [p for p in d.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    out += [f"missing {f}" for f in s["required_files"] if not (d / f).exists()]
    out += [f"{p.relative_to(d)}: file type not allowed" for p in files if p.suffix not in s["allowed_extensions"]]
    size = sum(p.stat().st_size for p in files)
    if size > s["max_total_bytes"]:
        out.append(f"{size} bytes > {s['max_total_bytes']}")
    for p in files:
        if p.suffix == ".py":
            txt = p.read_text(errors="replace")
            out += [f"{p.relative_to(d)}: banned pattern {pat}" for pat in RULES["derivation"]["banned_code_patterns"]
                    if re.search(pat, txt)]
    entry = read_json(d / "entry.json") if (d / "entry.json").exists() else {}
    out += [f"entry.json: missing {k}" for k in s["entry_fields"] if k not in entry]
    if entry.get("track") not in RULES["tracks"]:
        out.append(f"entry.json: unknown track {entry.get('track')}")
    if not set(entry.get("practice_data", [])) <= set(RULES["derivation"]["practice_data"]):
        out.append("entry.json: practice_data outside the allowed sets")
    if entry.get("forward_passes", 0) > RULES["derivation"]["max_forward_passes"]:
        out.append(f"entry.json: forward_passes > {RULES['derivation']['max_forward_passes']}")
    return out, entry


def check(path, track=None, build=False):
    path = Path(path)
    out, info = [], {}
    if path.is_dir():
        out, entry = check_submission(path)
        track = track or entry.get("track", "lite")
        spec_path = path / entry.get("edit", "edit.json")
    else:
        track, spec_path = track or "lite", path
    if track not in RULES["tracks"]:
        return out + [f"unknown track {track}"], info
    if spec_path.exists():
        o, i = check_spec(spec_path, track)
        out, info = out + o, {**info, **i}
        if build and not o:
            from .edits import build as build_edit  # noqa: PLC0415
            _, man = build_edit(spec_path, WORK / "models" / f"check__{spec_path.stem}")
            o, i = check_manifest(man, track)
            out, info = out + o, {**info, **i}
    else:
        out.append(f"missing edit spec {spec_path.name}")
    return out, {"track": track, **info}


def compare(a_path, b_path):
    """Reproduction check of two specs (e.g. submitted vs re-derived): same ops and tensors, direction cosine and
    selected-index overlap per op."""
    import numpy as np  # noqa: PLC0415
    from .edits import _resolve  # noqa: PLC0415

    r = RULES["derivation"]["reproduction"]
    a, b = read_json(a_path), read_json(b_path)
    key = [(o["op"], o["tensor"]) for o in a["ops"]]
    if r["same_ops_and_tensors"] and key != [(o["op"], o["tensor"]) for o in b["ops"]]:
        return ["different ops or tensors"], {}
    out, info = [], []
    for oa, ob in zip(a["ops"], b["ops"]):
        row = {"tensor": oa["tensor"]}
        if oa.get("u") is not None:
            ua = _resolve(oa["u"], Path(a_path).parent).ravel()
            ub = _resolve(ob["u"], Path(b_path).parent).ravel()
            row["cosine"] = float(ua @ ub / (np.linalg.norm(ua) * np.linalg.norm(ub)))
            if row["cosine"] < r["min_direction_cosine"]:
                out.append(f"{oa['tensor']}: direction cosine {row['cosine']:.4f} < {r['min_direction_cosine']}")
        for k in ("cols", "rows"):
            if oa.get(k) is not None:
                sa = set(_resolve(oa[k], Path(a_path).parent).tolist())
                sb = set(_resolve(ob[k], Path(b_path).parent).tolist())
                row[f"{k}_overlap"] = len(sa & sb) / max(len(sa), 1)
                if row[f"{k}_overlap"] < r["min_index_overlap"]:
                    out.append(f"{oa['tensor']}: {k} overlap {row[f'{k}_overlap']:.2f} < {r['min_index_overlap']}")
        info.append(row)
    return out, {"ops": info}


def doc():
    """The README section describing the limits, generated from rules.json."""
    t, e, s, dv, ev = (RULES[k] for k in ("tracks", "editable_tensors", "submission", "derivation", "evaluation"))
    rows = ["| limit | " + " | ".join(t) + " |", "|---|" + "---|" * len(t)]
    for key, label, fmt in (("max_tensors", "edited tensors", lambda v: "no limit" if v is None else str(v)),
                            ("max_payload_bytes", "payload (spec arrays)", lambda v: f"{v // 1024} KiB"),
                            ("max_relative_change_per_tensor", "relative change per tensor", lambda v: f"{v:.0%}"),
                            ("max_global_relative_change", "relative change of all editable matrices", lambda v: f"{v:.0%}")):
        rows.append(f"| {label} | " + " | ".join(fmt(t[k][key]) for k in t) + " |")
    return "\n".join([
        f"Rules v{RULES['version']} (`rules.json`; this section is generated by `python -m wsp.rules doc`).", "",
        f"- **Editable**: {e['about']}",
        f"- **Ops**: {', '.join(f'`{o}`' for o in RULES['ops']['allowed'])}. {RULES['ops']['about']}",
        f"- **Payload**: {RULES['payload']['about']}", "", *rows, "",
        f"- **Derivation**: {dv['about']} Reproduction tolerance: direction cosine >= "
        f"{dv['reproduction']['min_direction_cosine']}, selected-index overlap >= {dv['reproduction']['min_index_overlap']:.0%}, "
        f"score within {dv['reproduction']['max_score_diff_se']:g} SE. Practice data: {', '.join(f'`{p}`' for p in dv['practice_data'])}; "
        f"at most {dv['max_forward_passes']} forward passes ({dv['forward_pass']}); no network, "
        f"subprocesses or benchmark access (code scan: `rules.json` -> `derivation.banned_code_patterns`).",
        f"- **Protected**: {RULES['protected_paths']['about']} Paths: "
        f"{', '.join(f'`{p}`' for p in RULES['protected_paths']['paths'])}.",
        f"- **Submission**: {s['about']} Required: {', '.join(f'`{f}`' for f in s['required_files'])}; file types "
        f"{', '.join(s['allowed_extensions'])}; at most {s['max_total_bytes'] // 1024} KiB; `entry.json` fields: "
        f"{', '.join(f'`{f}`' for f in s['entry_fields'])}.",
        f"- **Evaluation**: {ev['about']} {ev['samples_per_problem']} samples per problem. Public sets "
        f"{', '.join(f'`{x}`' for x in ev['public_sets'])}; private sets: {ev['private_sets']}. Guardrails "
        f"{', '.join(f'`{x}`' for x in ev['guardrail_sets'])}: an entry is disqualified if it drops by more than "
        f"{ev['guardrail_max_drop_se']:g} SE. Ranking: {ev['ranking']}. At most "
        f"{ev['max_submissions_per_team_per_week']} submissions per team per week."])


def main():
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("path")
    c.add_argument("--track", choices=list(RULES["tracks"]))
    c.add_argument("--build", action="store_true")
    m = sub.add_parser("compare", help="reproduction check: submitted vs re-derived spec")
    m.add_argument("a")
    m.add_argument("b")
    sub.add_parser("doc")
    a = ap.parse_args()
    if a.cmd == "compare":
        violations, info = compare(a.a, a.b)
        print(json.dumps({"ok": not violations, "violations": violations, **info}, indent=1))
        raise SystemExit(1 if violations else 0)
    if a.cmd == "doc":
        readme = REPO / "README.md"
        block = f"<!-- rules:begin -->\n{doc()}\n<!-- rules:end -->"
        readme.write_text(re.sub(r"<!-- rules:begin -->.*<!-- rules:end -->", lambda _: block, readme.read_text(),
                                 flags=re.S))
        return
    violations, info = check(a.path, a.track, a.build)
    print(json.dumps({"ok": not violations, "violations": violations, **info}, indent=1))
    raise SystemExit(1 if violations else 0)


if __name__ == "__main__":
    main()
