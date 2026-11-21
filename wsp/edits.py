"""Weight edits as declarative, reproducible specs.

A spec (JSON: {name, base, ops, parent?}; base = a profile name, see profiles.py) lists operations on named
tensors of the base model. "parent" stacks it on another spec, e.g. the current champion's edit: the parent chain
is applied first, then the spec's own ops. "parent" is the name of a published edit placed next to the spec as
<name>/edit/edit.json (a published edit is a directory edit/; in a competition the name is its sid), a path to a spec
(local edits), or the base itself as "<hf_repo>@<revision>" (none: the base too). `build` applies them in
float32, stores the result in the base dtype, writes a loadable model dir (untouched files are symlinks), and an
exact diff + manifest (sha256 of every base and edited tensor) that re-applies bit for bit. The manifest measures
changes both against the base (whole chain) and against the parent model (own ops only: own_tensors,
relative_change_vs_parent).

Operations ("u", "cols", "rows" are inline lists or "file.npz#key" relative to the spec):
  dir_ablate  side "out" (W writes into the residual stream, e.g. down_proj): W -= alpha * u (u^T W)[:, cols]
              side "in"  (W reads from it, e.g. up_proj):                      W -= alpha * (W u)[rows] u^T
  scale_cols  W[:, cols] *= factor        scale_rows  W[rows, :] *= factor
  set_rows    W[rows, :] = values (len(rows) x in)   set_cols  W[:, cols] = values (out x len(cols))
              ("write a neuron": e.g. the gate_proj / up_proj rows and the down_proj column of one MLP unit)
  low_rank    W = best rank-k approximation of W (SVD), k = max(1, round(keep * min(W.shape)))  (LASER)
  round       W = W in the base dtype, back in float32: what building a chain does between its links (a chain carried
              into one spec rounds where its links ended, so it builds the same model bit for bit)
The manifest also records what limit checks need (changed entries, relative change, base norm per tensor); the
limits themselves are in rules.json and are checked by the subnet project, not here.
"""
import json
import os
import re
from pathlib import Path

import numpy as np

from . import profiles
from .common import sha256_bytes, write_json

PUBLISHED = "edit.json"  # a published edit is a directory edit/ with edit.json (+ edit.npz, REPORT.md, its parents)

AUXILIARY_STACKS = r"mtp|visual|vision"  # layer stacks that are not the main decoder (MTP heads, vision towers)


def tensor_index(model_dir):
    """tensor name -> shard file name"""
    idx = Path(model_dir) / "model.safetensors.index.json"
    if idx.exists():
        return json.load(open(idx))["weight_map"]
    from safetensors import safe_open  # noqa: PLC0415
    with safe_open(str(Path(model_dir) / "model.safetensors"), "pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


def decoder_prefix(names):
    """Name prefix of the main decoder stack: the one with the most layers that is not an auxiliary stack."""
    count = {}
    for n in names:
        m = re.match(r"(.*)layers\.(\d+)\.", n)
        if m and not re.search(AUXILIARY_STACKS, m.group(1)):
            count[m.group(1)] = max(count.get(m.group(1), 0), int(m.group(2)) + 1)
    return max(count, key=count.get)


ARRAY_FIELDS = ("u", "cols", "rows", "values")  # op fields that may hold an array, inline or as "file.npz#key"


def _resolve(v, base_dir):
    if v is None or isinstance(v, (list, int, float)):
        return None if v is None else np.asarray(v)
    path, _, key = str(v).partition("#")
    z = np.load(Path(base_dir) / path)
    return np.asarray(z[key] if key else z)


def _mask(sel, like):
    import torch  # noqa: PLC0415
    if sel is None:
        return torch.ones_like(like)
    m = torch.zeros_like(like)
    m[torch.as_tensor(sel, dtype=torch.long)] = 1
    return m


NAME = re.compile(r"[A-Za-z0-9_-]+")  # a published edit's name (a sid): one path component, nothing else


def parent_path(spec, d):
    """The spec file a spec (in directory d) stacks on; None when it starts from the base."""
    ref = spec.get("parent")
    if not ref or ref == profiles.base_name(spec["base"]):
        return None
    if NAME.fullmatch(ref):
        return (Path(d) / ref / "edit" / PUBLISHED).resolve()
    if ref.endswith(".json"):
        return (Path(d) / ref).resolve()
    raise ValueError(f"parent {ref!r} is neither a published edit's name, a spec, nor this spec's base "
                     f"({profiles.base_name(spec['base'])})")


def load_chain(spec_path):
    """[(spec, its directory)] from the root parent down to this spec; all on the same base."""
    chain, seen, p = [], set(), Path(spec_path).resolve()
    while p is not None:
        if p in seen:
            raise ValueError(f"parent cycle at {p}")
        seen.add(p)
        spec = json.load(open(p))
        chain.append((spec, p.parent))
        p = parent_path(spec, p.parent)
    chain.reverse()
    if len({s["base"] for s, _ in chain}) > 1:
        raise ValueError("a spec and its parents must share one base")
    return chain


def _published(p):
    p = Path(p).resolve()  # a relative path given from inside the directory names it as well
    return p.name == PUBLISHED and p.parent.name == "edit"  # kept as <name>/edit/edit.json


def start_dir(spec_path):
    """The name an inherited edit keeps in a case (<name>/edit/), which an upload also uses to reference it: a
    published edit keeps the name of the directory holding its edit/ (in a task, its sid); any other spec goes to
    parent/."""
    p = Path(spec_path).resolve()
    return p.parent.parent.name if _published(p) else "parent"


def copy_parent(spec_path, dst):
    """Copy an inherited edit into dst: a published edit directory verbatim to dst/edit/ (byte-identical, so it can
    be checked against its publication), any other spec by `export`. Returns the copied spec path."""
    import shutil  # noqa: PLC0415

    spec_path = Path(spec_path)
    if _published(spec_path):
        shutil.copytree(spec_path.parent, Path(dst) / "edit", dirs_exist_ok=True)
        return Path(dst) / "edit" / PUBLISHED
    return export(spec_path, dst, json.load(open(spec_path)).get("name") or spec_path.stem)


def _carry(links, name):
    """The ops of links [(spec, its directory)], in order, with every array they reference or hold inline moved to
    name.npz (one stored array per distinct file and reference, or inline list), and after every link but the last a
    round of each tensor it edited, as building the chain rounds each link's model: the ops build exactly the chain's
    model -> (ops, arrays)."""
    ops, arrays, keys = [], {}, {}
    for j, (spec, d) in enumerate(links):
        if j:
            ops += [{"op": "round", "tensor": t} for t in sorted({op["tensor"] for op in links[j - 1][0]["ops"]})]
        for op in spec["ops"]:
            op = dict(op)
            for k in ARRAY_FIELDS:
                ref = op.get(k)
                if isinstance(ref, (str, list)):
                    src = (Path(d).resolve(), ref) if isinstance(ref, str) else json.dumps(ref)
                    if src not in keys:
                        path, _, key = ref.partition("#") if isinstance(ref, str) else ("", "", k)
                        new, i = key or Path(path).stem, 1
                        while new in arrays:
                            new, i = f"{key or Path(path).stem}_{i}", i + 1
                        a = _resolve(ref, d)
                        if isinstance(ref, list):  # stored as spec_size counts it; builds the same float32 / long
                            a = a.astype(np.float32 if a.dtype.kind == "f" else np.int64)
                        arrays[new], keys[src] = a, new
                    op[k] = f"{name}.npz#{keys[src]}"
            ops.append(op)
    return ops, arrays


def _write(dst, name, spec, ops, arrays):
    dst.mkdir(parents=True, exist_ok=True)
    if arrays:
        np.savez(dst / f"{name}.npz", **arrays)
    out = {"base": spec["base"], **({"parent": spec["parent"]} if spec.get("parent") else {}), "ops": ops}
    write_json(dst / f"{name}.json", out)  # nothing else
    return dst / f"{name}.json"


def export(spec_path, dst, name):
    """Copy a spec into dst as <name>.json (+ <name>.npz holding its own arrays), without local bookkeeping, its
    parent chain next to it (see copy_parent; a published parent referenced by its name): self-contained. Returns
    the new spec path."""
    src, dst = Path(spec_path), Path(dst)
    spec = json.load(open(src))
    ops, arrays = _carry([(spec, src.parent)], name)
    parent = parent_path(spec, src.parent)
    if parent:
        where = start_dir(parent)
        copied = copy_parent(parent, dst / where)
        spec["parent"] = where if _published(copied) and NAME.fullmatch(where) else os.path.relpath(copied, dst)
    return _write(dst, name, spec, ops, arrays)


def export_upload(spec_path, dst, start=None, starts=None):
    """Write a spec as an upload, dst/edit.json (+ edit.npz), that builds the same model from what the protocol can
    reference (rules.json -> edit.inheritance): a published start of the task by its sid (not uploaded; the subnet
    places it at <sid>/edit/), or the base as "<hf_repo>@<revision>". Local edits the spec stacks on (not published) are
    carried: their ops go into the upload, in order. The published edit nearest in the chain is the start, referenced
    by the name of its directory (in a task, its sid); start: a sid names it instead, "base" carries the whole chain.
    starts: the sids the task offers (task.json -> starts), checked when given.
    -> (path, the "parent" it writes, names of the carried edits)."""
    chain = load_chain(spec_path)  # also refuses cycles and mixed bases
    base = profiles.base_name(chain[-1][0]["base"])
    if base is None:
        raise SystemExit(f"base {chain[-1][0]['base']!r} is not a profile (wsp/profiles.py)")
    paths, p = [], Path(spec_path).resolve()
    for spec, d in reversed(chain):
        paths.append(p)
        p = parent_path(spec, d)
    paths.reverse()
    i = max((j for j, q in enumerate(paths[:-1]) if _published(q)), default=None)
    if start not in (None, "base") and i is None:
        raise SystemExit(f"--start {start}: this edit stacks on no published edit (it builds from the base)")
    sid = None if i is None or start == "base" else start or start_dir(paths[i])
    if sid is not None and (not NAME.fullmatch(sid) or starts is not None and sid not in starts):
        offers = "" if starts is None else f", but the task offers {', '.join(starts) or 'none'} (task.json -> starts)"
        raise SystemExit(f"this edit stacks on the published edit in {paths[i].parent}, named {sid!r}{offers}. Keep the "
                         "start under its sid (init --parent <sid>/edit/edit.json does) or pass --start <sid>; or "
                         "--start base to carry its whole chain's ops in this edit (built from the base; the limits "
                         "then count them)")
    first = 0 if sid is None else i + 1
    ops, arrays = _carry(chain[first:], "edit")
    spec = {"base": chain[-1][0]["base"], "parent": sid or base}
    carried = [start_dir(paths[j]) if _published(paths[j]) else chain[j][0].get("name") or paths[j].stem
               for j in range(first, len(chain) - 1)]
    return _write(Path(dst), "edit", spec, ops, arrays), spec["parent"], carried


def flatten_chain(spec_path, name="edit"):
    """The whole chain as one spec on the base that builds the same model bit for bit, whatever its ops ->
    (spec, arrays), its arrays referenced as <name>.npz#key."""
    chain = load_chain(spec_path)
    ops, arrays = _carry(chain, name)
    return {"base": chain[-1][0]["base"], "parent": profiles.base_name(chain[-1][0]["base"]), "ops": ops}, arrays


OPS = ("dir_ablate", "scale_cols", "scale_rows", "set_rows", "set_cols", "low_rank", "round")  # every op apply_ops runs


def apply_ops(tensors, ops, base_dir, dtypes=None):
    """tensors: name -> float32 torch tensor, modified in place; dtypes: name -> the base dtype (for round)."""
    import torch  # noqa: PLC0415

    for op in ops:
        W = tensors[op["tensor"]]
        if op["op"] == "dir_ablate":
            u = torch.tensor(_resolve(op["u"], base_dir), dtype=torch.float32).flatten()
            u, a = u / u.norm(), float(op.get("alpha", 1.0))
            if op.get("side", "out") == "out":
                p = u @ W
                W -= a * torch.outer(u, p * _mask(_resolve(op.get("cols"), base_dir), p))
            else:
                p = W @ u
                W -= a * torch.outer(p * _mask(_resolve(op.get("rows"), base_dir), p), u)
        elif op["op"] in ("scale_cols", "scale_rows"):
            dim = 1 if op["op"] == "scale_cols" else 0
            sel = torch.as_tensor(_resolve(op["cols" if dim else "rows"], base_dir), dtype=torch.long)
            W.index_copy_(dim, sel, W.index_select(dim, sel) * float(op["factor"]))
        elif op["op"] in ("set_rows", "set_cols"):
            dim = 1 if op["op"] == "set_cols" else 0
            sel = torch.as_tensor(_resolve(op["cols" if dim else "rows"], base_dir), dtype=torch.long).flatten()
            val = torch.tensor(_resolve(op["values"], base_dir), dtype=torch.float32)
            val = val.reshape(len(sel), -1) if dim == 0 else val.reshape(-1, len(sel))
            if dim == 0:
                W[sel] = val
            else:
                W[:, sel] = val
        elif op["op"] == "round":  # to the base dtype and back, as between the links of a chain
            if dtypes is None:
                raise ValueError("round needs the base dtypes (apply_ops(..., dtypes))")
            W.copy_(W.to(dtypes[op["tensor"]]).to(torch.float32))
        elif op["op"] == "low_rank":
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            k = max(1, round(float(op["keep"]) * min(W.shape)))
            W.copy_((U[:, :k] * S[:k]) @ Vh[:k])
        else:
            raise ValueError(f"unknown op {op['op']}")


def build(spec_path, out_dir, diff_dir=None, base_dir=None):
    """Apply a spec to its base model -> (model_dir, manifest). base_dir: an already verified local copy of the base
    (default: profiles.base_dir(spec["base"]))."""
    import torch  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415
    from safetensors.torch import load_file, save_file  # noqa: PLC0415

    out_dir = Path(out_dir)
    chain = load_chain(spec_path)
    spec = chain[-1][0]
    base = Path(base_dir) if base_dir else profiles.base_dir(spec["base"])
    index = tensor_index(base)
    names = sorted({op["tensor"] for s, _ in chain for op in s["ops"]})
    orig = {}
    for n in names:
        with safe_open(str(base / index[n]), "pt") as f:
            orig[n] = f.get_tensor(n)
    work = {n: t.to(torch.float32).clone() for n, t in orig.items()}
    for i, (s, d) in enumerate(chain):  # parents first; each link yields a real model (rounded to the base dtype),
        if i == len(chain) - 1:         # so a child starts from exactly the parent model's weights
            parent = {n: work[n].to(orig[n].dtype) for n in names}
        apply_ops(work, s["ops"], d, {n: t.dtype for n, t in orig.items()})
        work = {n: w.to(orig[n].dtype).to(torch.float32) for n, w in work.items()}
    edited = {n: work[n].to(orig[n].dtype).contiguous() for n in names}
    own = sorted({op["tensor"] for op in spec["ops"]})

    out_dir.mkdir(parents=True, exist_ok=True)
    shards = {index[n] for n in names}
    for f in base.iterdir():
        dst = out_dir / f.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if f.name not in shards:
            os.symlink(f.resolve(), dst)
    for sh in shards:
        t = load_file(str(base / sh))
        t.update({n: edited[n] for n in names if index[n] == sh})
        save_file(t, str(out_dir / sh), metadata={"format": "pt"})

    def bits(x):
        return (x.view(torch.int16) if x.dtype == torch.bfloat16 else x).numpy()

    man = {"spec": spec, "chain": [s.get("name") for s, _ in chain], "base_dir": str(base), "own_tensors": own,
           "tensors": []}
    diff = {}
    for i, n in enumerate(names):
        a, b = bits(orig[n]), bits(edited[n])
        changed = np.flatnonzero(a.reshape(-1) != b.reshape(-1))
        diff[f"t{i}_idx"], diff[f"t{i}_val"] = changed.astype(np.int64), b.reshape(-1)[changed]
        of, pf, ef = orig[n].float(), parent[n].float(), edited[n].float()
        man["tensors"].append({"name": n, "shape": list(orig[n].shape), "dtype": str(orig[n].dtype),
                               "changed_entries": int(len(changed)),
                               "changed_cols": int((of != ef).any(0).sum()) if of.dim() == 2 else None,
                               "relative_change": float((ef - of).norm() / of.norm()), "base_norm": float(of.norm()),
                               "relative_change_vs_parent": float((ef - pf).norm() / pf.norm()), "parent_norm": float(pf.norm()),
                               "base_sha256": sha256_bytes(a.tobytes()), "edited_sha256": sha256_bytes(b.tobytes())})
    man["changed_entries_total"] = sum(t["changed_entries"] for t in man["tensors"])
    write_json(out_dir / "wsp_edit_manifest.json", man)
    if diff_dir:
        Path(diff_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(diff_dir) / "diff.npz", **diff)
        write_json(Path(diff_dir) / "manifest.json", man)
    return out_dir, man


def attn_out(index, prefix, layer):
    """The tensor a layer's attention writes the residual with: o_proj, or the output projection of another attention
    kind (e.g. linear attention's out_proj in hybrid models)."""
    pat = re.compile(re.escape(f"{prefix}layers.{layer}.") + r"[a-z_]*attn[a-z_]*\.(o_proj|out_proj)\.weight$")
    found = sorted(n for n in index if pat.match(n))
    if not found:
        raise ValueError(f"no attention output projection in layer {layer}")
    return found[0]


def dir_ablate_spec(directions, base, layers, topk, alpha, out_dir, name, random_control=False, seed=0, parent=None,
                    targets=("down_proj",)):
    """Spec ablating each layer's direction from its MLP down_proj, restricted to the top-k writer neurons, and/or
    from its attention output (o_proj, or the layer's other attention output: all columns, heads have no writer
    score).
    random_control: a random unit direction orthogonal to u instead (same layers, columns and alpha).
    parent: path of the spec this one stacks on, relative to out_dir."""
    d = np.load(directions)
    index = tensor_index(profiles.base_dir(base))
    prefix = decoder_prefix(index)
    rng, arrays, ops = np.random.default_rng(seed), {}, []
    for l in layers:
        u = d["U"][l].astype(np.float32)
        if random_control:
            r = rng.standard_normal(u.shape).astype(np.float32)
            u = (r - (r @ u) * u) / np.linalg.norm(r - (r @ u) * u)
        arrays[f"u{l}"] = u
        if "o_proj" in targets:
            ops.append({"op": "dir_ablate", "tensor": attn_out(index, prefix, l),
                        "u": f"{name}.npz#u{l}", "alpha": alpha, "side": "out"})
        if "down_proj" not in targets:
            continue
        op = {"op": "dir_ablate", "tensor": f"{prefix}layers.{l}.mlp.down_proj.weight", "u": f"{name}.npz#u{l}",
              "alpha": alpha, "side": "out"}
        if topk:
            arrays[f"cols{l}"] = np.argsort(-d["writer"][l])[:topk].astype(np.int64)
            op["cols"] = f"{name}.npz#cols{l}"
        ops.append(op)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{name}.npz", **arrays)
    spec = {"name": name, "base": base, "ops": ops}
    if parent:
        spec["parent"] = parent
    write_json(out_dir / f"{name}.json", spec)
    return out_dir / f"{name}.json"


def spec_size(spec_path):
    """(edited tensors, payload bytes) of the spec's own ops, without its parents. Payload = every distinct array the
    spec carries (npz arrays as stored, inline lists as float32 / int64)."""
    spec_path = Path(spec_path)
    spec, seen, payload = json.load(open(spec_path)), set(), 0
    for op in spec["ops"]:
        for k in ARRAY_FIELDS:
            v = op.get(k)
            if v is None or json.dumps(v) in seen:
                continue
            seen.add(json.dumps(v))
            a = _resolve(v, spec_path.parent)
            payload += a.nbytes if isinstance(v, str) else a.size * (4 if a.dtype.kind == "f" else 8)
    return len({op["tensor"] for op in spec["ops"]}), payload
