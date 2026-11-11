"""Weight edits as declarative, reproducible specs.

A spec (JSON: {name, base, ops, parent?}; base = a profile name, see profiles.py) lists operations on named
tensors of the base model. "parent" (a path relative to the spec) stacks it on another spec, e.g. the current
champion's edit: the parent chain is applied first, then the spec's own ops. `build` applies them in
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


def load_chain(spec_path):
    """[(spec, its directory)] from the root parent down to this spec; all on the same base."""
    chain, seen, p = [], set(), Path(spec_path).resolve()
    while p is not None:
        if p in seen:
            raise ValueError(f"parent cycle at {p}")
        seen.add(p)
        spec = json.load(open(p))
        chain.append((spec, p.parent))
        p = (p.parent / spec["parent"]).resolve() if spec.get("parent") else None
    chain.reverse()
    if len({s["base"] for s, _ in chain}) > 1:
        raise ValueError("a spec and its parents must share one base")
    return chain


PUBLISHED = "edit.json"  # a published edit is a directory edit/ with edit.json (+ edit.npz, parent/, ...)


def copy_parent(spec_path, dst):
    """Copy an inherited edit into dst: a published edit directory verbatim (byte-identical, so it can be checked
    against its publication), any other spec by `export`. Returns the copied spec path."""
    import shutil  # noqa: PLC0415

    spec_path = Path(spec_path)
    if spec_path.name == PUBLISHED:
        shutil.copytree(spec_path.parent, dst, dirs_exist_ok=True)
        return Path(dst) / PUBLISHED
    return export(spec_path, dst, json.load(open(spec_path)).get("name") or spec_path.stem)


def export(spec_path, dst, name, parents=True):
    """Copy a spec into dst as <name>.json (+ <name>.npz holding its own arrays), without local bookkeeping.
    parents=True: the parent chain goes to dst/parent/ (see copy_parent), self-contained.
    parents=False (uploads): the parent is only referenced as parent/edit.json; it is already published, and the
    subnet puts the published chain there before building. Returns the new spec path."""
    src, dst = Path(spec_path), Path(dst)
    spec, arrays = json.load(open(src)), {}
    for op in spec["ops"]:
        for k in ("u", "cols", "rows"):
            ref = op.get(k)
            if isinstance(ref, str):
                path, _, key = ref.partition("#")
                new = key if key not in arrays else f"{Path(path).stem}_{key}"
                arrays[new] = _resolve(ref, src.parent)
                op[k] = f"{name}.npz#{new}"
    if spec.get("parent"):
        spec["parent"] = (os.path.relpath(copy_parent(src.parent / spec["parent"], dst / "parent"), dst) if parents
                          else f"parent/{PUBLISHED}")
    dst.mkdir(parents=True, exist_ok=True)
    if arrays:
        np.savez(dst / f"{name}.npz", **arrays)
    write_json(dst / f"{name}.json", {k: spec[k] for k in ("base", "parent", "ops") if k in spec})  # nothing else
    return dst / f"{name}.json"


def apply_ops(tensors, ops, base_dir):
    """tensors: name -> float32 torch tensor, modified in place."""
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
        apply_ops(work, s["ops"], d)
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


def dir_ablate_spec(directions, base, layers, topk, alpha, out_dir, name, random_control=False, seed=0, parent=None,
                    targets=("down_proj",)):
    """Spec ablating each layer's direction from its MLP down_proj, restricted to the top-k writer neurons, and/or
    from its attention o_proj (all columns: heads have no writer score).
    random_control: a random unit direction orthogonal to u instead (same layers, columns and alpha).
    parent: path of the spec this one stacks on, relative to out_dir."""
    d = np.load(directions)
    prefix = decoder_prefix(tensor_index(profiles.base_dir(base)))
    rng, arrays, ops = np.random.default_rng(seed), {}, []
    for l in layers:
        u = d["U"][l].astype(np.float32)
        if random_control:
            r = rng.standard_normal(u.shape).astype(np.float32)
            u = (r - (r @ u) * u) / np.linalg.norm(r - (r @ u) * u)
        arrays[f"u{l}"] = u
        if "o_proj" in targets:
            ops.append({"op": "dir_ablate", "tensor": f"{prefix}layers.{l}.self_attn.o_proj.weight",
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
        for k in ("u", "cols", "rows", "values"):
            v = op.get(k)
            if v is None or json.dumps(v) in seen:
                continue
            seen.add(json.dumps(v))
            a = _resolve(v, spec_path.parent)
            payload += a.nbytes if isinstance(v, str) else a.size * (4 if a.dtype.kind == "f" else 8)
    return len({op["tensor"] for op in spec["ops"]}), payload


def low_rank_spec(base, layers, matrix, keep, out_dir, name, parent=None):
    """Spec replacing each layer's MLP `matrix` (gate_proj / up_proj / down_proj) by its rank-k approximation."""
    prefix = decoder_prefix(tensor_index(profiles.base_dir(base)))
    spec = {"name": name, "base": base,
            "ops": [{"op": "low_rank", "tensor": f"{prefix}layers.{l}.mlp.{matrix}.weight", "keep": keep} for l in layers]}
    if parent:
        spec["parent"] = parent
    write_json(Path(out_dir) / f"{name}.json", spec)
    return Path(out_dir) / f"{name}.json"
