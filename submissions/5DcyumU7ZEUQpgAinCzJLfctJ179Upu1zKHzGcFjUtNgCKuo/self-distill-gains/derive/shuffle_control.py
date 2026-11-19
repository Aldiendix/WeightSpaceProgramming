"""Control for a gains edit: the same gain values, randomly re-assigned to other units / heads of the same layer.

  python derive/shuffle_control.py --case C --edit NAME [--seed 0]   -> cases/<case>/edits/SHUF_NAME.json/.npz

Every scale_rows op keeps its factor and its number of rows, but the rows are drawn at random (without replacement,
per tensor) from the tensor's units; scale_cols head ops are permuted among the heads of the tensor. Same payload,
same magnitude, same layers: if the control scores like the parent while the edit gains, the effect comes from which
units were up- or down-weighted, not from perturbing the network.
"""
import argparse
import collections
import json

import numpy as np

from wsp import case as C
from wsp import profiles
from wsp.common import write_json
from wsp.edits import tensor_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--edit", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = C.cdir(a.case) / "edits"
    spec = json.load(open(d / f"{a.edit}.json"))
    arr = np.load(d / f"{a.edit}.npz")
    rng = np.random.default_rng(a.seed)
    name = f"SHUF_{a.edit}"
    from safetensors import safe_open  # noqa: PLC0415
    base = profiles.base_dir(spec["base"])
    idx = tensor_index(base)

    def shape(t):
        with safe_open(str(base / idx[t]), "pt") as f:
            return f.get_slice(t).get_shape()

    by_tensor = collections.defaultdict(list)
    for op in spec["ops"]:
        by_tensor[op["tensor"]].append(op)
    out_arrays, ops = {}, []
    for t, tops in by_tensor.items():
        key = "rows" if tops[0]["op"] == "scale_rows" else "cols"
        n = shape(t)[0 if key == "rows" else 1]
        sizes = [len(arr[op[key].split("#")[1]]) for op in tops]
        if key == "rows":
            perm = rng.permutation(n)
            pos = 0
            for op, k in zip(tops, sizes):
                kname = f"{op[key].split('#')[1]}"
                out_arrays[kname] = np.sort(perm[pos:pos + k]).astype(np.int64)
                pos += k
                ops.append(dict(op, **{key: f"{name}.npz#{kname}"}))
        else:  # head blocks: permute whole blocks
            width = sizes[0]
            nblocks = n // width
            blocks = rng.permutation(nblocks)[: len(tops)]
            for op, b in zip(tops, blocks):
                kname = f"{op[key].split('#')[1]}"
                out_arrays[kname] = np.arange(b * width, (b + 1) * width, dtype=np.int64)
                ops.append(dict(op, **{key: f"{name}.npz#{kname}"}))
    np.savez(d / f"{name}.npz", **out_arrays)
    write_json(d / f"{name}.json", dict(spec, name=name, ops=ops))
    print(d / f"{name}.json", len(ops), "ops")


if __name__ == "__main__":
    main()
