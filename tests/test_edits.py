import json

import numpy as np
import torch
from safetensors.torch import load_file

from wsp.edits import build, decoder_prefix, tensor_index

T = "model.layers.1.mlp.down_proj.weight"


def _spec(tmp_path, ops):
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"name": "e", "base": "unused", "ops": ops}))
    return p


def test_prefix_skips_auxiliary_stacks(tiny_model):
    assert decoder_prefix(tensor_index(tiny_model)) == "model."


def test_dir_ablate_removes_direction_on_selected_columns(tmp_path, tiny_model):
    u = [1.0, 0.0, 0.0, 0.0]
    spec = _spec(tmp_path, [{"op": "dir_ablate", "tensor": T, "u": u, "alpha": 1.0, "cols": [0, 2]}])
    out, man = build(spec, tmp_path / "m", tmp_path / "diff", base_dir=tiny_model)
    W0, W1 = load_file(str(tiny_model / "model.safetensors"))[T].float(), load_file(str(out / "model.safetensors"))[T].float()
    assert torch.allclose(W1[0, [0, 2]], torch.zeros(2), atol=1e-2)       # projection on u removed there
    assert torch.equal(W1[:, [1, 3, 4, 5]], W0[:, [1, 3, 4, 5]])           # other columns untouched
    assert torch.equal(W1[1:], W0[1:])                                     # orthogonal rows untouched
    t = man["tensors"][0]
    assert t["changed_cols"] <= 2 and t["base_norm"] > 0 and (tmp_path / "diff" / "diff.npz").exists()
    assert (out / "config.json").is_symlink()


def test_build_is_bit_exact_and_npz_refs(tmp_path, tiny_model):
    np.savez(tmp_path / "a.npz", u=np.ones(4, np.float32), cols=np.array([1, 5]))
    spec = _spec(tmp_path, [{"op": "dir_ablate", "tensor": T, "u": "a.npz#u", "cols": "a.npz#cols", "alpha": 3.0},
                            {"op": "scale_cols", "tensor": "model.layers.0.mlp.down_proj.weight", "cols": [2],
                             "factor": 0.0}])
    _, m1 = build(spec, tmp_path / "m1", base_dir=tiny_model)
    _, m2 = build(spec, tmp_path / "m2", base_dir=tiny_model)
    assert [t["edited_sha256"] for t in m1["tensors"]] == [t["edited_sha256"] for t in m2["tensors"]]
    assert m1["changed_entries_total"] > 0
