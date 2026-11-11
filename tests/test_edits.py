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


def test_stacked_edit_equals_sequential_application(tmp_path, tiny_model):
    from wsp.edits import export, load_chain, spec_size
    (tmp_path / "p").mkdir()
    (tmp_path / "c").mkdir()
    np.savez(tmp_path / "p" / "p.npz", u=np.array([1.0, 1.0, 0, 0], np.float32))
    (tmp_path / "p" / "p.json").write_text(json.dumps({"name": "p", "base": "b", "ops": [
        {"op": "dir_ablate", "tensor": T, "u": "p.npz#u", "alpha": 1.0}]}))
    (tmp_path / "c" / "c.json").write_text(json.dumps({"name": "c", "base": "b", "parent": "../p/p.json", "ops": [
        {"op": "scale_cols", "tensor": T, "cols": [0], "factor": 0.5}]}))
    _, m_chain = build(tmp_path / "c" / "c.json", tmp_path / "m1", base_dir=tiny_model)
    parent_model, m_parent = build(tmp_path / "p" / "p.json", tmp_path / "mp", base_dir=tiny_model)
    child_only = _spec(tmp_path, [{"op": "scale_cols", "tensor": T, "cols": [0], "factor": 0.5}])
    _, m_seq = build(child_only, tmp_path / "m2", base_dir=parent_model)  # child ops on the exact parent weights
    assert m_chain["tensors"][0]["edited_sha256"] == m_seq["tensors"][0]["edited_sha256"]
    t = m_chain["tensors"][0]
    assert m_chain["own_tensors"] == [T] and 0 < t["relative_change_vs_parent"] < t["relative_change"]
    assert abs(t["relative_change_vs_parent"] - m_seq["tensors"][0]["relative_change"]) < 1e-6
    assert m_parent["tensors"][0]["relative_change_vs_parent"] == m_parent["tensors"][0]["relative_change"]
    assert m_chain["chain"] == ["p", "c"] and len(load_chain(tmp_path / "c" / "c.json")) == 2
    assert spec_size(tmp_path / "c" / "c.json") == (1, 8)  # own ops only: one inline int list
    out = export(tmp_path / "c" / "c.json", tmp_path / "exp", "edit")  # self-contained copy builds identically
    _, m_exp = build(out, tmp_path / "m3", base_dir=tiny_model)
    assert m_exp["tensors"][0]["edited_sha256"] == m_chain["tensors"][0]["edited_sha256"]
    assert (tmp_path / "exp" / "parent" / "p.json").exists()


def test_chain_rejects_mixed_bases(tmp_path):
    import pytest
    from wsp.edits import load_chain
    (tmp_path / "p.json").write_text(json.dumps({"name": "p", "base": "a", "ops": []}))
    (tmp_path / "c.json").write_text(json.dumps({"name": "c", "base": "b", "parent": "p.json", "ops": []}))
    with pytest.raises(ValueError):
        load_chain(tmp_path / "c.json")


def test_low_rank_keeps_top_singular_components(tmp_path, tiny_model):
    spec = _spec(tmp_path, [{"op": "low_rank", "tensor": T, "keep": 0.25}])  # 4x6 -> rank 1
    out, _ = build(spec, tmp_path / "m", base_dir=tiny_model)
    W = load_file(str(out / "model.safetensors"))[T].float()
    S = torch.linalg.svdvals(W)
    assert S[1] / S[0] < 0.02  # rank 1 up to bf16 rounding
