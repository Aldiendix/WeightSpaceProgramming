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


def test_built_model_follows_its_sources(tmp_path, tiny_model, monkeypatch):
    """model_path builds a model whole, reuses it while spec and arrays are unchanged, and rebuilds when either changes
    or when a directory holds no record of what it was built from (an older or half-written build)."""
    from wsp import case, profiles
    monkeypatch.setattr(case, "CASES", tmp_path / "cases")
    monkeypatch.setattr(case, "MODELS", tmp_path / "models")
    monkeypatch.setattr(profiles, "base_dir", lambda name: tiny_model)
    d = tmp_path / "cases" / "c" / "edits"
    d.mkdir(parents=True)
    (tmp_path / "cases" / "c" / "case.json").write_text(json.dumps({"profile": "tiny"}))
    (d / "E.json").write_text(json.dumps({"base": "tiny", "ops": [
        {"op": "dir_ablate", "tensor": T, "u": "E.npz#u", "alpha": 1.0}]}))

    def weights():
        return load_file(str(tmp_path / "models" / "c__E" / "model.safetensors"))[T]
    np.savez(d / "E.npz", u=np.array([1.0, 0, 0, 0], np.float32))
    p = case.model_path("c", "E")
    w1, stamp = weights(), (tmp_path / "models" / "c__E" / "wsp_sources.sha256").stat().st_mtime_ns
    assert case.model_path("c", "E") == p and (tmp_path / "models" / "c__E" / "wsp_sources.sha256").stat().st_mtime_ns == stamp
    np.savez(d / "E.npz", u=np.array([0, 1.0, 0, 0], np.float32))  # same json, new arrays
    case.model_path("c", "E")
    assert not torch.equal(weights(), w1)
    (tmp_path / "models" / "c__E" / "wsp_sources.sha256").unlink()  # a build that never finished
    case.model_path("c", "E")
    assert (tmp_path / "models" / "c__E" / "wsp_sources.sha256").exists()
    assert not list((tmp_path / "models").glob(".c__E*"))  # no private directories left behind


def test_export_carries_every_array_an_op_reads(tmp_path, tiny_model):
    """An uploaded spec builds on its own: arrays referenced by any op field (values included) go into its npz, once
    per distinct reference."""
    from wsp.edits import export_upload, spec_size
    d0, d1 = "model.layers.0.mlp.down_proj.weight", T  # 4 x 6
    rng = np.random.default_rng(0)
    np.savez(tmp_path / "E1.npz", w=rng.standard_normal((1, 6)).astype(np.float32),
             down=rng.standard_normal((4, 1)).astype(np.float32), u=np.ones(4, np.float32))
    spec = _spec(tmp_path, [{"op": "set_rows", "tensor": d0, "rows": [3], "values": "E1.npz#w"},
                            {"op": "set_rows", "tensor": d1, "rows": [2], "values": "E1.npz#w"},
                            {"op": "set_cols", "tensor": d1, "cols": [3], "values": "E1.npz#down"},
                            {"op": "dir_ablate", "tensor": d0, "u": "E1.npz#u", "alpha": 1.0}])
    out, on, carried = export_upload(spec, tmp_path / "up" / "edit")
    assert on == "the base" and carried == []
    refs = [v for op in json.loads(out.read_text())["ops"] for k, v in op.items() if isinstance(v, str) and "#" in v]
    assert refs and all(r.startswith("edit.npz#") for r in refs)
    assert sorted(np.load(tmp_path / "up" / "edit" / "edit.npz").files) == ["down", "u", "w"]
    assert spec_size(out) == spec_size(spec)
    _, m1 = build(spec, tmp_path / "m1", base_dir=tiny_model)
    _, m2 = build(out, tmp_path / "m2", base_dir=tiny_model)
    assert [t["edited_sha256"] for t in m1["tensors"]] == [t["edited_sha256"] for t in m2["tensors"]]


def test_adopted_edit_builds_the_published_model(tmp_path, tiny_model, monkeypatch):
    """`adopt` adds a published edit (e.g. the champion) to a case as a model with no ops of its own: building it gives
    exactly the published edit's weights."""
    import argparse

    from wsp import case as C
    from wsp import profiles
    pub = tmp_path / "task" / "champion" / "edit"
    pub.mkdir(parents=True)
    np.savez(pub / "edit.npz", u=np.array([1.0, 0, 0, 0], np.float32))
    (pub / "edit.json").write_text(json.dumps({"base": "b", "ops": [
        {"op": "dir_ablate", "tensor": T, "u": "edit.npz#u", "alpha": 1.0}]}))
    d = tmp_path / "cases" / "c"
    (d / "edits").mkdir(parents=True)
    (d / "case.json").write_text(json.dumps({"profile": "b", "parent": None}))
    monkeypatch.setattr(C, "CASES", tmp_path / "cases")
    monkeypatch.setattr(profiles, "base_dir", lambda name: tiny_model)
    C.cmd_adopt(argparse.Namespace(case="c", edit=str(pub / "edit.json"), name="champion"))
    _, a = build(d / "edits" / "champion.json", tmp_path / "m1", base_dir=tiny_model)
    _, b = build(pub / "edit.json", tmp_path / "m2", base_dir=tiny_model)
    assert [t["edited_sha256"] for t in a["tensors"]] == [t["edited_sha256"] for t in b["tensors"]]


def test_attention_output_of_hybrid_layers():
    from wsp.edits import attn_out
    idx = {f"m.layers.{l}.{n}.weight": "x" for l, n in ((0, "linear_attn.out_proj"), (0, "mlp.down_proj"),
                                                        (3, "self_attn.o_proj"), (3, "self_attn.q_proj"))}
    assert attn_out(idx, "m.", 0) == "m.layers.0.linear_attn.out_proj.weight"
    assert attn_out(idx, "m.", 3) == "m.layers.3.self_attn.o_proj.weight"


def test_upload_builds_the_case_model_from_what_the_protocol_references(tmp_path, tiny_model):
    """Local edits a spec stacks on are carried into the upload; a published start is referenced by the task directory
    it came from (named, or given by start); start="base" carries the whole chain. Each upload builds the model the
    case built (here bit for bit: the only tensor two links edit is then scaled by 0.5, which rounds exactly)."""
    import os

    import pytest
    from wsp.edits import export_upload
    d0, d1 = (f"model.layers.{i}.mlp.down_proj.weight" for i in (0, 1))
    pub = tmp_path / "task" / "champion" / "edit"  # published start: base <- p
    pub.mkdir(parents=True)
    np.savez(pub / "edit.npz", u=np.array([1.0, 1.0, 0, 0], np.float32))
    (pub / "edit.json").write_text(json.dumps({"base": "b", "ops": [
        {"op": "dir_ablate", "tensor": d0, "u": "edit.npz#u", "alpha": 1.0}]}))

    def case(start_dir, name):
        import shutil
        c = tmp_path / name
        shutil.copytree(pub, c / start_dir)
        (c / "edits").mkdir()
        np.savez(c / "edits" / "edit.npz", u=np.array([0, 0, 1.0, 1.0], np.float32))  # same file name as the start's
        (c / "edits" / "L.json").write_text(json.dumps({"name": "L", "base": "b", "parent": f"../{start_dir}/edit.json",
                                                        "ops": [{"op": "dir_ablate", "tensor": d1, "u": "edit.npz#u",
                                                                 "alpha": 1.0}]}))  # a local edit on the start
        (c / "edits" / "E.json").write_text(json.dumps({"name": "E", "base": "b", "parent": "L.json", "ops": [
            {"op": "scale_cols", "tensor": d1, "cols": [0], "factor": 0.5}]}))
        return c / "edits" / "E.json"

    def same_model(spec, up, place=None):
        if place:  # the subnet places the published start where the upload points
            import shutil
            shutil.copytree(pub, up.parent / place)
        _, m1 = build(spec, tmp_path / "m" / os.urandom(4).hex(), base_dir=tiny_model)
        _, m2 = build(up, tmp_path / "m" / os.urandom(4).hex(), base_dir=tiny_model)
        return {t["name"]: t["edited_sha256"] for t in m1["tensors"]} == \
            {t["name"]: t["edited_sha256"] for t in m2["tensors"]}

    spec = case("champion", "c1")  # kept under the task's name: referenced, the local edit carried
    up, on, carried = export_upload(spec, tmp_path / "u1")
    s = json.loads(up.read_text())
    assert on == "champion/edit.json" == s["parent"] and carried == ["L"] and len(s["ops"]) == 2
    assert same_model(spec, up, "champion")

    spec = case("0009538688-0019", "c2")  # kept under another name: which task directory it is must be said
    with pytest.raises(SystemExit, match="--start"):
        export_upload(spec, tmp_path / "u2")
    up, on, _ = export_upload(spec, tmp_path / "u2", start="champion")
    assert on == "champion/edit.json" and same_model(spec, up, "champion")
    up, on, carried = export_upload(spec, tmp_path / "u3", start="base")  # or carry the whole chain from the base
    s = json.loads(up.read_text())
    assert on == "the base" and "parent" not in s and len(s["ops"]) == 3 and carried == ["edit", "L"]
    assert len(np.load(tmp_path / "u3" / "edit.npz").files) == 2  # two files' "edit.npz#u": two arrays
    assert same_model(spec, up)
    with pytest.raises(SystemExit):
        export_upload(tmp_path / "c1" / "champion" / "edit.json", tmp_path / "u4", start="parent")  # no start

    from wsp.edits import copy_parent  # a local edit named "edit" is still local when another case stacks on it
    local = json.loads((tmp_path / "c1" / "edits" / "L.json").read_text())
    (tmp_path / "c1" / "edits" / "edit.json").write_text(json.dumps({k: v for k, v in local.items() if k != "name"}))
    start = copy_parent(tmp_path / "c1" / "edits" / "edit.json", tmp_path / "c3" / "parent")
    (tmp_path / "c3" / "edits").mkdir()
    (tmp_path / "c3" / "edits" / "E.json").write_text(json.dumps({"name": "E", "base": "b", "ops": [],
                                                                  "parent": os.path.relpath(start, tmp_path / "c3" / "edits")}))
    up, on, carried = export_upload(tmp_path / "c3" / "edits" / "E.json", tmp_path / "u5")
    assert start.name == "edit_local.json" and on == "champion/edit.json" and carried == ["edit_local"]
