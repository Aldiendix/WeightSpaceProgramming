import json
import os

import numpy as np
import torch
from safetensors.torch import load_file

from wsp.edits import build, decoder_prefix, tensor_index

T = "model.layers.1.mlp.down_proj.weight"


def _spec(tmp_path, ops):
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"name": "e", "base": "qwen35-0.8b-think", "ops": ops}))
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
    assert on == "Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17" and carried == []
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
    assert (d / "adopted" / "champion" / "edit" / "edit.json").exists()
    for name, f in (("x", 0.5), ("y", 2.0)):   # unpublished edits of the same name stay apart, under the adopted name
        (tmp_path / name).mkdir()
        (tmp_path / name / "E1.json").write_text(json.dumps({"name": "E1", "base": "b", "ops": [
            {"op": "scale_cols", "tensor": T, "cols": [0], "factor": f}]}))
        C.cmd_adopt(argparse.Namespace(case="c", edit=str(tmp_path / name / "E1.json"), name=name))
    x, y = (build(d / "edits" / f"{n}.json", tmp_path / f"m_{n}", base_dir=tiny_model)[1] for n in "xy")
    assert x["tensors"][0]["edited_sha256"] != y["tensors"][0]["edited_sha256"]


def test_attention_output_of_hybrid_layers():
    from wsp.edits import attn_out
    idx = {f"m.layers.{l}.{n}.weight": "x" for l, n in ((0, "linear_attn.out_proj"), (0, "mlp.down_proj"),
                                                        (3, "self_attn.o_proj"), (3, "self_attn.q_proj"))}
    assert attn_out(idx, "m.", 0) == "m.layers.0.linear_attn.out_proj.weight"
    assert attn_out(idx, "m.", 3) == "m.layers.3.self_attn.o_proj.weight"


def test_lineage_through_the_protocol(tmp_path, tiny_model, monkeypatch):
    """Three generations of entries, the way a competition runs them. Every published edit lives as <sid>/edit/ next to
    what names it (storage, a task's bundle, a case, the subnet's build); an upload names its start by its sid, or the
    base by its full name, and builds the model the case built; local edits in between are carried."""
    import argparse
    import os
    import shutil

    import pytest
    from wsp import case as C
    from wsp import profiles
    from wsp.edits import copy_parent, export, export_upload, load_chain, start_dir
    P = "qwen35-0.8b-think"
    BASE = profiles.base_name(P)
    d0, d1 = (f"model.layers.{i}.mlp.down_proj.weight" for i in (0, 1))
    monkeypatch.setattr(C, "CASES", tmp_path / "cases")
    storage = tmp_path / "storage"  # sid -> the published edit/ directory

    def model(spec):
        _, m = build(spec, tmp_path / "m" / os.urandom(4).hex(), base_dir=tiny_model)
        return {t["name"]: t["edited_sha256"] for t in m["tensors"]}

    def place(sid, where):  # the subnet: a published edit at where/<sid>/edit/, its own chain inside it
        shutil.copytree(storage / sid / "edit", where / sid / "edit")
        ref = json.loads((where / sid / "edit" / "edit.json").read_text()).get("parent")
        if ref and ref != BASE:
            place(ref, where / sid / "edit")

    def init(case, start=None):  # wsp.case init: the start kept under its name, as the task keeps it
        d = tmp_path / "cases" / case
        (d / "edits").mkdir(parents=True)
        parent = copy_parent(start, d / start_dir(start)).relative_to(d).as_posix() if start else None
        (d / "case.json").write_text(json.dumps({"profile": P, "parent": parent}))
        return d, (os.path.relpath(d / parent, d / "edits") if parent else None)

    def edit(d, name, parent, ops, arrays=None):
        if arrays:
            np.savez(d / "edits" / f"{name}.npz", **arrays)
        (d / "edits" / f"{name}.json").write_text(json.dumps({"name": name, "base": P, "ops": ops,
                                                              **({"parent": parent} if parent else {})}))
        return d / "edits" / f"{name}.json"

    def admit(spec, sid, starts):  # upload, then build it as the subnet does; publish it under sid
        up, ref, carried = export_upload(spec, tmp_path / "up" / sid / "edit", starts=starts)
        assert json.loads(up.read_text())["parent"] == ref
        if ref != BASE:
            assert ref in starts
            place(ref, up.parent)
        assert model(up) == model(spec)  # the uploaded edit builds the model the case measured
        (up.parent / "REPORT.md").write_text("r")
        shutil.copytree(up.parent, storage / sid / "edit", ignore=shutil.ignore_patterns(*(starts or [])))
        return ref, carried

    # generation 1: two entries from the base
    d, _ = init("g1a")
    e = edit(d, "E", None, [{"op": "dir_ablate", "tensor": d0, "u": "E.npz#u", "alpha": 1.0}],
             {"u": np.array([1.0, 1.0, 0, 0], np.float32)})
    assert admit(e, "A", None) == (BASE, [])
    d, _ = init("g1c")
    assert admit(edit(d, "E", None, [{"op": "scale_cols", "tensor": d1, "cols": [5], "factor": 0.5}]), "C", None)[0] == BASE

    # generation 2: the entry's parent is A, the champion C; the agent starts from A with a local edit in between
    task = tmp_path / "task2"
    for sid in ("A", "C"):
        place(sid, task)
    monkeypatch.chdir(task / "A")  # a start given by a relative path keeps its name too
    assert start_dir("edit/edit.json") == "A"
    monkeypatch.chdir(task / "A" / "edit")
    assert start_dir("edit.json") == "A"
    d, start = init("g2", task / "A" / "edit" / "edit.json")
    assert start == "../A/edit/edit.json" and load_chain(d / "edits" / start)[0][0]["parent"] == BASE
    C.cmd_adopt(argparse.Namespace(case="g2", edit=str(task / "C" / "edit" / "edit.json"), name="champion"))
    assert model(d / "edits" / "champion.json") == model(task / "C" / "edit" / "edit.json")
    edit(d, "L", start, [{"op": "dir_ablate", "tensor": d1, "u": "L.npz#u", "alpha": 1.0}],
         {"u": np.array([0, 0, 1.0, 1.0], np.float32)})
    e = edit(d, "E", "L.json", [{"op": "scale_cols", "tensor": d1, "cols": [0], "factor": 0.5}])
    with pytest.raises(SystemExit, match="task offers C"):
        export_upload(e, tmp_path / "x", starts=["C"])  # a start this task does not offer
    assert export_upload(d / "edits" / "champion.json", tmp_path / "x", starts=["A", "C"])[1] == "C"  # by its sid
    shutil.copytree(task / "C" / "edit", tmp_path / "dl" / "champion" / "edit")  # a start kept under another name
    e2 = edit(d, "E2", os.path.relpath(tmp_path / "dl" / "champion" / "edit" / "edit.json", d / "edits"), [])
    with pytest.raises(SystemExit, match="--start"):
        export_upload(e2, tmp_path / "x", starts=["A", "C"])
    assert export_upload(e2, tmp_path / "x", start="C", starts=["A", "C"])[1] == "C"
    assert admit(e, "B", ["A", "C"]) == ("A", ["L"])
    assert json.loads((storage / "B" / "edit" / "edit.json").read_text())["parent"] == "A"
    assert not (storage / "B" / "edit" / "A").exists()  # the start is not uploaded

    # generation 3: the entry's parent is B (itself on A); the bundle holds B with its chain inside
    task = tmp_path / "task3"
    place("B", task)
    assert (task / "B" / "edit" / "A" / "edit" / "edit.json").exists()
    d, start = init("g3", task / "B" / "edit" / "edit.json")
    e = edit(d, "E", start, [{"op": "scale_cols", "tensor": d0, "cols": [3], "factor": 0.5}])
    assert len(load_chain(e)) == 3
    assert admit(e, "D", ["B"]) == ("B", [])
    up, ref, carried = export_upload(e, tmp_path / "flat", start="base")  # or the whole chain carried from the base
    assert ref == BASE and carried == ["A", "B"]
    assert [op["op"] for op in json.loads(up.read_text())["ops"]].count("round") == 3  # where A, the local L, B ended
    assert model(up) == model(e)
    out = export(e, tmp_path / "copy", "E")  # a self-contained copy names its published parent the same way
    assert json.loads(out.read_text())["parent"] == "B" and model(out) == model(e)

    legacy = tmp_path / "legacy"  # published before names: "parent" a path to parent/edit.json; still builds
    shutil.copytree(storage / "A" / "edit", legacy / "parent")
    (legacy / "edit.json").write_text(json.dumps({"base": P, "parent": "parent/edit.json", "ops": []}))
    assert model(legacy / "edit.json") == model(storage / "A" / "edit" / "edit.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"base": P, "parent": "Other/Model@abc", "ops": []}))
    with pytest.raises(ValueError, match="neither"):
        load_chain(bad)

    local = json.loads((tmp_path / "cases" / "g2" / "edits" / "L.json").read_text())  # a local edit named "edit"
    (tmp_path / "cases" / "g2" / "edits" / "edit.json").write_text(json.dumps({k: v for k, v in local.items()
                                                                              if k != "name"}))
    d, start = init("g4", tmp_path / "cases" / "g2" / "edits" / "edit.json")  # stays local: carried, not named
    up, ref, carried = export_upload(edit(d, "E", start, []), tmp_path / "u4")
    assert start == "../parent/edit.json" and ref == "A" and carried == ["edit"]


def test_flattened_chain_builds_the_chain_bit_for_bit(tmp_path, tiny_model):
    """Any chain carried into one spec (an upload carrying local edits, an archived champion) builds the chain's model
    bit for bit, for every op apply_ops runs: a new op without an example here fails this test."""
    from wsp.edits import OPS, flatten_chain, load_chain
    d0, d1 = (f"model.layers.{i}.mlp.down_proj.weight" for i in (0, 1))
    rng = np.random.default_rng(0)
    example = {
        "dir_ablate": {"op": "dir_ablate", "tensor": d0, "u": "L.npz#u", "alpha": 0.7, "cols": [1, 4]},
        "scale_cols": {"op": "scale_cols", "tensor": d1, "cols": [0, 2], "factor": 0.3},
        "scale_rows": {"op": "scale_rows", "tensor": d0, "rows": [1], "factor": 1.37},
        "set_rows": {"op": "set_rows", "tensor": d1, "rows": [3], "values": "L.npz#r"},
        "set_cols": {"op": "set_cols", "tensor": d0, "cols": [5], "values": "L.npz#c"},
        "low_rank": {"op": "low_rank", "tensor": d1, "keep": 0.5},
        "round": {"op": "round", "tensor": d0},
    }
    assert set(example) == set(OPS)
    links = [list(example.values()), [example[k] for k in reversed(list(example))],
             [{**example["scale_cols"], "factor": 0.7}, {**example["scale_rows"], "factor": 0.61}]]
    prev = None
    for i, ops in enumerate(links):  # three links, each with its own arrays under the same file and key names
        d = tmp_path / f"l{i}"
        d.mkdir()
        np.savez(d / "L.npz", u=rng.standard_normal(4).astype(np.float32),
                 r=rng.standard_normal((1, 6)).astype(np.float32), c=rng.standard_normal((4, 1)).astype(np.float32))
        spec = {"name": f"l{i}", "base": "qwen35-0.8b-think", "ops": ops}
        if prev:
            spec["parent"] = os.path.relpath(prev, d)
        (d / "e.json").write_text(json.dumps(spec))
        prev = d / "e.json"
    assert len(load_chain(prev)) == 3
    flat, arrays = flatten_chain(prev)
    out = tmp_path / "flat"
    out.mkdir()
    np.savez(out / "edit.npz", **arrays)
    (out / "edit.json").write_text(json.dumps(flat))
    assert len(arrays) == 6 and len(load_chain(out / "edit.json")) == 1  # 2 links x 3 arrays, kept apart; on the base

    def model(spec):
        _, m = build(spec, tmp_path / "m" / os.urandom(4).hex(), base_dir=tiny_model)
        return {t["name"]: t["edited_sha256"] for t in m["tensors"]}
    assert model(out / "edit.json") == model(prev)
    unrounded = {**flat, "ops": [op for op in flat["ops"] if op["op"] != "round"]}  # the rounds are what makes it so
    (out / "edit.json").write_text(json.dumps(unrounded))
    assert model(out / "edit.json") != model(prev)
