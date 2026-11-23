import json

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")


@pytest.fixture
def tiny_lm(tmp_path):
    cfg = transformers.Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                                   initializer_range=0.5)  # large weights: gains visibly change the outputs
    torch.manual_seed(0)
    m = transformers.Qwen2ForCausalLM(cfg).eval()
    d = tmp_path / "base"
    m.save_pretrained(d, safe_serialization=True)
    return m, d


def test_fit_lowers_the_loss_and_its_spec_builds_the_fitted_model(tmp_path, tiny_lm, monkeypatch):
    from wsp import case as C
    from wsp import fit, profiles
    from wsp.edits import build

    model, base = tiny_lm
    tgts = fit.targets(model, [0, 1], fit.DEFAULT_MODULES)
    assert {(t, w) for t, _, w in tgts} == {(f"layers.{l}.{n}.weight", w) for l in (0, 1)
                                          for n, w in (("mlp.down_proj", 1), ("self_attn.o_proj", 4))}
    rng = np.random.default_rng(0)
    ex = [{"kind": "nll", "w": 1.0, "ids": (rng.integers(0, 8, 6).tolist() * 4), "start": 6, "stop": 24}
          for _ in range(4)]  # repeats: learnable

    def mean_loss(vals=None):
        g = fit.Gains(tgts, "cpu")
        for t, (p, _) in g.params.items():
            if vals:
                p.data = torch.tensor(vals[t], dtype=torch.float32)
        with torch.no_grad():
            out = float(np.mean([float(-fit._Head(model).logp(e)[0].mean()) for e in ex]))
        g.remove()
        return out
    before = mean_loss()
    params = fit.Gains(tgts, "cpu")
    fit.train(model, ex, params, lr=5e-2, epochs=8, accum=2, log_every=100)
    gains = params.values()
    params.remove()
    assert mean_loss(gains) < before - 0.05 and mean_loss() == before  # fitted lower; hooks removed afterwards

    # the spec, built like any edit, is the model with the gains (few distinct values: quantisation is exact)
    vals = {t: rng.choice([0.9, 1.0, 1.15], size=g.shape) for t, g in gains.items()}
    d = tmp_path / "cases" / "c"
    (d / "edits").mkdir(parents=True)
    (d / "case.json").write_text(json.dumps({"profile": "tiny", "parent": None}))
    monkeypatch.setattr(C, "CASES", tmp_path / "cases")
    monkeypatch.setattr(profiles, "base_dir", lambda name: base)
    meta = {"params": "gains", "widths": {t: w for t, _, w in tgts}}
    spec = fit.write_spec("c", "F", vals, meta, levels=8)
    assert all(op["op"] == "scale_cols" for op in json.loads(spec.read_text())["ops"])
    out, _ = build(spec, tmp_path / "built", base_dir=base)
    edited = transformers.Qwen2ForCausalLM.from_pretrained(out).eval()
    ids = torch.tensor([ex[0]["ids"]])
    g = fit.Gains(tgts, "cpu")
    for t, (p, _) in g.params.items():
        p.data = torch.tensor(vals[t], dtype=torch.float32)
    with torch.no_grad():
        want, got = model(ids).logits, edited(ids).logits
    g.remove()
    assert torch.allclose(want, got, atol=1e-5)

    shuffled = fit.write_spec("c", "F_shuf", vals, meta, levels=8, shuffle=True)
    a, b = (json.loads(p.read_text())["ops"] for p in (spec, shuffled))
    assert sorted(o["factor"] for o in a) == sorted(o["factor"] for o in b)  # same values, other units


def test_quantisation_keeps_strongly_moved_units():
    from wsp.fit import _levels
    g = 1 + 0.005 * np.random.default_rng(0).standard_t(2, 9728)
    moved = np.where(np.abs(g - 1) >= 0.002)[0]
    q = g.copy()
    for grp in _levels(g, moved, 256):
        q[grp] = g[grp].mean()
    assert np.abs(q[moved] - g[moved]).max() < 0.01 and np.linalg.norm(q - g) / np.linalg.norm(g[moved] - 1) < 0.03


def test_svd_spec_builds_the_fitted_model_and_pairs_move_toward_the_winner(tmp_path, tiny_lm, monkeypatch):
    """Singular-value scales: the spec of dir_ablate ops builds the hooked model; a DPO pair raises the passing
    continuation against the failing one, and an anchor's KL stays near 0 while the pair moves."""
    from wsp import case as C
    from wsp import fit, profiles
    from wsp.edits import build

    model, base = tiny_lm
    tgts = fit.targets(model, [0, 1], fit.DEFAULT_MODULES)
    rng = np.random.default_rng(1)
    p = rng.integers(0, 64, 5).tolist()
    win, lose = {"ids": p + [3, 4, 5, 6], "start": 5, "stop": 9}, {"ids": p + [7, 8, 9, 10], "start": 5, "stop": 9}
    anchor = {"kind": "anchor", "ids": rng.integers(0, 64, 12).tolist(), "start": 4, "stop": 12}
    ex = [{"kind": "pair", "win": win, "lose": lose}, anchor]
    params = fit.SVF(tgts, "cpu", k=3, zmax=0.5)
    H = fit._Head(model)

    def margin():
        with torch.no_grad():
            return float(H.logp(win)[0].sum() - H.logp(lose)[0].sum())
    before = margin()
    fit.train(model, ex, params, lr=5e-2, epochs=20, accum=2, beta=1.0, nll=0.0, log_every=100)
    assert margin() > before + 0.1
    vals = params.values()
    assert all(np.abs(v).max() <= 0.5 for k, v in vals.items() if not k.startswith("U|"))

    d = tmp_path / "cases" / "c"
    (d / "edits").mkdir(parents=True)
    (d / "case.json").write_text(json.dumps({"profile": "tiny", "parent": None}))
    monkeypatch.setattr(C, "CASES", tmp_path / "cases")
    monkeypatch.setattr(profiles, "base_dir", lambda name: base)
    spec = fit.write_spec("c", "S", vals, {"params": "svd"}, min_change=0.0)
    assert {op["op"] for op in json.loads(spec.read_text())["ops"]} == {"dir_ablate"}
    out, _ = build(spec, tmp_path / "built", base_dir=base)
    edited = transformers.Qwen2ForCausalLM.from_pretrained(out).eval()
    ids = torch.tensor([win["ids"]])
    with torch.no_grad():
        want, got = model(ids).logits, edited(ids).logits
    assert torch.allclose(want, got, atol=1e-3)
    params.remove()
    with torch.no_grad():
        assert not torch.allclose(model(ids).logits, got, atol=1e-3)  # hooks removed: the starting model again
    rev = fit.write_spec("c", "S_rev", vals, {"params": "svd"}, dose=-1.0, min_change=0.0)
    a, b = (json.loads(q.read_text())["ops"] for q in (spec, rev))
    assert [o["alpha"] for o in a] == [-o["alpha"] for o in b]  # a negative dose reverses every scale


def test_examples_by_objective(tmp_path, monkeypatch):
    """sft keeps passing samples, rloo weights them by advantage, pairs match passing with failing samples of one
    start (a branch run: the same thinking), anchors keep their sources out of the objective."""
    from wsp import case as C
    from wsp import data, fit

    items = {f"m{i}": {"id": f"m{i}", "source": "math", "task": "math", "problem": f"q{i}", "answer": "1"}
             for i in range(3)}
    items["i0"] = {"id": "i0", "source": "ifsynth", "task": "math", "problem": "q", "answer": "1"}
    monkeypatch.setattr(data, "by_id", lambda: items)
    monkeypatch.setattr(C, "CASES", tmp_path / "cases")
    monkeypatch.setattr(C, "RUNS", tmp_path / "runs")
    (tmp_path / "cases" / "c" / "sets").mkdir(parents=True)
    (tmp_path / "cases" / "c" / "sets" / "dev.json").write_text(json.dumps(["m2"]))
    (tmp_path / "cases" / "c" / "sets" / "fit.json").write_text(json.dumps(["m0", "m1", "i0"]))
    row = lambda i, k, ok, **kw: {"id": i, "k": k, "source": items[i]["source"], "correct": ok, "cut": False,  # noqa: E731
                                  "text": f"<think>t{k}</think> a{k}", **kw}
    rows = [row("m0", 0, True), row("m0", 1, False), row("m0", 2, False), row("m1", 0, True), row("m1", 1, True),
            row("m2", 0, True), row("m2", 1, False), row("i0", 0, True)]
    (tmp_path / "runs" / "c").mkdir(parents=True)
    (tmp_path / "runs" / "c" / "base__fit.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    br = [row("m0", b, b == 0, from_k=0, prefix_chars=len("<think>t0</think>")) for b in range(3)]
    for r in br:
        r["text"] = "<think>t0</think>" + f" a{r['k']}"
    (tmp_path / "runs" / "c" / "base__fit.branches.jsonl").write_text("\n".join(json.dumps(r) for r in br))

    class Tok:
        all_special_ids, eos_token = [], "E"

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True):
            return "P:" + msgs[0]["content"]

        def __call__(self, text, add_special_tokens=True):
            return type("T", (), {"input_ids": [ord(ch) for ch in text]})()
    prof = {"thinking": True}
    ex, n = fit.examples("c", ["base__fit"], Tok(), prof, "sft", seed=0)
    assert {e["id"] for e in ex} == {"m0", "m1", "i0"} and n["held_out"] == 2  # dev problem m2 never enters
    (tmp_path / "runs" / "c" / "base__dev.jsonl").write_text(json.dumps(row("m2", 0, True)))
    assert fit.examples("c", ["base__dev"], Tok(), prof, "sft")[1]["held_out"] == 1  # not even from a run on dev
    ex, _ = fit.examples("c", ["base__fit"], Tok(), prof, "rloo", anchor=["ifsynth"])
    assert sorted(round(e["w"], 2) for e in ex if e["kind"] == "nll") == [-0.5, -0.5, 1.0]  # m1: all pass, no signal
    assert [e["id"] for e in ex if e["kind"] == "anchor"] == ["i0"]
    ex, _ = fit.examples("c", ["base__fit"], Tok(), prof, "rloo", credit="answer")
    e = ex[0]
    assert "".join(map(chr, e["ids"][e["start"]:e["stop"]])).startswith(" a")  # after the thinking only
    ex, _ = fit.examples("c", ["base__fit.branches"], Tok(), prof, "pairs", per_problem=5)
    assert len(ex) == 1 and ex[0]["kind"] == "pair"  # one passing branch: one pair
    w = ex[0]["win"]
    assert "".join(map(chr, w["ids"][w["start"]:w["stop"]])) == " a0E"  # credit after the shared thinking
