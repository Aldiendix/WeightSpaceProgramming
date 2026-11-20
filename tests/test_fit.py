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
    ex = [{"ids": (rng.integers(0, 8, 6).tolist() * 4), "start": 6} for _ in range(4)]  # repeats: learnable

    def mean_loss(vals=None):
        g = fit.Gains(tgts, "cpu")
        for t, (p, _) in g.params.items():
            if vals:
                p.data = torch.tensor(vals[t], dtype=torch.float32)
        with torch.no_grad():
            out = float(np.mean([float(fit._loss(model, e)) for e in ex]))
        g.remove()
        return out
    before = mean_loss()
    gains, log = fit.train(model, ex, tgts, lr=5e-2, epochs=8, accum=2, device="cpu", log_every=100)
    assert mean_loss(gains) < before - 0.05 and mean_loss() == before  # fitted lower; hooks removed afterwards

    # the spec, built like any edit, is the model with the gains (few distinct values: quantisation is exact)
    vals = {t: rng.choice([0.9, 1.0, 1.15], size=g.shape) for t, g in gains.items()}
    d = tmp_path / "cases" / "c"
    (d / "edits").mkdir(parents=True)
    (d / "case.json").write_text(json.dumps({"profile": "tiny", "parent": None}))
    monkeypatch.setattr(C, "CASES", tmp_path / "cases")
    monkeypatch.setattr(profiles, "base_dir", lambda name: base)
    spec = fit.write_spec("c", "F", vals, {t: w for t, _, w in tgts}, levels=8)
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

    shuffled = fit.write_spec("c", "F_shuf", vals, {t: w for t, _, w in tgts}, levels=8, shuffle=True)
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
