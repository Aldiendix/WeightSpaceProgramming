import json

import pytest
import torch
from safetensors.torch import save_file


@pytest.fixture
def tiny_model(tmp_path):
    """A 2-layer 'model' with an MTP layer and embeddings, enough for edit builds."""
    g = torch.Generator().manual_seed(0)
    t = {f"model.layers.{l}.mlp.down_proj.weight": torch.randn(4, 6, generator=g).to(torch.bfloat16) for l in range(2)}
    t["mtp.layers.0.mlp.down_proj.weight"] = torch.randn(4, 6, generator=g).to(torch.bfloat16)
    t["model.embed_tokens.weight"] = torch.randn(10, 4, generator=g).to(torch.bfloat16)
    d = tmp_path / "base"
    d.mkdir()
    save_file(t, str(d / "model.safetensors"))
    (d / "config.json").write_text(json.dumps({"model_type": "tiny"}))
    return d
