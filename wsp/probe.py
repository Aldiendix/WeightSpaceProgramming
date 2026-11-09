"""Teacher-forced reads with HF transformers: per layer, the residual stream after the layer and the MLP neuron
activations (input of down_proj), averaged over a marked segment of each text, plus the segment's mean log-prob.

A text is {"text": str, "seg": [char_start, char_end]}; tokens starting inside the segment are averaged.
"""
import numpy as np


def load_model(model, device="cuda"):
    if device == "cpu":  # flash-linear-attention kernels are GPU-only: use the PyTorch reference path
        import sys  # noqa: PLC0415
        sys.modules.setdefault("fla", None)
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    try:
        m = AutoModelForCausalLM.from_pretrained(str(model), dtype=torch.bfloat16)
    except (ValueError, KeyError):  # multimodal checkpoints
        from transformers import AutoModelForImageTextToText  # noqa: PLC0415
        m = AutoModelForImageTextToText.from_pretrained(str(model), dtype=torch.bfloat16)
    return m.to(device).eval(), AutoTokenizer.from_pretrained(str(model))


def decoder(model):
    """(decoder module that owns `layers` and `norm`, its layer list): the shortest-named ModuleList with
    num_hidden_layers entries."""
    import torch  # noqa: PLC0415

    n = model.config.get_text_config().num_hidden_layers
    name = min((k for k, m in model.named_modules() if k.endswith("layers") and isinstance(m, torch.nn.ModuleList)
                and len(m) == n), key=len)
    owner = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
    return owner, owner.get_submodule(name.rsplit(".", 1)[-1])


def _child(module, name):
    return next(m for k, m in module.named_modules() if k.split(".")[-1] == name)


class Probe:
    def __init__(self, model, device="cuda"):
        self.model, self.tok = load_model(model, device)
        owner, self.layers = decoder(self.model)
        self.norm, self.head, self.device = owner.norm, self.model.get_output_embeddings(), device

    def down_proj(self, layer):
        return _child(self.layers[layer], "down_proj").weight.detach().float().cpu().numpy()

    def read(self, items, want=("resid", "mlp", "logp"), max_tokens=16384):
        """-> {"resid": [N, L, d], "mlp": [N, L, inter], "logp": [N], "ntok": [N]} (only the requested keys)."""
        import torch  # noqa: PLC0415

        out = {k: [] for k in (*want, "ntok")}
        for it in items:
            enc = self.tok(it["text"], return_offsets_mapping=True, add_special_tokens=False)
            ids = enc["input_ids"][-max_tokens:]  # keep the end: the segment is at the end of the text
            s, e = it["seg"]
            mask = torch.tensor([s <= a < e for a, _ in enc["offset_mapping"][-max_tokens:]], device=self.device)
            if not mask.any():
                raise ValueError("empty segment")
            caps, hooks = {}, []

            def keep(key, x):
                caps[key] = (x[0] if isinstance(x, tuple) else x)[0][mask].float().mean(0).cpu().numpy()

            for li, layer in enumerate(self.layers):
                if "resid" in want:
                    hooks.append(layer.register_forward_hook(lambda m, i, o, li=li: keep(("resid", li), o)))
                if "mlp" in want:
                    hooks.append(_child(layer, "down_proj").register_forward_pre_hook(
                        lambda m, i, li=li: keep(("mlp", li), i)))
            if "logp" in want:
                hooks.append(self.norm.register_forward_hook(lambda m, i, o: caps.__setitem__("final", o)))
            try:
                with torch.no_grad():
                    self.model(input_ids=torch.tensor([ids], device=self.device), logits_to_keep=1, use_cache=False)
                    if "logp" in want:  # log-probs of the segment's tokens, computed only at those positions
                        pos = torch.nonzero(mask).flatten()
                        pos = pos[pos > 0]
                        tgt = torch.tensor(ids, device=self.device)[pos]
                        h = caps.pop("final")[0]
                        lp = torch.cat([torch.log_softmax(self.head(h[pos[c:c + 1024] - 1]).float(), -1)
                                        .gather(1, tgt[c:c + 1024, None])[:, 0] for c in range(0, len(pos), 1024)])
                        out["logp"].append(float(lp.mean()))
            finally:
                for h in hooks:
                    h.remove()
            for k in ("resid", "mlp"):
                if k in want:
                    out[k].append(np.stack([caps[(k, li)] for li in range(len(self.layers))]))
            out["ntok"].append(int(mask.sum()))
        return {k: np.asarray(v) for k, v in out.items()}
