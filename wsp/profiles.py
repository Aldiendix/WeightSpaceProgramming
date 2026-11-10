"""Model profiles = base models: HF repo pinned to a revision, chat mode and the model card's recommended sampling.
A profile name is the base id used everywhere else (edit specs, cases, submissions).

The base weights are read from $WSP_MODELS_DIR/<org>__<name>@<revision>/ when that directory exists (pre-mounted,
offline), otherwise from the HF cache (downloaded on first use)."""
import os
from pathlib import Path

QWEN35_THINK = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32768}
QWEN35_NOTHINK = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 8192}
QWEN3_THINK = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 32768}
R1_STYLE = {"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768}

QWEN35_08B = ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")
PROFILES = {
    "qwen35-0.8b-think": (*QWEN35_08B, True, QWEN35_THINK),
    "qwen35-0.8b-nothink": (*QWEN35_08B, False, QWEN35_NOTHINK),
    "qwen35-2b-think": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc", True, QWEN35_THINK),
    "qwen35-4b-think": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", True, QWEN35_THINK),
    "qwen3-0.6b-think": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", True, QWEN3_THINK),
    "qwen3-1.7b-think": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", True, QWEN3_THINK),
    "r1-distill-1.5b": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562", True,
                        R1_STYLE),
    "smollm3-3b-think": ("HuggingFaceTB/SmolLM3-3B", "a07cc9a04f16550a088caea529712d1d335b0ac1", True, R1_STYLE),
    "phi4-mini-reasoning": ("microsoft/Phi-4-mini-reasoning", "0e3b1e2d02ee478a3743abe3f629e9c0cb722e0a", True,
                            dict(R1_STYLE, temperature=0.8)),
}


def get(name):
    hf, revision, thinking, sampling = PROFILES[name]
    return {"name": name, "hf": hf, "revision": revision, "thinking": thinking, "sampling": dict(sampling)}


def base_dir(name):
    """Local directory with the profile's pinned base model files."""
    p = get(name)
    local = Path(os.environ.get("WSP_MODELS_DIR", "/nonexistent")) / f"{p['hf'].replace('/', '__')}@{p['revision']}"
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    return Path(snapshot_download(p["hf"], revision=p["revision"],
                                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "tokenizer*"]))
