"""Model profiles = base models: HF repo pinned to a revision, chat mode and the model card's recommended sampling.
A profile name is the base id used everywhere else (edit specs, cases, submissions).

The base weights are read from $WSP_MODELS_DIR/<org>__<name>@<revision>/ when that directory exists (pre-mounted,
offline), otherwise from the HF cache (downloaded on first use)."""
import os
from pathlib import Path

QWEN35_THINK = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32768}
QWEN35_NOTHINK = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 8192}
QWEN3_THINK = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 32768}
QWEN3_NOTHINK = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": 1024}
R1_STYLE = {"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768}
INSTRUCT = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 8192}  # chat models without a card recommendation

QWEN35_08B = ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")
PROFILES = {
    "qwen35-0.8b-think": (*QWEN35_08B, True, QWEN35_THINK),
    "qwen35-0.8b-nothink": (*QWEN35_08B, False, QWEN35_NOTHINK),
    "qwen35-2b-think": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc", True, QWEN35_THINK),
    "qwen35-2b-nothink": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc", False, QWEN35_NOTHINK),
    "qwen35-4b-think": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", True, QWEN35_THINK),
    "qwen35-4b-nothink": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", False, QWEN35_NOTHINK),
    "qwen3-0.6b-think": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", True, QWEN3_THINK),
    "qwen3-0.6b-nothink": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", False, QWEN3_NOTHINK),
    "qwen3-1.7b-nothink": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", False, QWEN3_NOTHINK),
    "qwen3-1.7b-think": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", True, QWEN3_THINK),
    "r1-distill-1.5b": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562", True,
                        R1_STYLE),
    "smollm3-3b-think": ("HuggingFaceTB/SmolLM3-3B", "a07cc9a04f16550a088caea529712d1d335b0ac1", True, R1_STYLE),
    "phi4-mini-reasoning": ("microsoft/Phi-4-mini-reasoning", "0e3b1e2d02ee478a3743abe3f629e9c0cb722e0a", True,
                            dict(R1_STYLE, temperature=0.8)),
    "qwen3-4b-thinking-2507": ("Qwen/Qwen3-4B-Thinking-2507", "768f209d9ea81521153ed38c47d515654e938aea", True,
                               QWEN3_THINK),
    "qwen3-4b-instruct-2507": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554", False,
                              dict(QWEN3_NOTHINK, max_tokens=8192)),
    "nemotron-1.5b": ("nvidia/OpenReasoning-Nemotron-1.5B", "f7e9c457f57387ce4eefa856795a5afa0622ec56", True, R1_STYLE),
    "lfm25-1.2b-thinking": ("LiquidAI/LFM2.5-1.2B-Thinking", "fa4681e33c27d2861d1eacc9d29ccb68840344a0", True, R1_STYLE),
    "lfm25-1.2b-instruct": ("LiquidAI/LFM2.5-1.2B-Instruct", "b9d816ab510f9a28dca37cb182e9da8b0226e13c", False,
                            {"temperature": 0.1, "top_k": 50, "top_p": 0.1, "repetition_penalty": 1.05,
                             "max_tokens": 8192}),  # model card
    "qwen25-1.5b-instruct": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", False, INSTRUCT),
    "smollm2-1.7b-instruct": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "31b70e2e869a7173562077fd711b654946d38674", False,
                              dict(INSTRUCT, max_tokens=4096)),  # 8k context
    "granite33-2b-instruct": ("ibm-granite/granite-3.3-2b-instruct", "707f574c62054322f6b5b04b6d075f0a8f05e0f0", False,
                              INSTRUCT),
    "phi4-mini-instruct": ("microsoft/Phi-4-mini-instruct", "cfbefacb99257ffa30c83adab238a50856ac3083", False, INSTRUCT),
    "olmo2-1b-instruct": ("allenai/OLMo-2-0425-1B-Instruct", "48d788eca847d4d7548f375ad03d3c9312f6139e", False,
                          dict(INSTRUCT, max_tokens=2048)),  # 4k context
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
    kw = dict(revision=p["revision"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "tokenizer*"])
    try:  # the pinned revision never changes: use the local cache without asking the hub
        return Path(snapshot_download(p["hf"], local_files_only=True, **kw))
    except Exception:  # noqa: BLE001 - not cached yet
        return Path(snapshot_download(p["hf"], **kw))
