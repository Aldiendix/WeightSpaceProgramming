"""Model profiles: HF id, chat mode and the model card's recommended sampling.
Always evaluate with the model card's settings, so a gain is not an artifact of a bad decoding choice."""

QWEN35_THINK = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32768}
QWEN35_NOTHINK = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 8192}
QWEN3_THINK = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 32768}
R1_STYLE = {"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768}

PROFILES = {
    "qwen35-0.8b-think": ("Qwen/Qwen3.5-0.8B", True, QWEN35_THINK),
    "qwen35-0.8b-nothink": ("Qwen/Qwen3.5-0.8B", False, QWEN35_NOTHINK),
    "qwen35-2b-think": ("Qwen/Qwen3.5-2B", True, QWEN35_THINK),
    "qwen35-4b-think": ("Qwen/Qwen3.5-4B", True, QWEN35_THINK),
    "qwen3-0.6b-think": ("Qwen/Qwen3-0.6B", True, QWEN3_THINK),
    "qwen3-1.7b-think": ("Qwen/Qwen3-1.7B", True, QWEN3_THINK),
    "r1-distill-1.5b": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", True, R1_STYLE),
    "smollm3-3b-think": ("HuggingFaceTB/SmolLM3-3B", True, R1_STYLE),
    "phi4-mini-reasoning": ("microsoft/Phi-4-mini-reasoning", True, dict(R1_STYLE, temperature=0.8)),
}


def get(name):
    hf, thinking, sampling = PROFILES[name]
    return {"name": name, "hf": hf, "thinking": thinking, "sampling": dict(sampling)}
