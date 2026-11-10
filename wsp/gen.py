"""Generation with vLLM and per-answer grading. The per-request seed depends only on (problem id, sample index,
salt), so a base run and an edited run of the same problems are paired."""
import time

from .common import stable_seed, write_jsonl
from .data import user_message
from .grade import score

CONTEXT_MARGIN = 2048  # prompt room on top of max_tokens


def _is_multimodal(model):
    from transformers import AutoConfig  # noqa: PLC0415
    return getattr(AutoConfig.from_pretrained(str(model)), "vision_config", None) is not None


def generate(model, items, profile, n=1, out=None, salt="s0", gpu_mem=0.90, max_tokens=None):
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    samp = dict(profile["sampling"], **({"max_tokens": max_tokens} if max_tokens else {}))
    # an edited model dir symlinks the base files, so it has the base's tokenizer and chat template
    kw = dict(model=str(model), dtype="bfloat16", max_model_len=samp["max_tokens"] + CONTEXT_MARGIN,
              gpu_memory_utilization=gpu_mem, enable_prefix_caching=True, seed=0)
    if _is_multimodal(model):
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    reqs = []
    for r in items:
        prompt = tok.apply_chat_template([{"role": "user", "content": user_message(r)}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=profile["thinking"])
        reqs += [(r, k, prompt) for k in range(n)]
    t0 = time.time()
    outs = llm.generate([p for _, _, p in reqs], [SamplingParams(seed=stable_seed(r["id"], k, salt), **samp)
                                                  for r, k, _ in reqs])
    dt = time.time() - t0
    rows = []
    for (r, k, _), o in zip(reqs, outs):
        c = o.outputs[0]
        rows.append({"id": r["id"], "k": k, "source": r.get("source"), "gold": r.get("answer"),
                     "n_tokens": len(c.token_ids), "finish": c.finish_reason,
                     **score(r, c.text, c.finish_reason, profile["thinking"]), "text": c.text})
    if out:
        write_jsonl(out, rows)
    ntok = sum(r["n_tokens"] for r in rows)
    print(f"[gen] {len(rows)} answers, {ntok} tokens in {dt:.0f}s ({ntok / max(dt, 1):.0f} tok/s)", flush=True)
    return rows


def main():
    import argparse  # noqa: PLC0415

    from . import data, profiles  # noqa: PLC0415
    from .common import read_json  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="generate and grade answers for a problem set")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--model", help="model dir (default: the profile's pinned base model)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ids", help="json list of practice problem ids")
    src.add_argument("--set", help="problem set spec, e.g. bench:math500")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--salt", default="s0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-tokens", type=int)
    a = ap.parse_args()
    prof = profiles.get(a.profile)
    items = [data.by_id()[i] for i in read_json(a.ids)] if a.ids else data.load(a.set)
    generate(a.model or profiles.base_dir(a.profile), items[:a.limit], prof, n=a.n, out=a.out, salt=a.salt, gpu_mem=a.gpu_mem,
             max_tokens=a.max_tokens)


if __name__ == "__main__":
    main()
