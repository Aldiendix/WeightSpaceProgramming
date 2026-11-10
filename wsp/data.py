"""Problem sets. Every item: {id, source, task, problem, answer, ...}; `task` selects the prompt and the grader
(see grade.py): "math" (\\boxed answer), "choice" (option letter), "ifeval" (official instruction checkers).

practice:<name>   problems the editor may use freely (selection, probing, DEV/HOLDOUT splits): GSM8K and MATH train,
                  shipped in data/practice/ (no test items)
bench:<name>      benchmark test sets, only for frozen evaluations (see BENCHMARKS)
private:<name>    organizer-only sets from $WSP_PRIVATE_DIR (never in the repo)
"""
import hashlib
import json
import os
import random
from pathlib import Path

from .common import DATA, REPO, read_jsonl, write_jsonl

PRACTICE_DIR = REPO / "data" / "practice"  # GSM8K / MATH train with answers (from the post's released logs)
PRACTICE = ("gsm8k_train", "math_train")

PROMPTS = {
    "math": "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    "choice": "{problem}\n\nOptions:\n{options}\n\nAnswer with the letter of the correct option, within \\boxed{{}}.",
    "ifeval": "{problem}",
}


def _hf(name, *args, **kw):
    from datasets import load_dataset  # noqa: PLC0415
    return load_dataset(name, *args, **kw)


def _math500():
    return [{"id": "math500_" + r["unique_id"].replace("/", "_").replace(".json", ""), "problem": r["problem"],
             "answer": r["answer"], "subject": r["subject"], "level": r["level"]}
            for r in _hf("HuggingFaceH4/MATH-500", split="test")]


def _gsm8k():
    return [{"id": f"gsm8k_test_{i}", "problem": r["question"], "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
            for i, r in enumerate(_hf("openai/gsm8k", "main", split="test"))]


def _qa(repo, split, prefix, q="question", a="answer"):
    return lambda: [{"id": f"{prefix}_{i}", "problem": r[q], "answer": str(r[a])} for i, r in enumerate(_hf(repo, split=split))]


def _mmlupro(n=500, seed=0):
    d = _hf("TIGER-Lab/MMLU-Pro", split="test")
    return [{"id": f"mmlupro_{d[i]['question_id']}", "problem": d[i]["question"], "answer": d[i]["answer"],
             "options": "\n".join(f"{chr(65 + j)}. {o}" for j, o in enumerate(d[i]["options"])),
             "category": d[i]["category"]} for i in random.Random(seed).sample(range(len(d)), n)]


def _ifeval():
    return [{"id": f"ifeval_{r['key']}", "problem": r["prompt"], "answer": None,
             "instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"]}
            for r in _hf("google/IFEval", split="train")]


# name -> (loader, task, number of items kept (None = all))
BENCHMARKS = {
    "math500": (_math500, "math", None),
    "gsm8k": (_gsm8k, "math", None),
    "gsm8k500": (_gsm8k, "math", 500),
    "amc23": (_qa("math-ai/amc23", "test", "amc23"), "math", None),
    "aime24": (_qa("HuggingFaceH4/aime_2024", "train", "aime24", q="problem"), "math", None),
    "mmlupro500": (_mmlupro, "choice", None),
    "ifeval": (_ifeval, "ifeval", None),
}


def _bench(name):
    loader, task, limit = BENCHMARKS[name]
    src = loader.__name__.strip("_") if limit else name
    path = DATA / f"bench_{src}.jsonl"
    if not path.exists():
        write_jsonl(path, loader())
    rows = read_jsonl(path)[:limit]
    for r in rows:
        r.setdefault("source", name)
        r.setdefault("task", task)
    return rows


def _practice(name):
    rows = read_jsonl(PRACTICE_DIR / f"{name}.jsonl.gz")
    for r in rows:
        r.setdefault("task", "math")
    return rows


def _private(name):
    """Organizer-only sets: $WSP_PRIVATE_DIR/<name>.jsonl with items like the benchmarks' (incl. `task`)."""
    rows = read_jsonl(Path(os.environ["WSP_PRIVATE_DIR"]) / f"{name}.jsonl")
    for r in rows:
        r.setdefault("source", name)
        r.setdefault("task", "math")
    return rows


def load(spec):
    """'bench:<name>', 'private:<name>', or 'practice:<name>' / 'practice:all'."""
    kind, _, name = spec.partition(":")
    if kind == "bench":
        return _bench(name)
    if kind == "private":
        return _private(name)
    if kind == "practice":
        return [r for n in (PRACTICE if name == "all" else (name,)) for r in _practice(n)]
    raise ValueError(f"unknown problem set {spec}")


def by_id(spec="practice:all"):
    return {r["id"]: r for r in load(spec)}


def balanced_sample(n, seed, exclude=()):
    """n practice problems, equally many from each practice source, none in `exclude`."""
    ex, rng = set(exclude), random.Random(seed)
    out = []
    for name in PRACTICE:
        out += rng.sample([r for r in _practice(name) if r["id"] not in ex], n // len(PRACTICE))
    return out


def user_message(item):
    return PROMPTS[item.get("task", "math")].format(**{**item, "problem": item["problem"].rstrip()})


def manifest(spec):
    """Freezable identity of a problem set: per item id + sha256 of its canonical JSON, and the sha256 of the list."""
    items = [{"id": r["id"], "sha256": hashlib.sha256(json.dumps(r, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
             for r in load(spec)]
    return {"set": spec, "n": len(items), "items": items,
            "sha256": hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()}
