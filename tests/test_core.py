import json

from wsp import case, data
from wsp.grade import score, tail_dup_ratio
from wsp.stats import paired, summarize

MATH = {"id": "m1", "task": "math", "problem": "1+1?", "answer": "2"}
CHOICE = {"id": "c1", "task": "choice", "problem": "?", "answer": "C"}


def test_math_and_choice():
    assert score(MATH, "<think>x</think> so \\boxed{2}", "stop", True)["correct"]
    assert not score(MATH, "<think>x</think> \\boxed{3}", "stop", True)["correct"]
    assert score(MATH, "no think block \\boxed{2}", "stop", True)["kind"] == "finished_no_answer"  # thinking never closed
    assert score(CHOICE, "<think>x</think> \\boxed{(C)}", "stop", True)["correct"]
    assert score(CHOICE, "<think>x</think> The answer is (C).", "stop", True)["correct"]


def test_loop_detection():
    loop = "<think>" + "wait let me check this again carefully. " * 400
    r = score(MATH, loop, "length", True)
    assert r["kind"] == "cut_loop" and tail_dup_ratio(loop) > 0.9
    assert tail_dup_ratio(" ".join(f"w{i}" for i in range(2000))) == 0


def _rows(correct):
    return [{"id": f"p{i}", "k": 0, "correct": c, "cut": not c, "n_tokens": 10, "kind": "correct" if c else "cut_loop",
             "source": "s"} for i, c in enumerate(correct)]


def test_paired():
    base, edit = _rows([0, 0, 1, 1]), _rows([1, 1, 1, 0])
    p = paired(base, edit)
    assert (p["base_acc"], p["edit_acc"], p["rescues"], p["losses"]) == (0.5, 0.75, 2, 1)
    assert abs(p["gain"] - 0.25) < 1e-9 and p["se"] > 0
    assert summarize(base)["acc_when_finished"] == 1.0


def test_practice_and_prompts():
    items = data.load("practice:all")
    assert len(items) > 10000 and all(r["task"] == "math" for r in items[:10])
    r = items[0]
    assert data.user_message(r) == r["problem"].rstrip() + \
        "\nPlease reason step by step, and put your final answer within \\boxed{}."
    a, b = data.balanced_sample(64, 1), data.balanced_sample(64, 1)
    assert [x["id"] for x in a] == [x["id"] for x in b] and len({x["source"] for x in a}) == 2


def test_manifest_is_deterministic():
    assert data.manifest("practice:gsm8k_train")["sha256"] == data.manifest("practice:gsm8k_train")["sha256"]


def test_subsets_are_derived_from_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(case, "CASES", tmp_path)
    d = tmp_path / "c"
    (d / "sets").mkdir(parents=True)
    dev = [f"gsm8k_train_{i}" for i in range(4)] + [f"math_train_algebra_{i}" for i in range(4)]
    (d / "sets" / "dev.json").write_text(json.dumps(dev))
    (d / "case.json").write_text(json.dumps({"profile": "p", "subsets": {"dev4": "dev:4"}}))
    assert case.set_ids("c", "dev") == dev
    assert case.set_ids("c", "dev4") == ["gsm8k_train_0", "gsm8k_train_1", "math_train_algebra_0", "math_train_algebra_1"]
