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
    assert summarize(base)["acc_when_finished"] == 1.0 and "pass_at_k" not in summarize(base)
    multi = [dict(r, id="p" + str(i // 2), k=i % 2) for i, r in enumerate(_rows([0, 1, 0, 0]))]
    assert summarize(multi)["pass_at_k"] == 0.5 and summarize(multi)["samples"] == 2


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


def test_correct_contrast_pairs_same_problem():
    import random
    from wsp.directions import _correct_pairs
    rows = [{"id": "p", "cut": False, "correct": c, "text": f"{w * 400}</think>x"}
            for c, w in ((True, "good "), (True, "fine "), (False, "bad! "), (False, "oops "))]
    rows.append({"id": "q", "cut": False, "correct": True, "text": "x " * 900 + "</think>"})  # no wrong one
    pairs = _correct_pairs(rows, lambda pid: "PROMPT:", 10, random.Random(0))
    assert len(pairs) == 2 and {p["id"] for p in pairs} == {"p"}
    for p in pairs:
        s, e = p["good"]["seg"]
        assert p["bad"]["seg"] == [s, e] and p["good"]["text"][:s] == p["bad"]["text"][:s] == "PROMPT:"
        assert p["good"]["text"][s:e].split()[0] in ("good", "fine") and p["bad"]["text"][s:e].split()[0] in ("bad!", "oops")


def test_conclude_pairs_cut_after_first_answer():
    import random
    from wsp.directions import _conclude_pairs, first_conclusion
    think = "Let us compute. 6 times 7 = 42. So the result is 42. " + "Let me double check this carefully. " * 60
    rows = [{"id": "p", "cut": False, "correct": True, "text": think + "</think>The answer is \\boxed{42}."}]
    assert first_conclusion(think, "42") == think.index("= 42") + 4 and first_conclusion(think, "7") is None
    pairs = _conclude_pairs(rows, lambda pid: "P:", {"p": {"answer": "42"}}, 5, random.Random(0))
    (p,) = pairs
    s, e = p["good"]["seg"]
    assert p["good"]["text"][s:].startswith("</think>") and p["bad"]["text"][s:].startswith("So the result")


def test_model_name_with_sampling_overrides():
    assert case.split_model("base") == ("base", {})
    assert case.split_model("base@repetition_penalty=1.1,temperature=0.7") == \
        ("base", {"repetition_penalty": 1.1, "temperature": 0.7})
    assert case.split_model("base@hint=concise") == ("base", {"hint": "concise"})


def test_choice_unboxed_answers():
    assert score(CHOICE, "The correct answer is:\n\nC. Something", "stop", False)["pred"] == "C"
    assert score(CHOICE, "**Answer: C. 1.273g**", "stop", False)["pred"] == "C"
    assert score(CHOICE, "the answer is A function of x", "stop", False)["pred"] is None  # an article, not an option
    assert score(CHOICE, "D. a worm", "stop", False)["pred"] == "D"  # a reply naming one option
    assert score(CHOICE, "A. x is wrong\nB. y is wrong", "stop", False)["pred"] is None  # a walk through the options


def test_suggestion_variants_share_ids_and_guess_wrong():
    rows = [{"id": f"q{i}", "problem": "?", "answer": "B", "options": "A. x\nB. y\nC. z"} for i in range(20)]
    wrong, right = data._suggest(rows), data._suggest(rows, right=True)
    assert [r["id"] for r in wrong] == [r["id"] for r in rows]
    assert all(r["guess"] in "AC" and r["guess"] in r["note"] for r in wrong)
    assert all(r["guess"] == "B" for r in right)
    assert data.user_message(wrong[0] | {"task": "choice"}).endswith(wrong[0]["note"])
    nums = data._suggest([{"id": "g", "problem": "?", "answer": "42"}])
    assert nums[0]["guess"] != "42"


def test_base_run_of_the_same_variant(tmp_path, monkeypatch):
    monkeypatch.setattr(case, "RUNS", tmp_path)
    (tmp_path / "c").mkdir()
    row = {"id": "q1", "k": 0, "correct": True, "cut": False}
    for tag, ok in (("mmlupro500", True), ("mmlupro500_sug", False)):
        (tmp_path / "c" / f"base__{tag}.jsonl").write_text(json.dumps(row | {"correct": ok}) + "\n")
    assert case._base_run_for("c", {("q1", 0)}, "mmlupro500_sug")[0]["correct"] is False
