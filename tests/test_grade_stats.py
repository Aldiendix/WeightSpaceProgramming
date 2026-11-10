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
