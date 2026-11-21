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
    two = [dict(r, source="t") if i >= 2 else r for i, r in enumerate(base)], \
        [dict(r, source="t") if i >= 2 else r for i, r in enumerate(edit)]
    by = paired(*two)["by_source"]  # each source has its own paired gain and SE
    assert (by["s"]["gain"], by["s"]["rescues"], by["t"]["gain"], by["t"]["losses"]) == (1.0, 2, -0.5, 1)
    assert by["t"]["se"] > 0 and by["s"]["problems"] == 2
    multi = [dict(r, id="p" + str(i // 2), k=i % 2) for i, r in enumerate(_rows([0, 1, 0, 0]))]
    assert summarize(multi)["pass_at_k"] == 0.5 and summarize(multi)["samples"] == 2


def test_practice_and_prompts():
    items = data.load("practice:all")
    assert len(items) > 10000
    r = next(x for x in items if x["task"] == "math")
    assert data.user_message(r) == r["problem"].rstrip() + \
        "\nPlease reason step by step, and put your final answer within \\boxed{}."
    a, b = data.balanced_sample(64, 1), data.balanced_sample(64, 1)
    assert [x["id"] for x in a] == [x["id"] for x in b] and len({x["source"] for x in a}) == len(data.PRACTICE)


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
    assert case.split_model("base@max_tokens=12000") == ("base", {"max_tokens": 12000})  # an int for vLLM


def test_puzzle_practice_is_solved_and_graded():
    """practice:bbhsynth: every generated kind, its own answer graded correct; an option is named by letter, with or
    without its text."""
    from wsp.puzzles import generators
    rows = data.load("practice:bbhsynth")
    assert len({r["id"] for r in rows}) == len(rows) and {r["subject"] for r in rows} == set(generators())
    assert all(score(r, f"so \\boxed{{{r['answer']}}}", "stop", False)["kind"] == "correct" for r in rows)
    item = next(r for r in rows if r["answer"] == "(B)")
    for box, ok in (("B", True), ("(B) the text", True), ("B. the text", True), ("A circle", False), ("(A)", False)):
        assert (score(item, f"\\boxed{{{box}}}", "stop", False)["kind"] == "correct") == ok, box


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


def test_set_rows_and_cols_write_a_neuron():
    import torch

    from wsp.edits import apply_ops
    W = {"g": torch.zeros(4, 3), "d": torch.zeros(3, 4)}
    apply_ops(W, [{"op": "set_rows", "tensor": "g", "rows": [2], "values": [[1.0, 2.0, 3.0]]},
                  {"op": "set_cols", "tensor": "d", "cols": [2], "values": [[5.0], [6.0], [7.0]]}], ".")
    assert W["g"][2].tolist() == [1.0, 2.0, 3.0] and W["g"][[0, 1, 3]].abs().sum() == 0
    assert W["d"][:, 2].tolist() == [5.0, 6.0, 7.0] and W["d"][:, [0, 1, 3]].abs().sum() == 0


def test_gate_fires_only_on_positives():
    import numpy as np

    from wsp import gate
    rng = np.random.default_rng(0)
    N = rng.normal(size=(2000, 16)) + 3.0  # ordinary positions (with a common offset, as real activations)
    P = rng.normal(size=(200, 16)) + 3.0
    P[:, 0] += 8.0  # the positives differ along one coordinate
    w = np.zeros(16)
    w[5] = 1.0
    u = gate.build(P, N, w)
    assert u["recall"] > 0.95 and u["neg_fire"] == 0

    def out(X):  # the unit's contribution to the residual
        return (gate.silu(X @ u["gate"]) * (X @ u["up"]))[:, None] * u["down"][None]
    assert np.abs(out(N)).max() < 0.05 and abs(out(P)[:, 5].mean() - 1.0) < 0.05


def test_think_override_switches_template_only():
    from wsp.case import split_model
    name, samp = split_model("base@think=0")
    assert name == "base" and samp == {"think": 0.0}


def test_always_on_unit_adds_a_constant():
    import numpy as np

    from wsp import gate
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3000, 32)) * 0.3 + 1.0
    w = np.arange(32.0)
    u = gate.always_on(X, w)
    out = (gate.silu(X @ u["gate"]) * (X @ u["up"]))[:, None] * u["down"][None]
    assert np.abs(out.mean(0) - w).max() < 0.05 * np.abs(w).max() and u["act_cv"] < 0.15


def test_grader_exception_scores_wrong(monkeypatch):
    from wsp import grade

    def boom(item, answer):
        raise TimeoutError("regex time limit")
    monkeypatch.setitem(grade.GRADERS, "ifeval", boom)
    r = grade.score({"task": "ifeval"}, "</think>answer", "stop", True)
    assert r["correct"] is False and r["kind"] == "grade_error"


def test_math_grading_off_the_main_thread_raises():
    """math-verify's time limit needs the main thread: elsewhere grading stops instead of scoring answers wrong."""
    import threading
    from wsp import grade
    item, out = {"task": "math", "answer": "1/2"}, {}

    def run():
        try:
            grade.score(item, "\\boxed{1/2}", "stop", False)
        except grade.GraderError as e:
            out["e"] = e
    t = threading.Thread(target=run)
    t.start()
    t.join()
    assert "e" in out and grade.score(item, "\\boxed{1/2}", "stop", False)["kind"] == "correct"


def test_practice_has_no_benchmark_question():
    from wsp import data
    fps = data.bench_fingerprints()
    assert len(fps) > 9000  # BBH: every test question, whichever tasks are evaluated
    hits = [r["id"] for r in data.load("practice:all") if data.fingerprint(r["problem"]) in fps]
    assert not hits, hits


def test_request_practice_is_graded_by_the_official_checkers():
    """practice:tuluif: one checker instruction each, with arguments the official checker accepts."""
    import pytest
    R = pytest.importorskip("lm_eval.tasks.ifeval.instructions_registry")
    rows = data.load("practice:tuluif")
    assert len(rows) == 1024 and len({r["id"] for r in rows}) == 1024
    for r in rows:
        assert r["task"] == "ifeval" and len(r["instruction_id_list"]) == len(r["kwargs"]) == 1, r["id"]
        iid, kw = r["instruction_id_list"][0], r["kwargs"][0]
        R.INSTRUCTION_DICT[iid](iid).build_description(**kw)
    item = next(r for r in rows if r["instruction_id_list"] == ["punctuation:no_comma"])
    assert score(item, "<think>x</think> A short text without that mark.", "stop", True)["correct"]
    assert not score(item, "<think>x</think> A text, with one.", "stop", True)["correct"]


def test_benchmarks_are_supplied_never_built(tmp_path, monkeypatch):
    """The toolkit holds no benchmark: bench:<name> reads only what the organizer supplied, and no module loads a
    dataset from elsewhere."""
    import pytest
    monkeypatch.setattr(data, "DATA", tmp_path)
    with pytest.raises(FileNotFoundError, match="not part of the toolkit"):
        data.load("bench:gsm8k500")
    item = {"id": "b1", "source": "x500", "task": "math", "problem": "1+1?", "answer": "2"}
    (tmp_path / "bench_x500.jsonl").write_text(json.dumps(item) + "\n")
    assert data.load("bench:x500") == [item]
    assert data.manifest("bench:x500")["n"] == 1
    src = "".join(p.read_text() for p in (data.REPO / "wsp").glob("*.py"))
    assert "load_dataset" not in src


def test_gpu_split_keeps_the_run_order(monkeypatch):
    import argparse

    from wsp import gen

    items = [{"id": f"p{i}"} for i in range(7)]
    parts = gen.split_items(items, 3)
    assert sorted(r["id"] for p in parts for r in p) == sorted(r["id"] for r in items)
    shards = [[{"id": r["id"], "k": k} for r in p for k in (1, 0)] for p in parts]
    assert [(r["id"], r["k"]) for r in gen.merge_rows(items, shards)] == [(f"p{i}", k) for i in range(7) for k in (0, 1)]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert gen.visible_gpus() == ["2", "3"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert gen.visible_gpus() == []
    smi = "0, 20000, 24564\n1, 1, 24564\n2, 3, 24564\n"  # GPU 0 holds another job
    monkeypatch.setattr(gen.subprocess, "run", lambda *a, **k: argparse.Namespace(stdout=smi))
    assert gen.free_gpus(["0", "1", "2"], 0.9) == ["1", "2"]
    try:  # nothing free: stop at once with what holds the GPU, not minutes later in the engine's start
        gen.free_gpus(["0"], 0.9)
        raise AssertionError("busy GPU accepted")
    except SystemExit as e:
        assert "GPU 0 busy" in str(e)


def test_compare_two_edits_on_shared_problems(tmp_path, monkeypatch, capsys):
    import argparse

    monkeypatch.setattr(case, "RUNS", tmp_path)
    (tmp_path / "c").mkdir()
    for name, correct in (("A__holdout256", [0, 1, 1, 0]), ("B__holdout", [1, 1, 1, 0, 1, 1])):
        (tmp_path / "c" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _rows(correct)))
    case.cmd_compare(argparse.Namespace(case="c", a="A", b="B", set=["holdout256"], set_b=["holdout"]))
    out = capsys.readouterr().out
    assert "4 answers" in out and "(4 problems; B better on 1, A on 0)" in out
    try:
        case.cmd_compare(argparse.Namespace(case="c", a="A", b="B", set=["holdout256"], set_b=None))
        raise AssertionError("missing run accepted")
    except SystemExit as e:
        assert "no run B__holdout256" in str(e)


def test_run_adds_samples_and_split_is_disjoint(tmp_path, monkeypatch):
    """`run --n` on an existing run generates only the new samples; `split` never reuses a problem of the case."""
    import argparse

    from wsp import gen
    monkeypatch.setattr(case, "CASES", tmp_path / "cases")
    monkeypatch.setattr(case, "RUNS", tmp_path / "runs")
    d = tmp_path / "cases" / "c"
    (d / "sets").mkdir(parents=True)
    ids = [r["id"] for r in data.balanced_sample(8, 0)]
    (d / "sets" / "dev.json").write_text(json.dumps(ids))
    (d / "case.json").write_text(json.dumps({"profile": "qwen35-0.8b-think", "subsets": {}, "parent": None}))
    calls, digest = [], ["d1"]

    def fake(model, items, prof, n=1, k0=0, **kw):
        start = {r["id"]: k0.get(r["id"], 0) if isinstance(k0, dict) else k0 for r in items}
        calls.append(sorted((r["id"], start[r["id"]]) for r in items))
        return [{"id": r["id"], "k": k, "correct": True, "cut": False, "n_tokens": 1, "kind": "correct",
                 "source": r["source"]} for r in items for k in range(start[r["id"]], n)]
    monkeypatch.setattr(gen, "generate", fake)
    monkeypatch.setattr(case, "model_path", lambda c, m: "unused")
    monkeypatch.setattr(case, "model_digest", lambda c, m: digest[0])
    run = argparse.Namespace(case="c", model="base", set="dev", n=1, force=False)
    path = tmp_path / "runs" / "c" / "base__dev.jsonl"

    def keys():
        return [(r["id"], r["k"]) for r in map(json.loads, path.read_text().splitlines())]
    case.cmd_run(run)
    case.cmd_run(argparse.Namespace(**{**vars(run), "n": 3}))
    case.cmd_run(argparse.Namespace(**{**vars(run), "n": 3}))  # complete: nothing generated
    assert calls == [sorted((i, 0) for i in ids), sorted((i, 1) for i in ids)]
    assert keys() == [(i, k) for i in ids for k in range(3)]
    extra = data.balanced_sample(10, 5, exclude=ids)[0]["id"]  # the set changes: one problem out, one in
    (d / "sets" / "dev.json").write_text(json.dumps(ids[1:] + [extra]))
    case.cmd_run(argparse.Namespace(**{**vars(run), "n": 3}))
    assert calls[-1] == [(extra, 0)] and keys() == [(i, k) for i in ids[1:] + [extra] for k in range(3)]
    digest[0] = "d2"  # the model changed (an edit rewritten under its name): every sample again
    case.cmd_run(argparse.Namespace(**{**vars(run), "n": 3}))
    assert len(calls[-1]) == len(ids) and all(k == 0 for _, k in calls[-1])
    (d / "sets" / "dev.json").write_text(json.dumps(ids))
    case.cmd_split(argparse.Namespace(case="c", name="fit", n=10, practice=None, seed=1))
    fit = json.loads((d / "sets" / "fit.json").read_text())
    assert len(fit) == 10 and not set(fit) & set(ids)
    try:
        case.set_ids("c", "nope")
        raise AssertionError("unknown set accepted")
    except SystemExit as e:
        assert "dev, fit" in str(e)
