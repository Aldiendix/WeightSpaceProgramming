"""Grading of one generated answer by task, and failure diagnosis.

Graders (item["task"]):  math    last \\boxed{} of the final answer, exact match or math-verify equivalence
                         choice  option letter from \\boxed{}, else the last "answer is (X)" / "Answer: **X." (a letter
                                 followed by a delimiter, so "the answer is A function" does not count)
                         exact   \\boxed{} answer equal to the target after normalization (case, brackets, final period)
                         ifeval  official IFEval checkers (lm-eval), strict prompt level
The final answer is the text after </think> in thinking mode; an answer whose thinking never closed has none.
Failure kinds: correct | finished_wrong | finished_no_answer | cut_loop (hit the token limit in a loop) | cut_other |
grade_error (the grader raised on this answer, e.g. a regex time limit in an IFEval checker: scored wrong).
Grade on the main thread: math-verify's time limit uses signal.alarm (GraderError elsewhere, never a silent "wrong").
"""
import re
import threading

LOOP_DUP = 0.5  # tail duplicate-n-gram share above which a cut answer counts as a loop


def last_boxed(text):
    """Content of the last \\boxed{...} (balanced braces), or None."""
    i = text.rfind("\\boxed")
    while i >= 0:
        j = i + len("\\boxed")
        while j < len(text) and text[j] == " ":
            j += 1
        if j < len(text) and text[j] == "{":
            depth = 0
            for k in range(j, len(text)):
                depth += {"{": 1, "}": -1}.get(text[k], 0)
                if depth == 0:
                    return text[j + 1:k].strip()
        i = text.rfind("\\boxed", 0, i)
    return None


def answer_part(text):
    return text.split("</think>")[-1]


def tail_dup_ratio(text, n=8, tail_words=1500):
    """Share of word n-grams in the tail that already occurred earlier in the tail: ~0 normally, ~1 in a loop."""
    w = text.split()[-tail_words:]
    if len(w) < n * 4:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1 - len(set(grams)) / len(grams)


_LAZY = {}


def _lazy(mod):
    if mod not in _LAZY:
        _LAZY[mod] = __import__(mod, fromlist=["_"])
    return _LAZY[mod]


class GraderError(RuntimeError):
    """The grader cannot run here: not the answer's fault, so it is raised instead of scored."""


def _norm(s):
    s = s.strip().strip("$").replace(",", "").replace("\\!", "").replace(" ", "")
    return re.sub(r"\\text\{([^}]*)\}", r"\1", s).rstrip(".")


def _math(item, answer):
    if threading.current_thread() is not threading.main_thread():
        raise GraderError("math answers are graded on the main thread (math-verify's time limit uses signal.alarm)")
    pred = last_boxed(answer)
    if pred is None:
        return None, False
    gold = item["answer"]
    if _norm(pred) == _norm(gold):
        return pred, True
    mv = _lazy("math_verify")
    try:
        ok = mv.verify(mv.parse("\\boxed{" + gold + "}", parsing_timeout=5),
                       mv.parse("\\boxed{" + pred + "}", parsing_timeout=5), timeout_seconds=5)
    except Exception:  # noqa: BLE001 - unparsable answers are wrong
        ok = False
    return pred, bool(ok)


def _choice(item, answer):
    box = last_boxed(answer)
    m = re.match(r"\s*\(?\s*(?:\\text\{)?\s*([A-J])\b", box) if box else None
    found = re.findall(r"[Aa]nswer(?: is)?\s*:?\s*\**\s*\(?([A-J])(?=[.):*,]|\s*$|\s*\n)", answer)
    lead = re.findall(r"^\s*\**\(?([A-J])[.)]\**(?:\s|$)", answer, re.M)  # a reply naming one option: "D. a worm"
    pred = m.group(1) if m else (found[-1] if found else (lead[0] if len(lead) == 1 else None))
    return pred, pred == item["answer"]


def _ifeval(item, answer):
    U = _lazy("lm_eval.tasks.ifeval.utils")
    doc = {"key": 0, "prompt": item["problem"], "instruction_id_list": item["instruction_id_list"], "kwargs": item["kwargs"]}
    res = U.process_results(doc, [answer.strip()])
    return res["inst_level_strict_acc"], bool(res["prompt_level_strict_acc"])


def _exact(item, answer):
    pred = last_boxed(answer)

    def norm(x):
        x = re.sub(r"\\text\{([^}]*)\}", r"\1", x).strip().strip(".").lower()
        x = re.sub(r"^\((.*)\)$", r"\1", x).strip()
        return " ".join(x.replace(",", " ").split())  # word lists: "a, b" == "a b"
    if pred is not None and re.fullmatch(r"\([A-R]\)", item["answer"]):  # an option, named with its text too
        m = re.match(r"\s*(?:\\text\{)?\s*(?:\(([A-R])\)|([A-R])(?:[.)]?\s*$|[.)]\s))", pred)
        return pred, bool(m) and (m.group(1) or m.group(2)) == item["answer"][1]
    return pred, pred is not None and norm(pred) == norm(item["answer"])


GRADERS = {"math": _math, "choice": _choice, "exact": _exact, "ifeval": _ifeval}


def score(item, text, finish_reason, thinking):
    cut = finish_reason == "length"
    closed = (not thinking) or "</think>" in text
    err = False
    try:
        pred, correct = GRADERS[item.get("task", "math")](item, answer_part(text)) if closed else (None, False)
    except GraderError:
        raise
    except Exception:  # noqa: BLE001 - a checker that fails on an answer (e.g. a regex time limit) scores it wrong
        pred, correct, err = None, False, True
    dup = tail_dup_ratio(text)
    ws_tail = len(text) > 2000 and len(text[-2000:].strip()) < 200
    if err:
        kind = "grade_error"
    elif correct:
        kind = "correct"
    elif cut:
        kind = "cut_loop" if dup > LOOP_DUP or ws_tail else "cut_other"
    else:
        kind = "finished_no_answer" if pred is None else "finished_wrong"
    return {"pred": pred, "correct": correct, "cut": cut, "think_closed": closed, "tail_dup": round(dup, 3),
            "kind": kind}
