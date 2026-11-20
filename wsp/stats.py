"""Summary of a generation run, and paired comparison of two runs of the same problems and seeds."""
import collections
import math

from .common import read_jsonl


def _mean(v):
    v = list(v)
    return sum(v) / len(v) if v else None


def summarize(rows):
    fin = [r for r in rows if not r["cut"]]
    correct_len = sorted(r["n_tokens"] for r in rows if r["correct"])
    by, per = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        by[r.get("source")].append(r["correct"])
        per[r["id"]].append(r["correct"])
    k = max(len(v) for v in per.values()) if per else 0
    return {"n": len(rows), "acc": _mean(r["correct"] for r in rows), "cut_share": _mean(r["cut"] for r in rows),
            **({"samples": k, "pass_at_k": _mean(any(v) for v in per.values())} if k > 1 else {}),
            "finished": len(fin), "acc_when_finished": _mean(r["correct"] for r in fin),
            "mean_tokens": _mean(r["n_tokens"] for r in rows),
            "median_correct_tokens": correct_len[len(correct_len) // 2] if correct_len else None,
            "kinds": dict(collections.Counter(r["kind"] for r in rows)),
            "by_source": {k: _mean(v) for k, v in by.items()}}


def _gain_se(diff):
    """diff: {(problem, sample): edit - base} -> (gain, its SE, problems), both over per-problem means: every
    problem weighs the same however many samples it has (also when sets with different samples are pooled)."""
    per = collections.defaultdict(list)
    for (pid, _), x in diff.items():
        per[pid].append(x)
    pm = [_mean(v) for v in per.values()]
    g = _mean(pm) or 0.0
    se = math.sqrt(sum((x - g) ** 2 for x in pm) / max(len(pm) - 1, 1) / max(len(pm), 1))
    return g, se, len(pm)


def paired(base_rows, edit_rows):
    """Gain of edit over base on the shared (problem, sample) keys, overall and per source. SE is clustered by
    problem, since samples of one problem are not independent (with one sample per problem it is the ordinary paired
    SE). Rows of several sets may be passed together: they are pooled."""
    b = {(r["id"], r["k"]): r for r in base_rows}
    e = {(r["id"], r["k"]): r for r in edit_rows}
    keys = sorted(set(b) & set(e))
    diff = {k: int(e[k]["correct"]) - int(b[k]["correct"]) for k in keys}
    by = collections.defaultdict(list)
    for k in keys:
        by[b[k].get("source")].append(k)

    def part(ks):
        gain, se, problems = _gain_se({k: diff[k] for k in ks})
        return {"n": len(ks), "problems": problems,
                "base_acc": _mean(b[k]["correct"] for k in ks), "edit_acc": _mean(e[k]["correct"] for k in ks),
                "gain": gain, "se": se,
                "rescues": sum(diff[k] > 0 for k in ks), "losses": sum(diff[k] < 0 for k in ks)}
    return {**part(keys),
            "base_cut": _mean(b[k]["cut"] for k in keys), "edit_cut": _mean(e[k]["cut"] for k in keys),
            "by_source": {s: part(ks) for s, ks in by.items()}}


def main():
    import argparse  # noqa: PLC0415
    import json  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="summarize runs, or compare runs[1:] with the base runs[0]")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--paired", action="store_true")
    a = ap.parse_args()
    rows = [read_jsonl(p) for p in a.runs]
    res = [paired(rows[0], r) for r in rows[1:]] if a.paired else [summarize(r) for r in rows]
    print(json.dumps(res[0] if len(res) == 1 else res, indent=1))


if __name__ == "__main__":
    main()
