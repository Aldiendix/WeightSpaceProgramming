"""Forced recall: each answer's thinking starts with the puzzle family's rule, written as the model's own recollection
(it refers to no question and to no hint), and the model continues freely with the normal sampling settings. Answers
(the rule sentence included) are graded as usual, so the right ones are natural text after the PLAIN question.
    python recall_gen.py CASE MODEL SET OUT.jsonl --n 6 [--shard i/k]     (one GPU per process: CUDA_VISIBLE_DEVICES=i)"""
import sys

PREFIX = {
    "hyperbaton": "First, recall the rule: in English, adjectives before a noun go in this order: opinion, size, age, "
                  "shape, color, origin, material, purpose (for example, \"a lovely little old rectangular green "
                  "French silver whittling knife\"). I will classify each adjective by this list and pick the option "
                  "whose adjectives follow it.\n\n",
    "temporal_sequences": "First, recall how this kind of question works: the person was seen doing something else in "
                          "every period of the day except one. They could only have gone during the period, after "
                          "waking up and before the place closed, in which nobody saw them. So I need to find that "
                          "gap.\n\n",
    "object_counting": "First, recall how to count: I count only the things of the asked kind and add up their "
                       "quantities (for example, \"three radishes\" counts as 3). The kinds are separate: fruits, "
                       "vegetables (pumpkins, leeks, garlic, cauliflower and lettuce included), animals, musical "
                       "instruments, and objects (furniture, household appliances and cars).\n\n",
}


def main():
    from vllm import LLM, SamplingParams
    from wsp import case as C, data, profiles
    from wsp.common import stable_seed, write_jsonl
    from wsp.grade import score
    a = sys.argv[1:]
    case, model, s, out = a[:4]
    n = int(a[a.index("--n") + 1]) if "--n" in a else 6
    pre = PREFIX
    fams = a[a.index("--fams") + 1].split(",") if "--fams" in a else list(pre)
    i, k = map(int, a[a.index("--shard") + 1].split("/")) if "--shard" in a else (0, 1)
    _, items = C.set_items(case, s)
    items = [r for r in items if r.get("subject") in fams][i::k]
    prof = profiles.get(C.conf(case)["profile"])
    samp = dict(prof["sampling"])
    llm = LLM(model=str(C.model_path(case, model)), dtype="bfloat16", max_model_len=samp["max_tokens"] + 2048,
              gpu_memory_utilization=0.90, enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    reqs = []
    for r in items:
        p = tok.apply_chat_template([{"role": "user", "content": data.user_message(r)}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=prof["thinking"]) + pre[r["subject"]]
        reqs += [(r, j, p) for j in range(n)]
    outs = llm.generate([p for *_, p in reqs], [SamplingParams(seed=stable_seed(r["id"], j, "pf"), **samp)
                                                 for r, j, _ in reqs])
    rows = []
    for (r, j, _), o in zip(reqs, outs):
        c = o.outputs[0]
        text = pre[r["subject"]] + c.text
        rows.append({"id": r["id"], "k": j, "source": r.get("source"), "gold": r.get("answer"),
                     "n_tokens": len(c.token_ids), "finish": c.finish_reason,
                     **score(r, text, c.finish_reason, prof["thinking"]), "text": text})
    write_jsonl(out, rows)
    print(f"{len(rows)} answers, right {sum(x['correct'] for x in rows) / len(rows):.3f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 5 or sys.argv[1] in ("-h", "--help"):
        print(__doc__); sys.exit(0)
    main()
