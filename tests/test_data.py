from wsp import data


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
