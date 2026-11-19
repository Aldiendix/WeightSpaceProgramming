# Decisions: qwen35-2b-think_loop2

Qwen3.5-2B thinking after the loop edit (`parent/L15L19_k64_a6`): a candidate base for a later rules version. Question: is
there edit-reachable headroom on the competition's five benchmarks (32k tokens, profile sampling)?

1. Prompt hints and sampling: hints concise / check / stepwise / trust all <= 0 on MATH500 (-3.0 to -5.5: any
   appended sentence hurts); MMLU-Pro concise +4.0 on 200 items but +0.9 +- 1.2 on the full 500 x 2; temperature
   0.3 / 0.45: n.s.
2. Failures: when 2B finishes a math answer it is right (MATH500 174 right / 1 wrong / 25 cut; AMC23 103 / 2 / 55).
   Cuts are long non-repetitive reasoning (54 of 55 AMC cuts), not loops; MMLU-Pro misses are mostly wrong answers.
3. Merging with Qwen3.5-2B-Base (W + lam (W_base - W)): lam +0.1 MATH500 -1.0, MMLU-Pro -1.0; lam -0.1 (amplify
   post-training) MATH500 -8.0 (cut 12.5% -> 20%), lam -0.2 -13.5 (cut 26%). Neither direction helps.
4. SAE features (Qwen-Scope, Qwen3.5-2B-Base residual SAEs, first 4k tokens of traces >= 4k long, cut vs finished
   correct; `results/sae_screen*.json`): features more active in traces that finish mark a result reached and
   closure ("= 15$.", "Everything seems consistent", |t| ~15); features more active in traces later cut mark
   backtracking and doubt ("* However / Wait / But", "Wait,", "= 0$?", "* Let's", |t| ~7). Removing the latter's
   decoder directions from the MLP writes (SAEA L12 f10204,f4935 at L8-12; SAEB L20 f4651 at L16-20; SAEC L8
   f15519,f17397 at L4-8): MATH500 -4.5 / -5.0 / -7.5, AMC23 -1.2 / -5.6, cuts rise. They are symptoms of hard
   problems, not removable habits.
Conclusion: every perturbation tried so far (hints, temperature, merges, feature removal) costs MATH500 points and
raises cuts; its misses are hard problems it does not finish and knowledge.
