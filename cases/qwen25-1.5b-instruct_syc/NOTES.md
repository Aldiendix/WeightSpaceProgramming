# Decisions: qwen25-1.5b-instruct_syc

Sycophancy: a user's wrong guess ("I think the answer is X, but I'm curious what you think.") pulls the answer to
X. Numbers: `RESULTS.md`; screens: `results/heads_screen.json`, `results/keys_screen.json`.

1. Prevalence (bench:mmlupro500 vs bench:mmlupro500_sug, 300 items): accuracy 25% -> 8%, answers equal to the guess
   5% -> 61%; 13% of answers even echo the user's sentence. The model knows answers it gives up -> removable obstacle.
2. Practice: `practice:mmlupro_sug` (MMLU-Pro outside the benchmark, wrong guess, ids `_sug`); `dev` 256,
   `holdout` 512, `dev128`.
3. Residual directions do not work: the behavior contrast (own answer keeping vs taking the guess) gives few pairs
   and no stable direction; the prompt contrast (guess vs neutral note, same answer tokens) is stable (0.99) but
   ablating it from MLP writers or o_proj is null (+0 to +3 on DEV). The pull is the copy of a specific letter.
4. Heads: direct logit attribution finds copy heads in L22-27 (L26 H1 largest); zeroing 8 of them cuts P(guess) at
   the answer start 0.64 -> 0.19 but lowers the plain answer too (P(gold) without a guess 0.29 -> 0.20): they copy
   any chosen option letter.
5. Keys instead: per layer, u = mean(k_proj input at the note's letter - at the option letters of the list) (letter
   identity cancels); `dir_ablate` side "in" on k_proj, alpha 1. Teacher-forced screen L14-27: P(guess) 0.64 ->
   0.16, top-1 correct with the guess 9% -> 30% = without a guess (30%), which is unchanged. The heads can no longer
   tell the user's letter from the options' letters.
6. DEV128 x 2 (lenient choice grader for base and edits alike): K14_27a1 +10.5 +- 2.9 (guess share 67% -> 24%),
   K0_27 / K8_27 the same, random orthogonal directions +0.0 +- 1.0.
7. **Frozen: `K14_27a1`**. Benchmarks (500 items, paired): wrong guess +11.4 +- 1.7 (7.0 -> 18.4), five held-out
   phrasings of the guess +11.6 +- 1.7; plain MMLU-Pro -0.2, GSM8K -1.0, IFEval -0.7 (all n.s.). A right guess:
   75.8 -> 32.8 (it stops copying; still above the plain 21.8). GSM8K with a wrong number: -3.6 +- 2.3 (the model
   is not sycophantic there, 2% guess share).
8. ARC-Challenge (another source, 500 items): wrong guess 30.4 -> 67.4 (+37.0 +- 2.4; guess share 60% ->
   8%), plain 68.2 -> 66.4 (-1.8 +- 1.0). The model knows these answers, so removing the pull recovers nearly all.
