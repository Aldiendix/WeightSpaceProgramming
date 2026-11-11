# Decisions: qwen3-0.6b-nothink_syc

Same behavior and method as `qwen25-1.5b-instruct_syc` (read that first). Numbers: `RESULTS.md`.

1. Prevalence: with a wrong guess, 68% of MMLU-Pro answers take it (plain 26% -> 9% accuracy, lenient grader); the
   typical answer starts "The correct answer is: **X" with X the guess.
2. Residual directions and o_proj / MLP ablations null on DEV (-3.5 to +3.1); copy heads in L21-27, zeroing them
   destroys the plain answer (P(gold) without a guess 0.22 -> 0.11 at 16 heads).
3. Key direction (k_proj side "in", alpha 1), teacher-forced: L14-27 P(guess) 0.86 -> 0.49, weaker than in
   Qwen2.5. DEV128 x 2: K14_27a1 +6.6 +- 2.2 (guess share 67% -> 50%), K0_27 +2.7, random -0.8 +- 1.2.
4. **Frozen: `K14_27a1`**. Benchmarks: wrong guess +5.0 +- 1.5, held-out phrasings +3.8 +- 1.4; plain MMLU-Pro
   -3.0 +- 1.9, GSM8K +3.2 +- 1.7, IFEval -0.4 +- 1.5; right guess -15.6 (81.4 -> 65.8).
5. ARC-Challenge: wrong guess 26.2 -> 37.0 (+10.8 +- 1.8), plain 58.2 -> 59.8 (+1.6 +- 1.8).
